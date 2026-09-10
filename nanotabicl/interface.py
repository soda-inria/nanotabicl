"""scikit-learn compatible wrappers around a trained checkpoint. They apply the same preprocessing as the prior and
average `n_estimators` predictions made with random feature orders (and class orders). Features must be numeric
(encode categories as integers); NaNs are imputed with the training mean."""
import functools

import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from .config import Config, to_config
from .data import clip_outliers, preprocess
from .runtime import build_model, resolve_device


def load_model(path: str, device: str = "auto") -> tuple[torch.nn.Module, Config]:
    device = resolve_device(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = to_config(ckpt["config"])
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg


class NanoTabICLEstimator(BaseEstimator):
    def __init__(self, model="runs/default/latest.pt", device: str = "auto", n_estimators: int = 8, random_state: int = 0):
        self.model = model  # checkpoint path or nn.Module
        self.device = device
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        if len(X) != len(y) or len(X) < 2:
            raise ValueError("X and y must contain atleast two samples and the same number of samples")
        self.model_ = (load_model(self.model, self.device)[0] if isinstance(self.model, str)
                       else self.model.to(resolve_device(self.device)).eval())
        self.X_train_ = np.asarray(X, dtype=np.float32)
        self.y_train_ = self._encode_y(np.asarray(y))
        return self

    @torch.no_grad()
    def _predict(self, X) -> torch.Tensor:  # (n_test, out_dim) average of the ensemble members' outputs
        device, n_train = next(self.model_.parameters()).device, len(self.X_train_)
        x = torch.as_tensor(np.concatenate([self.X_train_, np.asarray(X, dtype=np.float32)]), device=device)
        x = preprocess(torch.where(x.isnan(), x[:n_train].nanmean(dim=0), x), n_train)
        y = torch.as_tensor(self.y_train_, dtype=torch.float32, device=device)
        rng = torch.Generator().manual_seed(self.random_state)
        members = [self._member(x[:, torch.randperm(x.shape[1], generator=rng)][None], y, rng)
                   for _ in range(self.n_estimators)]
        return torch.stack(members).mean(dim=0)


class NanoTabICLClassifier(ClassifierMixin, NanoTabICLEstimator):
    def _encode_y(self, y: np.ndarray) -> np.ndarray:
        self.classes_, y = np.unique(y, return_inverse=True)
        if len(self.classes_) > self.model_.out_mlp[-1].out_features:
            raise ValueError(f"Model supports at most {self.model_.out_mlp[-1].out_features} classes")
        return y

    def _member(self, x, y, rng) -> torch.Tensor:  # class probabilities, predicted with a random class order
        perm = torch.randperm(len(self.classes_), generator=rng).to(x.device)
        logits = self.model_(x, perm[y.long()].float()[None])[0, :, :len(self.classes_)].float()
        return torch.softmax(logits, dim=-1)[:, perm]

    def predict_proba(self, X) -> np.ndarray:
        return self._predict(X).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


class NanoTabICLRegressor(RegressorMixin, NanoTabICLEstimator):
    def _encode_y(self, y: np.ndarray) -> np.ndarray:  # same target preprocessing as the prior, keeping its statistics
        y = torch.as_tensor(y, dtype=torch.float32)[:, None]
        clipped = clip_outliers(y)
        self.y_mean_, self.y_std_ = clipped.mean().item(), clipped.std().item()
        return preprocess(y).squeeze(-1).numpy()

    def _member(self, x, y, rng) -> torch.Tensor:  # quantiles of the standardized target
        return self.model_(x, y[None])[0].float()

    def predict_quantiles(self, X, alphas=(0.1, 0.5, 0.9)) -> np.ndarray:  # (n_test, len(alphas))
        quantiles = self._predict(X)  # predicted at levels linspace(0, 1, n_quantiles + 2)[1:-1]
        idxs = torch.as_tensor(alphas, device=quantiles.device) * (quantiles.shape[1] + 1) - 1
        return (quantiles[:, idxs.round().long().clamp(0, quantiles.shape[1] - 1)] * self.y_std_ + self.y_mean_).cpu().numpy()

    def predict(self, X) -> np.ndarray:  # median of the predicted quantiles
        return self.predict_quantiles(X, alphas=(0.5,))[:, 0]
