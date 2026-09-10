import numpy as np
import pytest
import torch

from nanotabicl import load_config
from nanotabicl.data import PriorDataset
from nanotabicl.eval import evaluate
from nanotabicl.interface import NanoTabICLClassifier, NanoTabICLRegressor
from nanotabicl.train import build_model, train

TINY = ["model.embed_dim=16", "model.col_num_blocks=1", "model.row_num_blocks=1", "model.icl_num_blocks=2",
        "model.col_nhead=2", "model.row_nhead=2", "model.icl_nhead=2", "model.n_cls_rows=8",
        "data.micro_batch_size=2", "data.min_seq_len=32", "data.max_seq_len=64", "data.max_features=6", "data.num_workers=0",
        "data.filter_unpredictable=false", "optim.max_steps=2", "optim.accum_steps=2", "save_every=1", "log_every=1"]


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_train_resume_eval(tmp_path, task):
    cfg = load_config([], TINY + [f"data.task={task}", f"out_dir={tmp_path}"])
    train(cfg)
    assert (tmp_path / "latest.pt").exists()
    cfg.optim.max_steps = 3
    model = train(cfg)  # resumes from latest.pt and trains one more step
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["step"] == 3
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 3
    scores = evaluate(model.eval(), task, max_rows=100)
    assert len(scores) > 0 and all(np.isfinite(s) for s in scores.values())


def test_config_overrides():
    cfg = load_config(["configs/small.yaml", "configs/stage2.yaml"], ["optim.lr=1e-3", "model.ln_bias=false"])
    assert cfg.optim.lr == 1e-3 and cfg.model.ln_bias is False  # overrides win
    assert cfg.data.max_seq_len == 10240 and cfg.data.micro_batch_size == 1  # from stage2.yaml
    assert cfg.model.embed_dim == 64 and cfg.data.max_features == 20  # from small.yaml
    with pytest.raises(KeyError):
        load_config([], ["optim.typo=1"])


def test_prior_batch():
    overrides = ["data.micro_batch_size=3", "data.min_seq_len=50", "data.max_seq_len=60", "data.max_features=7"]
    x, y, n_train = PriorDataset(load_config([], overrides).data).sample_batch()
    assert x.shape[:2] == y.shape and x.shape[0] == 3 and 50 <= x.shape[1] <= 60 and 1 <= x.shape[2] <= 7
    assert torch.isfinite(x).all() and torch.isfinite(y).all() and 1 <= n_train < x.shape[1]
    for yi in y:  # train and test rows contain the same classes
        assert torch.equal(yi[:n_train].unique(), yi[n_train:].unique()) and yi.unique().numel() >= 2


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_sklearn_interface(task):
    cfg = load_config([], TINY + [f"data.task={task}"])
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(40, 5)), rng.integers(0, 3, size=40) + 5
    x[0, 0] = np.nan  # imputed with the training mean
    x[:, 4] = 1.0  # constant feature is dropped
    if task == "classification":
        clf = NanoTabICLClassifier(model=build_model(cfg), device="cpu", n_estimators=3).fit(x[:30], y[:30])
        proba = clf.predict_proba(x[30:])
        assert proba.shape == (10, 3) and np.allclose(proba.sum(axis=1), 1) and set(clf.predict(x[30:])) <= {5, 6, 7}
        assert np.allclose(proba, clf.predict_proba(x[30:]))  # seeded ensemble is deterministic
    else:
        reg = NanoTabICLRegressor(model=build_model(cfg), device="cpu").fit(x[:30], y[:30].astype(float))
        assert reg.predict(x[30:]).shape == (10,) and reg.predict_quantiles(x[30:]).shape == (10, 3)
