"""Muon optimizer (K. Jordan et al.), applied to all parameters as in the TabICL training code."""
import math
import torch


def orthogonalize(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximates U V^T for g = U S V^T with a quintic Newton-Schulz iteration (coefficients from K. Jordan)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.T if g.shape[0] > g.shape[1] else g
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.T if g.shape[0] > g.shape[1] else x


class Muon(torch.optim.Optimizer):
    """SGD with Nesterov momentum whose update is orthogonalized per parameter. Every parameter is reshaped to 2D,
    so biases and norm weights become normalized vectors. The step size is scaled to match the RMS of an AdamW update
    (Moonlight), so `lr` is on the AdamW scale."""

    def __init__(self, params, lr: float, weight_decay: float = 0.0, momentum: float = 0.95,
                 matched_adamw_rms: float = 0.2, ns_steps: int = 5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      matched_adamw_rms=matched_adamw_rms, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if "momentum_buffer" not in self.state[p]:
                    self.state[p]["momentum_buffer"] = torch.zeros_like(p)
                buf = self.state[p]["momentum_buffer"].mul_(group["momentum"]).add_(p.grad)
                g = p.grad.add(buf, alpha=group["momentum"]).reshape(len(p), -1)  # Nesterov momentum, 2D
                update = orthogonalize(g, group["ns_steps"]).view_as(p)
                p.mul_(1 - group["lr"] * group["weight_decay"])  # decoupled weight decay
                p.add_(update, alpha=-group["lr"] * group["matched_adamw_rms"] * math.sqrt(max(g.shape)))
