# NanoTabICL: a minimal TabICLv2 implementation

| [Full TabICLv2 code](https://github.com/soda-inria/tabicl) | [TabICLv2 Paper](https://arxiv.org/abs/2602.11139) |
|----------------------------------------------------|------------------------|

This repository provides a short implementation of the [TabICLv2](https://arxiv.org/abs/2602.11139) architecture (<170 LOC)
and a slightly simplified implementation of the TabICLv2 prior (<330 LOC).
For using our pre-trained TabICLv2 model,
please visit the [main repository](https://github.com/soda-inria/tabicl). This repository also contains a compact
pre-training loop for experiments and reproduction attempts.

Compared to [nanoTabPFN](https://github.com/automl/nanoTabPFN),

- the model is TabICLv2, not TabPFN,
- we implement the full model (including RoPE, QASSMax, etc.) with speed optimizations
  (but without the inference wrappers + memory optimizations from the main repository),
- we implement regression as well,
- we provide a (slightly simplified but still well-performing) prior for dataset generation,
- we provide a small sklearn-compatible inference interface,
- we provide a minimal pre-training loop, but not the distributed and memory-optimized training system from the
  full repository.

Note that this repo uses LayerNorm with bias, which is used by the classification checkpoint of TabICLv2,
while the regression checkpoint of TabICLv2 uses LayerNorm without bias.

## Model usage

```python
from nanotabicl import NanoTabICLv2
import torch

batch_size, n_train, n_test, n_cols = 2, 32, 16, 10
model = NanoTabICLv2(max_classes=10, out_dim=10)  # original model size
X_train_and_test = torch.randn(batch_size, n_train+n_test, n_cols)
y_train = torch.randint(10, size=(batch_size, n_train)).float()
y_test_pred_logits = model(X_train_and_test, y_train)

# if you want a smaller model + regression with 999 quantiles instead
# warning: for regression, you need to standardize y yourself (and backtransform the output)
model = NanoTabICLv2(max_classes=0, out_dim=999, embed_dim=96,
                 col_num_blocks=2, row_num_blocks=2, icl_num_blocks=4,
                 col_nhead=4, row_nhead=4, icl_nhead=4)
y_train = torch.randn(batch_size, n_train)
y_test_pred_quantiles = model(X_train_and_test, y_train)
```

Note that `X_train_and_test` is standardized inside the model (based on train only).
We do not include other preprocessing options from TabICLv2
since they are normally not part of the architecture.

For a checkpoint, the sklearn-compatible wrappers apply the prior's preprocessing and average predictions over
random feature/class orders:

```python
from nanotabicl import NanoTabICLClassifier, NanoTabICLRegressor

clf = NanoTabICLClassifier(model="runs/small/latest.pt", n_estimators=8).fit(X_train, y_train)
proba = clf.predict_proba(X_test)

reg = NanoTabICLRegressor(model="runs/regression/latest.pt").fit(X_train, y_train)
quantiles = reg.predict_quantiles(X_test, alphas=(0.1, 0.5, 0.9))
pred = reg.predict(X_test)  # median prediction
```

## Training and evaluation


> **Warning:** the complete three-stage recipe is far too costly for a casual experiment. Stage 1 is 500,000
> optimizer steps, stage 2 adds 40,000 steps with sequences up to 10,240 rows, and stage 3 adds 10,000 steps with
> sequences up to 60,000 rows. It requires substantial GPU time, CPU data-generation capacity, storage, and memory. These configuration files are only included to replicate TabICLv2 actual setup.
> Use `configs/small.yaml` or override `optim.max_steps`, sequence lengths, and model size for quick experiments.


Install the package in editable mode (the optional dependencies add pytest, matplotlib, and Weights & Biases):

```bash
python -m pip install -e ".[dev]"
# if you get an error about Go not being installed to build wandb, try instead
python -m pip install -e ".[dev]" "wandb==0.21.4" 
```

To run the small configuration first, use the following commands. While it is intended as a development run, it still performs relatively well (see figures below)

```bash
python -m nanotabicl.train configs/small.yaml
python -m nanotabicl.eval runs/small/latest.pt
```
Which gives:
| Dataset | Score |
|---|---:|
| iris | 0.920 |
| wine | 0.966 |
| breast_cancer | 0.933 |
| digits | 0.904 |

The reference stage-1 configurations are:

```bash
python -m nanotabicl.train configs/classification.yaml
python -m nanotabicl.train configs/regression.yaml
```

Training writes `latest.pt`, the resolved `config.yaml`, and `metrics.jsonl` to `out_dir`. If `latest.pt` already
exists, the run resumes automatically. To start the next curriculum stage from an earlier checkpoint, use a new output directory and `init_from`:

```bash
python -m nanotabicl.train configs/classification.yaml configs/stage2.yaml \
    init_from=runs/classification/latest.pt \
    out_dir=runs/classification_stage2

python -m nanotabicl.train configs/classification.yaml configs/stage2.yaml configs/stage3.yaml \
    init_from=runs/classification_stage2/latest.pt \
    out_dir=runs/classification_stage3
```

The same commands work with `configs/regression.yaml` in place of `configs/classification.yaml`.

Run the test suite with:

```bash
python -m pytest
```

## Nanoprior

In `nanotabicl/prior.py`, we provide a concise and slightly simplified
implementation of the TabICLv2 prior,
which nevertheless performs similarly well in our
(smaller-scale) experiments.
Changes compared to the TabICLv2 prior are:

- No correlated sampling of scalar variables / hyperparameters.
- Removed graph filtering (should be captured by dataset filtering anyway).
- No graph pruning of irrelevant nodes
(can be a bit slower, but yields the same output).
- Categorical converters with cardinality k always use k dimensions,
never fewer.
- Kumaraswamy warping now affects the extracted dataset column,
not the propagated value.
- Fixed the constant in the random EM function.
- The random activation used for random activation matrices
follows the general random activation now
by only using two of the four possible random power activation types.
- Use corrected categorical size limit of 100 instead of 10.
- Dataset filtering uses the same categorical sizes
in every attempt until a non-filtered dataset is generated.
- fill NaN/inf with zero instead of discarding the whole dataset.
- (Use torch.sign() in random activation,
which has a different behavior at 0 than `2*(x>=0).float()-1`.)

Preprocessing code (outlier handling, standard scaling)
is also not included as it could be done inside the model.

When running the prior module directly, it generates a plot of some random datasets.

## Updates

- 2026/08/17: Replace class-embedding with one-hot + linear, which has a bias compared to the nn.Embedding used before, and matches the full model implementation.
- 2026/06/10: Add nanoprior.
Bugfix based on [#2](https://github.com/soda-inria/nanotabicl/issues/2): Subtract mean when standardizing input data.
- 2026/03/25: Add faster + cached RoPE implementation based on the TabICLv2 version (warning: this permutes the neurons, so it's not compatible with older nanotabicl checkpoints).
- 2026/09/09: Add pretraining code and some evaluations for the trained checkpoints.
