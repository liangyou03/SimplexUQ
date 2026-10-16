# SimplexUQ

Code for **SimplexUQ: An Evaluation Framework and Benchmark for Conformal
Uncertainty on Simplex-Valued Predictions**.

> **Please cite:** If you use SimplexUQ or SimplexTasks-12 in your research, please cite our paper: Liang You, Hengyu Shi, and Dongwen Ou. [SimplexUQ: An Evaluation Framework and Benchmark for Conformal Uncertainty on Simplex-Valued Predictions](https://arxiv.org/abs/2610.00523), 2026.

SimplexUQ compares conformal wrappers on six synthetic regimes and six
fixed-predictor real tasks. It evaluates marginal coverage, prediction-space
stratified coverage, maximum disparity, worst-stratum coverage, and within-task
radius and runtime. The repository includes the additional BatchMVP comparison.
It is a code release, not a mirror of the source datasets or a paper archive.

Maintainer: [Liang You](https://github.com/liangyou03).

## Installation

Use Python 3.10--3.13. With `uv`:

```bash
uv sync --extra dev
uv run python -m pytest tests -q
```

Or install in an isolated environment with pip:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q
```

Real-task rebuilds have optional dependencies. The `gpu` extra supplies PyTorch
and torchvision; `bio` includes scanpy, anndata, and rpy2 (which needs R).
The API cache builder requires the selected provider's client and a credential
supplied through an environment variable. API calls are separate from conformal
evaluation. No credentials or generated API responses are included here.

## Synthetic quickstart

No external dataset or API key is needed:

```bash
python scripts/run_synthetic.py --config configs/synthetic/smoke_d2_methods.yaml
```

Run a benchmark-scale configuration:

```bash
python scripts/run_synthetic.py --config configs/synthetic/d2_pure_scale.yaml
```

A small BatchMVP comparison:

```bash
python scripts/run_multivalid_benchmark.py --task D1 --n-rep 2
```

Smoke runs are installation checks, not reproductions of reported estimates.
Results are written locally under `results/`, which is excluded from Git.

## Real tasks and frozen artifacts

Real tasks cover CIFAR-10, 20 Newsgroups, SemEval AffectiveText, Samson, PBMC3K
pseudobulk mixtures, and UTKFace age distributions. Obtain source assets under
their own terms and follow the notes in `rebuild/`. Use each runner's `--help`
to inspect paths and settings. The open AffectiveText cache builder is an
alternative predictor, not a numerical reproduction of the frozen API predictor.

The multivalid configuration records relative `task.npz` and strata JSON paths
from the artifact layout. Those external files are deliberately not in this
repository. Supply the corresponding locally authorized artifacts, or update the
configuration paths, before running all six real tasks. Renaming an artifact or
using a newly trained predictor does not reproduce a frozen prediction cache.

## Repository contents

- `src/`: data generators, wrappers, metrics, and simplex utilities.
- `scripts/`: experiment runners, cache builders, and plotting/table scripts.
- `configs/`: experiment parameters and seeds.
- `tests/`: numerical and runner tests.
- `rebuild/`: source-asset rebuild notes.
- `docs/`: protocol cautions and publication boundaries.

Plotting scripts retain the existing templates and method colors. They require
locally generated result files; this code-only release does not include those
outputs or guarantee that a new run recreates every historical paper cell.

## Interpretation

Keep the predictor, score, and stratification protocol fixed within a comparison.
Do not pool historical main and smaller exact-reference runs as paired results.
In particular, some archived PBMC auxiliary runs use different strata from the
main run, and UTKFace/D6 exact references use smaller samples. See
`docs/evaluation_notes.md` before interpreting historical results.

## License status

No open-source license has been selected for this code release. Public access
does not itself grant MIT/Apache-style reuse rights. See `LICENSE` for the notice.
Source datasets, pretrained assets, and provider outputs retain their own terms.
