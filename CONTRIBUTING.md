# Contributing to boolmf

Thanks for helping. This file covers how changes get into `main` and what a change needs.

## Workflow

- `main` is protected: every change goes through a pull request, and the CI check `ci-passed`
  must pass before it can be merged. Branch from the latest `main`.
- Pull requests are squash-merged, so the PR title becomes the commit message. Make it say what
  the change does.
- Open a draft pull request early if you want feedback on the approach.

## Development setup

```bash
git clone https://github.com/siddC/boolmf
cd boolmf
pip install -e ".[dev]"
```

The package uses a src layout (`src/boolmf`). Supported Python versions are 3.12 to 3.14.

## Checks

```bash
ruff check .                  # lint (CI pins the ruff version; see .github/workflows/ci.yml)
pytest -q -m "not slow"       # what CI runs
pytest -q                     # also the slower recovery tests
```

CI also builds the wheel and runs `.github/smoke_test.py` against it, so a new module must be
importable from the installed package, not only from the source tree.

## What a change needs

- **Tests.** New behaviour gets a test. Estimators must keep passing scikit-learn's estimator
  checks (`tests/test_sklearn_compat.py`).
- **A changelog entry** in `CHANGELOG.md`, under the unreleased version: what changed and why it
  matters to a user.
- **Docstrings** in numpydoc style for public functions, classes and parameters.

## Methods from papers

boolmf implements published methods from the papers themselves, not from reference code.

- The original method of a cited paper stays available as an option (a keyword value or a
  preset in `boolmf.presets`), even when boolmf's default differs.
- Each such option is checked by reproducing the paper's published results on the paper's own
  inputs (`benchmarks/papers/`). Where the paper is ambiguous or the reproduction departs from
  it, the docstring says how and why.
- A method whose correct version cannot be established is left out until it can be (MEBF was
  dropped for this reason; see `ROADMAP.md`).

## Reporting problems

Open an issue with a minimal example (a small matrix and the call that fails), the boolmf
version (`boolmf.__version__`), and the Python, NumPy and scikit-learn versions.
