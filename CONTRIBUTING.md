# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest mypy
```

## Before every PR

```bash
pytest        # all tests must pass
mypy          # type check must pass
```

CI (`.github/workflows/ci.yml`) runs both on Python 3.10 and 3.11.

## Branch & PR flow

- `main` is the default, protected branch — don't push to it directly.
- Branch from `main`: `git switch -c feature/<short-name>` (or `fix/<name>`).
- Keep changes focused; update or add tests in `tests/` for any behaviour change.
- Open a pull request into `main`; let CI pass before merging.

## Ground rules

- Never commit datasets, real DICOM, PHI, or trained weights (see `.gitignore`).
- Keep the "not for clinical use" disclaimers intact.
- Don't weaken `deidentify.py` or `audit.py` without explicit review.
