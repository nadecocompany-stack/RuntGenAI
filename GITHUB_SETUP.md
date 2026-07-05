# GitHub + Claude Code setup

## 0. What's in the repo

Everything is already here: source (`src/`), training (`src/training/`), tests
(`tests/`), demos (`scripts/`), UI (`ui/`), docs (`README.md`, `ARCHITECTURE.md`,
`index.html`), config (`pyproject.toml`, `requirements.txt`), CI
(`.github/workflows/ci.yml`), and repo files (`.gitignore`, `.gitattributes`,
`LICENSE`, `CLAUDE.md`, `CONTRIBUTING.md`).

Before pushing: open `LICENSE` and replace `<YOUR NAME OR ORGANIZATION>`.

## 1. Put it on GitHub

### Option A — GitHub CLI (fastest)

```bash
cd radiology_platform
git init
git add .
git commit -m "Initial commit: radiology abnormality detection pipeline"
git branch -M main
gh repo create radiology-abnormality-detection --private --source=. --push
```

`--private` is recommended for anything medical-adjacent; switch to `--public`
if you intend to open-source it.

### Option B — plain git (repo created on github.com first)

1. On github.com: **New repository** → name it, choose Private, **do not** add a
   README/.gitignore/license (you already have them) → Create.
2. Then locally:

```bash
cd radiology_platform
git init
git add .
git commit -m "Initial commit: radiology abnormality detection pipeline"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
```

### Option C — web upload (no git)

New repository → on the empty repo page, **uploading an existing file** → drag the
whole folder in → commit to `main`. Fine for a one-off, but the CLI/git options
are better for ongoing work.

### Branch strategy

- Initial push goes to **`main`** (the default branch).
- After that, don't commit to `main` directly — branch per change
  (`git switch -c feature/x`), push, open a PR into `main`. Consider enabling
  branch protection on `main` (Settings → Branches).

## 2. Connect to Claude Code

Two ways, depending on how you want to work.

### A. Claude Code on the web / mobile app (cloud — matches "the app")

Claude clones your repo into a cloud VM, works, and pushes a branch as a PR.

1. Go to **claude.ai/code** (or the **Code** tab in the Claude mobile app) and
   sign in.
2. Follow the prompt to **connect GitHub** — this installs the **Claude GitHub
   App**; grant it access to your repo (all repos or select ones).
3. When prompted, **create a cloud environment** (controls network access + setup).
4. Pick your repository (and a branch) with the selector, type a task, press Enter.
5. Review the diff → **Create PR**.

Requires a Pro/Max/Team plan (or Enterprise premium seat); it's in research
preview. Docs: https://code.claude.com/docs/en/web-quickstart

### B. Claude Code in the terminal (local)

```bash
npm install -g @anthropic-ai/claude-code   # needs Node.js
git clone https://github.com/<you>/<repo>.git
cd <repo>
claude                                      # start Claude Code in the repo
```

To let people trigger Claude with `@claude` in issues/PRs, run this inside the
repo and follow the prompts (it sets up the GitHub App + Action + secrets):

```bash
/install-github-app
```

Docs: https://docs.claude.com/en/docs/claude-code/overview

## 3. Verify CI

After the first push, open the **Actions** tab — the `CI` workflow runs `mypy`
and `pytest` on Python 3.10 and 3.11. (First run installs CPU PyTorch + MONAI, so
it takes a few minutes.)
