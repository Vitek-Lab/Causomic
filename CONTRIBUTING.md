# Contributing to Causomic

Thanks for your interest in contributing! This document explains how to set up a
development environment, the conventions we follow, and how to submit changes.

## Development setup

Causomic targets Python 3.11–3.12 and depends on PyTorch (2.3+) and Pyro. We
recommend a clean virtual environment.

```bash
git clone https://github.com/Vitek-Lab/Causomic.git
cd Causomic

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package together with the development tooling
pip install -e ".[dev]"
```

The `dev` extra installs everything needed to run the test suite and the
formatting/linting tools. Some dependencies (e.g. `indra`, `indra-cogex`) are
installed directly from GitHub and require network access.

## Running the tests

```bash
pytest                 # run the full suite
pytest --cov=causomic  # with coverage
```

Tests live in `tests/` and should not require network access or external
databases (Neo4j, INDRA REST). Mock those boundaries instead.

## Code style

We use [Black](https://black.readthedocs.io/) for formatting and
[isort](https://pycqa.github.io/isort/) for import ordering, configured in
`pyproject.toml` (line length 100).

```bash
black src/ tests/
isort src/ tests/
```

Please run both before opening a pull request. The CI workflow checks formatting
and will fail if files are not formatted.

## Submitting changes

1. Fork the repository and create a feature branch:
   `git checkout -b feature/short-description`
2. Make your changes, including tests and docstrings for new functionality.
3. Ensure `black`, `isort`, and `pytest` all pass locally.
4. Update `CHANGELOG.md` under the `[Unreleased]` section.
5. Open a pull request describing the change and its motivation.

## Reporting issues

Please use the [issue tracker](https://github.com/Vitek-Lab/Causomic/issues)
and include a minimal reproducible example where possible.
