# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-07-10

First publicly packaged release, focused on publication readiness.

### Added
- `LICENSE` file (MIT).
- `CONTRIBUTING.md` with development setup and contribution guidelines.
- `CHANGELOG.md`.
- `dev` and `test` optional-dependency groups in `pyproject.toml`.
- pytest configuration (`[tool.pytest.ini_options]`) so the suite is discoverable
  without manual `sys.path` manipulation.
- Test suite (154 tests) covering the core modules.
- Continuous integration workflow (GitHub Actions) running linting and tests,
  with coverage reporting via Codecov.
- PyPI packaging metadata: trove `classifiers` and search `keywords`.

### Changed
- Set the supported Python range to 3.10–3.11 (`requires-python = ">=3.10,<3.12"`).
- Refreshed the README to match the current package and point to the
  `user_manual.ipynb` vignette instead of inline quick-start code.
- Pinned `black==25.9.0` and `isort==8.0.1` in CI and the `dev` extra so local
  and CI formatting agree.
- Bumped version from `0.0.1-dev` to `0.9.0`.

### Fixed
- Two latent bugs surfaced while building the test suite.
- `test_run_graph_sim` now skips gracefully when the optional `notears`
  dependency is absent instead of erroring.
- Restored `data/images/logo.png`, which had been committed as a corrupt
  489-byte download.

### Removed
- Large data files, `.pkl` graph objects, notebooks, and binary docs from the
  repository and from git history (`.git` reduced from ~433 MB to ~7 MB).
- Stale benchmark and vignette scripts no longer part of the package.

## [0.0.1-dev] - 2024

- Initial development release.

[Unreleased]: https://github.com/Vitek-Lab/Causomic/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/Vitek-Lab/Causomic/releases/tag/v0.9.0
