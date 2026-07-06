# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `LICENSE` file (MIT).
- `CONTRIBUTING.md` with development setup and contribution guidelines.
- `CHANGELOG.md`.
- `dev` and `test` optional-dependency groups in `pyproject.toml`.
- pytest configuration (`[tool.pytest.ini_options]`) so the suite is discoverable
  without manual `sys.path` manipulation.
- Continuous integration workflow (GitHub Actions) running linting and tests.

### Changed
- Reconciled the supported Python range in `pyproject.toml` with the project's
  actual development environment.

## [0.0.1-dev] - 2024

- Initial development release.
