# Contributing to JAQSI

Contributions are welcome! Here is the usual path:

1. Open an issue using the bug report or feature request template.
2. Fork the repository and make your changes.
3. Open a pull request.

## Setup

Install the `dev` and `docs` dependency groups:
```
uv sync --all-groups
```

Install the pre-commit hooks:
```
uv run pre-commit install
```

The hook runs Ruff before each commit, so formatting issues are caught early.

## Testing

Run the checks before opening a pull request:
```
uv run ruff format --check jaqsi tests
uv run ruff check jaqsi tests
uv run pytest --dist load -m "not benchmark" -n auto
```
The pytest command skips benchmarks, which helps keep the feedback loop short. CI runs the project checks again on your pull request.

## Packaging

GitHub Actions publishes a release when the version in `pyproject.toml` differs from the latest Git tag. The workflow tags the release, generates release notes, and publishes to PyPI.

## Documentation

For a live look at documentation changes, start the Zensical preview:
```
uv run zensical serve
```
The preview rebuilds when files change. See the [Zensical documentation](https://zensical.org/) for details.

GitHub Actions builds and publishes the documentation on release. The strict build treats warnings as errors.
