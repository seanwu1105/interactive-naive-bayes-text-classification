# How to Contribute

## Getting Started

Install Python 3.10 or later.

Install Poetry 1.2 or later. See [Poetry's documentation](https://python-poetry.org/docs/) for details.

Install the project's dependencies:

```sh
poetry install --no-root
```

Install pre-commit hooks:

```sh
poetry run pre-commit install
```

Activate the virtual environment:

```sh
poetry shell
```

## Running Tests

```sh
poetry run pytest
```
