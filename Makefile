.PHONY: lint test format

lint:
	uv run ruff check .

test:
	uv run pytest

format:
	uv run ruff format .
