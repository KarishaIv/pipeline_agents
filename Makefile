.PHONY: test test-meta test-cov lint format

# Test meta_agent (the primary test suite for this project)
test:
	uv run pytest src/meta_agent/test -q

test-meta: test

# Full test with coverage (matches pyproject.toml config)
test-cov:
	uv run pytest src/meta_agent/test -q --cov=src/meta_agent --cov-report=term-missing

# Linting and formatting
lint:
	uv run ruff check src/meta_agent
	uv run ruff check src

format:
	uv run ruff format src

# Help
help:
	@echo "Available commands:"
	@echo "  make test        - Run meta_agent tests (quiet)"
	@echo "  make test-meta   - Alias for test"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make lint        - Run linter"
	@echo "  make format      - Format code"
