.PHONY: install chat benchmark compare report test lint

install:
	python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"

chat:
	slm-chat

benchmark:
	slm-benchmark

compare:
	slm-compare

report:
	slm-report

test:
	pytest -q

lint:
	ruff check src tests
