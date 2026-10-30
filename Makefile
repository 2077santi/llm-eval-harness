PY ?= python

ratings:
	$(PY) -m evalh.dataset

judge-smoke:
	$(PY) -m evalh.judge --limit 3

judge:
	$(PY) -m evalh.judge

judge-mock:
	$(PY) -m evalh.judge --mock

report:
	$(PY) -m evalh.analyze

test:
	$(PY) -m pytest -q

.PHONY: ratings judge-smoke judge judge-mock report test
