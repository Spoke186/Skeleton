# Every target is a one-line wrapper over a Python entrypoint, so the same work runs on
# Windows without make. See docs/adr/0003-makefile-wraps-python-entrypoints.md.
.PHONY: help install lint typecheck test test-fast gates coverage figures experiments \
        migrate worker nightly chaos paper clean

PYTHON ?= python

help:
	@echo "install      install runtime + dev dependencies"
	@echo "lint         ruff check + ruff format --check"
	@echo "typecheck    mypy voldesk/"
	@echo "test         full pytest suite with coverage"
	@echo "test-fast    pytest excluding slow tests"
	@echo "gates        only the eight Phase 1 acceptance gates"
	@echo "figures      regenerate all eight figures as vector PDF"
	@echo "experiments  run E1-E4 end to end"
	@echo "migrate      apply Django migrations"
	@echo "worker       run the Job queue worker"
	@echo "chaos        inject faults into the local stack"
	@echo "paper        compile paper/voldesk.tex"

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy voldesk/

test:
	pytest --cov=voldesk --cov-report=term-missing

test-fast:
	pytest -m "not slow"

gates:
	pytest -m gate -v

coverage:
	pytest --cov=voldesk --cov-report=html
	@echo "open htmlcov/index.html"

figures:
	$(PYTHON) -m voldesk.figures.build_all

experiments:
	$(PYTHON) -m voldesk.experiments.run_all

migrate:
	$(PYTHON) manage.py migrate

worker:
	$(PYTHON) manage.py run_worker

nightly:
	$(PYTHON) manage.py enqueue_nightly

chaos:
	$(PYTHON) scripts/chaos.py --all

paper:
	$(PYTHON) scripts/build_paper.py

clean:
	$(PYTHON) -c "import shutil,pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ['.pytest_cache','.mypy_cache','.ruff_cache','htmlcov','build']]"
