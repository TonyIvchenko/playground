app := $(word 2,$(MAKECMDGOALS))
port := $(or $(word 3,$(MAKECMDGOALS)),8080)
PYTHON := conda run -n playground python
RUFF := $(PYTHON) -m ruff

.PHONY: setup update run smoke test lint format

setup: environment.yml
	conda env create -f environment.yml

update: environment.yml
	conda env update -f environment.yml --prune

run:
	cd src/$(app) && PORT=$(port) python main.py

smoke:
	$(PYTHON) scripts/smoke_service.py $(app) --port $(port)

test:
	$(PYTHON) scripts/test_service.py $(app)

lint:
	$(RUFF) check .
	$(PYTHON) scripts/check_docs.py
	$(PYTHON) scripts/check_docker_smoke_docs.py
	$(PYTHON) scripts/check_json_yaml_configs.py
	$(PYTHON) scripts/check_tracked_junk.py
	$(PYTHON) scripts/check_browser_apps.py
	$(PYTHON) scripts/check_services.py --check static

format:
	$(RUFF) format .

%:
	@:
