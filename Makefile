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
	@tests_dir="src/$(app)/tests"; \
	if [ -d "$$tests_dir" ]; then \
		echo "Running $(PYTHON) -m pytest -q $$tests_dir"; \
		$(PYTHON) -m pytest -q "$$tests_dir"; \
	else \
		available=$$(find src -maxdepth 2 -type d -name tests | sort | sed 's#^src/##; s#/tests$$##' | tr '\n' ',' | sed 's/,$$//; s/,/, /g'); \
		if [ -z "$$available" ]; then available="(none)"; fi; \
		echo "No test suite found for service '$(app)'. Available test targets: $$available" >&2; \
		exit 1; \
	fi

lint:
	$(RUFF) check .
	$(PYTHON) scripts/validate_repo.py
	$(PYTHON) scripts/check_docker_smoke_docs.py
	$(PYTHON) scripts/check_browser_apps.py
	$(PYTHON) scripts/check_services.py --check static

format:
	$(RUFF) format .

%:
	@:
