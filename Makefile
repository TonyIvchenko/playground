app := $(word 2,$(MAKECMDGOALS))
port := $(or $(word 3,$(MAKECMDGOALS)),8080)

.PHONY: setup update run

setup: environment.yml
	conda env create -f environment.yml

update: environment.yml
	conda env update -f environment.yml --prune

run:
	PORT=$(port) python -m src.$(app).main

%:
	@:
