# Targets
.PHONY: setup update clean build start smoke

setup: environment.yml
	conda env create -f environment.yml

update: environment.yml
	conda env update -f environment.yml --prune

clean:
	rm -rf __pycache__

build:
	docker build -t $(app) -f src/$(app)/Dockerfile .

start: build
	docker rm -f $(app) >/dev/null 2>&1 || true
	docker run --name $(app) -p $(port):$(port) -e API_PORT=$(port) -d $(app)

smoke:
	curl --fail --silent --show-error http://localhost:$(port)/health
	curl --fail --silent --show-error http://localhost:$(port)/ui
