port ?= 8080

# Targets
.PHONY: setup update kernel clean build start smoke

setup: environment.yml
	conda env create -f environment.yml
	$(MAKE) kernel

update: environment.yml
	conda env update -f environment.yml --prune
	$(MAKE) kernel

kernel:
	conda run -n playground python -m ipykernel install --user --name playground --display-name "Python (playground)"

clean:
	rm -rf __pycache__

build:
	docker build -t $(app) -f src/$(app)/Dockerfile .

start: build
	docker rm -f $(app) >/dev/null 2>&1 || true
	docker run --name $(app) -p $(port):$(port) -e API_PORT=$(port) -d $(app)

smoke:
	curl --fail --silent --show-error http://localhost:$(port)/
