# Services
SERVICES := hurricane wildfire
SERVICE ?=

SERVICE_DIR_hurricane := src/hurricane_service
SERVICE_DIR_wildfire := src/wildfire_service

IMAGE_hurricane := hurricane-service-docker
IMAGE_wildfire := wildfire-service-docker

CONTAINER_hurricane := hurricane-service
CONTAINER_wildfire := wildfire-service

PORT_hurricane := 8000
PORT_wildfire := 8010

# Targets
.PHONY: setup update clean check-service build start smoke \
	build-hurricane-service start-hurricane-service smoke-hurricane-service \
	build-wildfire-service start-wildfire-service smoke-wildfire-service \
	build-service start-service

setup: environment.yml
	conda env create -f environment.yml

update: environment.yml
	conda env update -f environment.yml --prune

clean:
	rm -rf __pycache__

# Legacy sample service targets.
build-service:
	docker build -t test-docker -f src/test_service/Dockerfile src/test_service/

start-service: build-service
	docker network create shared-net || true
	docker run --net shared-net --name redis-service -d redis/redis-stack:latest
	docker run --net shared-net -v /Volumes:/data --name test-service -d test-docker

check-service:
	@if [ -z "$(SERVICE)" ]; then \
		echo "SERVICE is required (one of: $(SERVICES))."; \
		exit 1; \
	fi
	@if ! echo "$(SERVICES)" | tr ' ' '\n' | grep -qx "$(SERVICE)"; then \
		echo "Invalid SERVICE='$(SERVICE)' (expected one of: $(SERVICES))."; \
		exit 1; \
	fi

build: check-service
	docker build -t $(IMAGE_$(SERVICE)) -f $(SERVICE_DIR_$(SERVICE))/Dockerfile .

start: build
	docker rm -f $(CONTAINER_$(SERVICE)) >/dev/null 2>&1 || true
	docker run --name $(CONTAINER_$(SERVICE)) -p $(PORT_$(SERVICE)):$(PORT_$(SERVICE)) -d $(IMAGE_$(SERVICE))

smoke: check-service
	curl --fail --silent --show-error http://localhost:$(PORT_$(SERVICE))/health
	curl --fail --silent --show-error http://localhost:$(PORT_$(SERVICE))/ui

build-hurricane-service:
	$(MAKE) build SERVICE=hurricane

start-hurricane-service:
	$(MAKE) start SERVICE=hurricane

smoke-hurricane-service:
	$(MAKE) smoke SERVICE=hurricane

build-wildfire-service:
	$(MAKE) build SERVICE=wildfire

start-wildfire-service:
	$(MAKE) start SERVICE=wildfire

smoke-wildfire-service:
	$(MAKE) smoke SERVICE=wildfire
