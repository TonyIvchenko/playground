# Targets
.PHONY: setup update test clean build-service start-service \
	build-hurricane-service start-hurricane-service smoke-hurricane-service \
	build-wildfire-service start-wildfire-service smoke-wildfire-service

setup: environment.yml
	conda env create -f environment.yml

update: environment.yml
	conda env update -f environment.yml --prune

test:
	python3 -m pytest -q

clean:
	rm -rf __pycache__

build-service:
	docker build -t test-docker -f src/test_service/Dockerfile src/test_service/

start-service: build-service
	docker network create shared-net || true
	docker run --net shared-net --name redis-service -d redis/redis-stack:latest
	docker run --net shared-net -v /Volumes:/data --name test-service -d test-docker

build-hurricane-service:
	docker build -t hurricane-service-docker -f src/hurricane_service/Dockerfile .

start-hurricane-service: build-hurricane-service
	docker rm -f hurricane-service >/dev/null 2>&1 || true
	docker run --name hurricane-service -p 8000:8000 -d hurricane-service-docker

smoke-hurricane-service:
	curl --fail --silent --show-error http://localhost:8000/health
	curl --fail --silent --show-error http://localhost:8000/ui

build-wildfire-service:
	docker build -t wildfire-service-docker -f src/wildfire_service/Dockerfile .

start-wildfire-service: build-wildfire-service
	docker rm -f wildfire-service >/dev/null 2>&1 || true
	docker run --name wildfire-service -p 8010:8010 -d wildfire-service-docker

smoke-wildfire-service:
	curl --fail --silent --show-error http://localhost:8010/health
	curl --fail --silent --show-error http://localhost:8010/ui
