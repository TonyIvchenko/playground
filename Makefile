# Targets
.PHONY: setup test clean build-service start-service smoke

setup: environment.yml
	conda env create -f environment.yml

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

smoke:
	python3 -m compileall src/test_service
	python3 -m pytest -q tests/test_service_loop.py tests/test_settings.py
