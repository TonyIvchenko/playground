# Targets
.PHONY: setup, test, clean, build-service, start-service

setup: environment.yml
	conda env create -f environment.yml

test:
	py.test tests

clean:
	rm -rf __pycache__

build-service:
	docker build -t test-docker -f src/test_service/Dockerfile src/test_service/

start-service: build-service
	docker network create shared-net`
	docker run --net shared-net --name redis-service -d redis/redis-stack:latest
	docker run --net shared-net -v /Volumes:/data --name test-service -d test-docker
