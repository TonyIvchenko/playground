# Targets
.PHONY: setup test clean build-service start-service \
	build-hurricane-service train-hurricane-model start-hurricane-service smoke-hurricane-service

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

build-hurricane-service:
	docker build -t hurricane-intensity-docker -f src/hurricane_service/Dockerfile .

train-hurricane-model:
	python scripts/train_hurricane_intensity_model.py \
		--allow-demo-data \
		--input-csv data/sample/hurricane_training_sample.csv \
		--output-path artifacts/hurricane/model_bundle.joblib \
		--model-version 2026.03.v1

start-hurricane-service: build-hurricane-service
	docker rm -f hurricane-service >/dev/null 2>&1 || true
	docker run \
		--name hurricane-service \
		-p 8000:8000 \
		-d hurricane-intensity-docker

smoke-hurricane-service:
	curl --fail --silent --show-error http://localhost:8000/health
	curl --fail --silent --show-error http://localhost:8000/ui
