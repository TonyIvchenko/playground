FROM python:3.11-slim

WORKDIR /app

COPY src/wildfire_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data

ENV PYTHONPATH=/app/src
ENV MODEL_BUNDLE_PATH=/app/model/model_bundle.joblib

RUN python /app/scripts/train_wildfire_ignition_model.py \
    --allow-demo-data \
    --input-csv /app/data/sample/wildfire_training_sample.csv \
    --output-path /app/model/model_bundle.joblib \
    --model-version 2026.03.v1

EXPOSE 8010

CMD ["python", "-u", "/app/src/wildfire_service/main.py"]
