FROM python:3.11-slim

WORKDIR /app

COPY src/wildfire_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/wildfire_service /app/wildfire_service

ENV PYTHONPATH=/app
ENV API_PORT=8010
ENV UI_PATH=/ui

EXPOSE 8010

CMD ["python", "-m", "wildfire_service.main"]
