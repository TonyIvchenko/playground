FROM python:3.11-slim

WORKDIR /app

COPY src/hurricane_service/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/hurricane_service /app/hurricane_service

ENV PYTHONPATH=/app
ENV API_PORT=8000
ENV UI_PATH=/ui

EXPOSE 8000

CMD ["python", "-m", "hurricane_service.main"]
