FROM python:3.11-slim

WORKDIR /app

COPY src/hurricane/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/hurricane /app/hurricane

ENV PYTHONPATH=/app
ENV API_PORT=8000

EXPOSE 8000

CMD ["python", "-m", "hurricane.main"]
