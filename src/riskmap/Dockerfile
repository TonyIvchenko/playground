FROM python:3.11-slim

WORKDIR /app

COPY src/riskmap/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src

ENV API_PORT=8080

EXPOSE 8080

CMD ["python", "-m", "src.riskmap.main"]
