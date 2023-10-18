FROM python:3.11-slim

WORKDIR /app

COPY src/test/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src/test /app

CMD ["python", "-u", "/app/main.py"]
