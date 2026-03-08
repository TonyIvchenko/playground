FROM python:3.11-slim

WORKDIR /app

COPY src/test/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src

WORKDIR /app/src/test
CMD ["python", "main.py"]
