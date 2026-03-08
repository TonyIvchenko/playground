FROM python:3.11-slim

WORKDIR /app

COPY src/disasters/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src

ENV PORT=8080

EXPOSE 8080

WORKDIR /app/src/disasters
CMD ["python", "main.py"]
