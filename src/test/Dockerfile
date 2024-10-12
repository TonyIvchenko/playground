FROM python:3.11-slim

WORKDIR /app

COPY src/test/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY src /app/src

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os, redis; redis.Redis(host=os.environ.get('REDIS_HOST', 'redis-service'), port=int(os.environ.get('REDIS_PORT', '6379'))).ping()" || exit 1

WORKDIR /app/src/test
CMD ["python", "main.py"]
