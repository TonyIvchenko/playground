from service import build_client, run_forever
from settings import load_settings

mount_path = "/data"

if __name__ == "__main__":
    print("starting test-service")
    settings = load_settings()
    cache = build_client(host=settings.redis_host, port=settings.redis_port)
    run_forever(
        cache,
        sleep_seconds=settings.sleep_seconds,
        key=settings.redis_key,
        value=settings.redis_value,
    )

        
            
