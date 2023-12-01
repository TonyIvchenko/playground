import time
import redis

mount_path = "/data"

if __name__ == '__main__':
    
    print(f"Hi Bob")

    cache = redis.Redis("redis-service") # by name of container

    while True:

        try:
            # test redis
            cache.set("key", "value")

            # sleep for an minute
            time.sleep(60)
        
        except Exception as ex:
            print(f"Error: {ex}")
            time.sleep(60)

        
            

