from service import build_client, run_forever

mount_path = "/data"

if __name__ == "__main__":
    print("starting test-service")
    cache = build_client()
    run_forever(cache)

        
            
