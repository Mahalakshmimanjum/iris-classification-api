import asyncio
import time
import httpx

URL = "http://127.0.0.1:8000/api/v1/predict"
API_KEY = "my-secret-api-key-123"

PAYLOAD = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2,
}

TOTAL_REQUESTS = 50


async def send_request(client):
    start = time.perf_counter()

    try:
        response = await client.post(
            URL,
            json=PAYLOAD,
            headers={"X-API-Key": API_KEY},
        )

        elapsed = time.perf_counter() - start

        return response.status_code, elapsed

    except Exception as e:
        elapsed = time.perf_counter() - start
        return f"ERROR: {e}", elapsed


async def main():
    async with httpx.AsyncClient(timeout=10.0) as client:
        start_time = time.perf_counter()

        tasks = [
            send_request(client)
            for _ in range(TOTAL_REQUESTS)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

    successful = [
        elapsed for status, elapsed in results
        if status == 200
    ]

    failed = [
        (status, elapsed) for status, elapsed in results
        if status != 200
    ]

    print("\n===== LOAD TEST RESULTS =====")
    print(f"Total requests   : {TOTAL_REQUESTS}")
    print(f"Successful       : {len(successful)}")
    print(f"Failed           : {len(failed)}")
    print(f"Total time       : {total_time:.4f} seconds")

    if successful:
        print(f"Average response : {sum(successful) / len(successful):.4f} seconds")
        print(f"Minimum response : {min(successful):.4f} seconds")
        print(f"Maximum response : {max(successful):.4f} seconds")

    if failed:
        print("\nFailed requests:")
        for status, elapsed in failed:
            print(f"  {status} - {elapsed:.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())