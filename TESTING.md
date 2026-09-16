# Testing

## Task 19 - Integration Testing, Load Testing and Bug Fixing

### 1. Docker Testing

The API was run using Docker Compose.

Command:

```bash
docker compose up --build -d

The container was checked using:

docker compose ps

The container was running successfully.

2. Integration Testing

The following endpoints were tested:

GET /
GET /api/v1/health
POST /api/v1/predict
POST /api/v1/predict-batch
GET /metrics

All endpoints worked successfully.

The health endpoint confirmed that the ML model was loaded correctly.

3. Bug Found and Fixed

During batch prediction testing, two inputs were sent but only one prediction was returned.

The problem was caused by incorrect indentation of results.append().

The statement was outside the for loop, so only the last prediction was added to the response.

The indentation was corrected and the batch endpoint was tested again.

After the fix, two inputs correctly returned two predictions.

4. Prometheus Metrics

The prediction metrics were checked using:

curl http://127.0.0.1:8000/metrics

The custom prediction counter was working correctly.

Example:

ml_predictions_total{predicted_class="0"} 1.0
ml_predictions_total{predicted_class="1"} 1.0
5. Automated Testing

Pytest was executed using:

python -m pytest -v

Result:

15 passed
6016 warnings

All tests passed successfully.

6. Load Testing

A load test was performed with 50 concurrent requests.

Command:

python load_test.py

Results:

Metric	Result
Total requests	50
Successful	50
Failed	0
Total time	1.9613 seconds
Average response	1.8832 seconds
Minimum response	1.2911 seconds
Maximum response	1.9428 seconds

All 50 requests were successful.

Final Result
Docker testing: PASS
Integration testing: PASS
Batch prediction bug: FIXED
Prometheus metrics: PASS
Pytest: 15/15 PASS
Load testing: 50/50 PASS

Task 19 completed successfully.