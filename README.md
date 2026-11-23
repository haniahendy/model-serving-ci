# model-serving-ci

Train a scikit-learn model, serve it with FastAPI (input validation, health check, 503 when the model is missing), test the API, and run the tests in GitHub Actions. Includes a Dockerfile.

## Run
```bash
pip install -r requirements.txt
python -m pytest -q           # 4 tests
python -m serving.train       # writes model.joblib
uvicorn serving.app:app       # POST /predict, GET /health
```
Example: `curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"sepal_length":5,"sepal_width":3.4,"petal_length":1.5,"petal_width":0.2}'`

## Status
Tests pass locally (see below). The GitHub Actions workflow and Dockerfile have **not been run** by the author yet; check the Actions tab after the first push. The Docker image has not been built.

## Limits
Iris is a toy dataset. No authentication, rate limiting, model versioning, monitoring or batch endpoint.

## Provenance
A learning and portfolio project.
