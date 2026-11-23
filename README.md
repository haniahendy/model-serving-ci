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
Tests pass locally and in GitHub Actions: the CI run on the latest commit (run #2) passed. The first run (#1) failed; it ran on the commit that added the workflow, one upload step before the final commit, and its failure cause was not investigated. The Dockerfile has **not been built or run** by the author.

## Limits
Iris is a toy dataset. No authentication, rate limiting, model versioning, monitoring or batch endpoint.

## Provenance
A learning and portfolio project.
