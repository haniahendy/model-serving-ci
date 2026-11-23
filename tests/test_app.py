from fastapi.testclient import TestClient

from serving.app import create_app
from serving.train import train


def client(tmp_path):
    path = tmp_path / "m.joblib"
    acc = train(str(path))
    return TestClient(create_app(path)), acc


def test_training_accuracy_reasonable(tmp_path):
    assert client(tmp_path)[1] > 0.9


def test_predict_setosa_and_virginica(tmp_path):
    c, _ = client(tmp_path)
    r = c.post("/predict", json={"sepal_length": 5.0, "sepal_width": 3.4, "petal_length": 1.5, "petal_width": 0.2}).json()
    assert r["species"] == "setosa"
    r = c.post("/predict", json={"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.8, "petal_width": 2.2}).json()
    assert r["species"] == "virginica"


def test_validation_rejects_bad_input(tmp_path):
    c, _ = client(tmp_path)
    assert c.post("/predict", json={"sepal_length": -1, "sepal_width": 3, "petal_length": 1, "petal_width": 1}).status_code == 422
    assert c.post("/predict", json={"sepal_length": 1}).status_code == 422


def test_health_and_missing_model(tmp_path):
    c, _ = client(tmp_path)
    assert c.get("/health").json() == {"status": "ok", "model_loaded": True}
    empty = TestClient(create_app(tmp_path / "nope.joblib"))
    assert empty.get("/health").json()["model_loaded"] is False
    assert empty.post("/predict", json={"sepal_length": 5, "sepal_width": 3, "petal_length": 1, "petal_width": 1}).status_code == 503
