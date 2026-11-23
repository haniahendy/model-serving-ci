import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model.joblib"))


class Flower(BaseModel):
    sepal_length: float = Field(gt=0, lt=20)
    sepal_width: float = Field(gt=0, lt=20)
    petal_length: float = Field(gt=0, lt=20)
    petal_width: float = Field(gt=0, lt=20)


def create_app(model_path: Path = MODEL_PATH) -> FastAPI:
    app = FastAPI(title="iris-serving")
    bundle = joblib.load(model_path) if Path(model_path).exists() else None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": bundle is not None}

    @app.post("/predict")
    def predict(f: Flower):
        if bundle is None:
            raise HTTPException(503, "model not loaded")
        x = [[f.sepal_length, f.sepal_width, f.petal_length, f.petal_width]]
        proba = bundle["model"].predict_proba(x)[0]
        i = int(proba.argmax())
        return {"species": bundle["classes"][i], "probability": round(float(proba[i]), 3)}

    return app


app = create_app()
