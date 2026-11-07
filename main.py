from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Insurance Cost API")

model = joblib.load("model.joblib")


class Person(BaseModel):
    age: int = Field(ge=18, le=100)
    sex: Literal["male", "female"]
    bmi: float = Field(ge=10, le=60)
    children: int = Field(ge=0, le=10)
    smoker: Literal["yes", "no"]
    region: Literal["northeast", "northwest", "southeast", "southwest"]


@app.get("/")
def home():
    return {"message": "Insurance API is running"}


@app.post("/predict")
def predict(p: Person):
    df = pd.DataFrame([p.model_dump()])
    pred = model.predict(df)[0]
    return {"predicted_charges": round(float(pred), 2)}