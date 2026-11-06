from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")
columns = joblib.load("columns.joblib")

class Person(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/predict")
def predict(p: Person):
    df = pd.DataFrame([p.model_dump()])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    return {"predicted_cost": float(model.predict(df)[0])}