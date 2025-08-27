from fastapi import FastAPI
from pydantic import BaseModel
import joblib

model = joblib.load('Bayesian_model.pkl')

app = FastAPI(title="Crop Recommendation System")

class CropInput(BaseModel):
    N: float = 0
    P: float = 0
    K: float = 25
    ph: float = 6.5
    rainfall: float = 100

@app.post("/")
def home():
    return {"message": "Welcome to the Crop Recommendation System"}

from fastapi import Query

@app.get("/predict")
def predict(
    N: float = Query(...),
    P: float = Query(...),
    K: float = Query(...),
    ph: float = Query(...),
    rainfall: float = Query(...)
):
    features = [[N, P, K, ph, rainfall]]
    prediction = model.predict(features)[0]
    return {"prediction": prediction}

