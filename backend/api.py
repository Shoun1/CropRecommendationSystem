from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from crop_rec import predictions_naive_bayes
import pandas as pd
from pydantic import BaseModel

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

'''@app.post("/predict")
def predict_crop(input_data: CropInput):
    data = np.array([[input_data.N, input_data.P, input_data.K, input_data.ph, input_data.rainfall]])
    prediction = model.predict(data)
    return {"prediction": prediction[0]}'''

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

