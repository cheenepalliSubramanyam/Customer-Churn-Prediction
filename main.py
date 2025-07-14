from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from apis.churn import predictChurn
from mangum import Mangum

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API!"}

class ChurnPredictionRequest(BaseModel):
    MonthlyCharges: float
    TotalCharges: float
    InternetService: str
    tenure: float
    Contract: str

@app.post("/predict_churn")
def predict_churn(request: ChurnPredictionRequest):
    try:
        result = predictChurn(
            request.MonthlyCharges,
            request.TotalCharges,
            request.InternetService,
            request.tenure,
            request.Contract
        )
        return {"prediction": "More probable to Churn" if result == 1 else "Less probable to Churn"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)

def lambda_handler(event, context):
    return handler(event, context)
