from fastapi import APIRouter
from pydantic import BaseModel, Field
import pandas as pd
import pickle
import os

router = APIRouter()

# Paths to model and scaler files
MODEL_PATH = os.path.join("app", "model_api", "inflation_model", "prophet_inflation_model_final.pkl")
SCALER_PATH = os.path.join("app", "model_api", "inflation_model", "regressor_scaler.pkl")

# Load model and scaler at import time
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# List of regressors (features) used in training/scaling
REGRESSORS = [
    "OverallCPI",
    "FoodCPI",
    "FoodInflation",
    "NonFoodCPI",
    "NonFoodInflation",
    "ExchangeRate",
    "Bank Rate/ Policy Rate",
    "Lending Rate",
    "Interest Rate Spread (base lending rate less 3-month deposit rate)"
]

# Define the input data schema with explicit fields and aliases
class InflationPredictionRequest(BaseModel):
    ds: str = Field(..., description="Date in YYYY-MM-DD format")

    OverallCPI: float
    FoodCPI: float
    FoodInflation: float
    NonFoodCPI: float
    NonFoodInflation: float
    ExchangeRate: float
    Bank_Rate_Policy_Rate: float = Field(..., alias="Bank Rate/ Policy Rate")
    Lending_Rate: float = Field(..., alias="Lending Rate")
    Interest_Rate_Spread_base_lending_rate_less_3_month_deposit_rate: float = Field(
        ..., alias="Interest Rate Spread (base lending rate less 3-month deposit rate)"
    )

@router.post("/predict")
async def predict_inflation(data: InflationPredictionRequest):
    # Prepare input dataframe with exact feature names used by the model
    input_dict = {
        "ds": [data.ds],
        "OverallCPI": [data.OverallCPI],
        "FoodCPI": [data.FoodCPI],
        "FoodInflation": [data.FoodInflation],
        "NonFoodCPI": [data.NonFoodCPI],
        "NonFoodInflation": [data.NonFoodInflation],
        "ExchangeRate": [data.ExchangeRate],
        "Bank Rate/ Policy Rate": [data.Bank_Rate_Policy_Rate],
        "Lending Rate": [data.Lending_Rate],
        "Interest Rate Spread (base lending rate less 3-month deposit rate)": [
            data.Interest_Rate_Spread_base_lending_rate_less_3_month_deposit_rate
        ],
    }
    input_df = pd.DataFrame(input_dict)

    # Scale regressors
    input_df[REGRESSORS] = scaler.transform(input_df[REGRESSORS])

    # Use the loaded Prophet model to predict
    forecast = model.predict(input_df)

    pred = forecast.iloc[0]
    return {
        "ds": pred['ds'].strftime("%Y-%m-%d"),
        "yhat": pred['yhat'],
        "yhat_lower": pred['yhat_lower'],
        "yhat_upper": pred['yhat_upper'],
    }
