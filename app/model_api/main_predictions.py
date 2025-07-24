from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Optional
import os
import pandas as pd
import pickle
from pathlib import Path
import re

# Paths
BASE_DIR = Path("app") / "model_api"
MODEL_PATH = BASE_DIR / "inflation_model" / "prophet_inflation_model_final.pkl"
SCALER_PATH = BASE_DIR / "inflation_model" / "regressor_scaler.pkl"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "my_data.csv"

router = APIRouter()

# Load inflation model and scaler
with open(MODEL_PATH, "rb") as f:
    inflation_model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

REGRESSORS = [
    "OverallCPI", "FoodCPI", "FoodInflation", "NonFoodCPI", "NonFoodInflation",
    "ExchangeRate", "Bank Rate/ Policy Rate", "Lending Rate",
    "Interest Rate Spread (base lending rate less 3-month deposit rate)",
]

# Map model feature names to CSV column names
FEATURE_NAME_MAP = {
    "Bank_Rate__Policy_Rate": "Bank Rate/ Policy Rate",
    "ExchangeRate": "ExchangeRate",
    "FoodCPI": "FoodCPI",
    "FoodInflation": "FoodInflation",
    "Interest_Rate_Spread__base_lending_rate_less_3-month_deposit_rate_":
        "Interest Rate Spread (base lending rate less 3-month deposit rate)",
    "Lending_Rate": "Lending Rate",
    "NonFoodCPI": "NonFoodCPI",
    "NonFoodInflation": "NonFoodInflation",
    "OverallCPI": "OverallCPI",
}


# Prophet Forecaster class
class ProphetForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def load_all_models(self, load_dir):
        if not os.path.exists(load_dir):
            print(f"Model directory {load_dir} does not exist.")
            return
        files = [f for f in os.listdir(load_dir) if f.endswith(".pkl")]
        for filename in files:
            if filename.startswith("prophet_model_"):
                feature_name = filename[len("prophet_model_"):-4]
                path = os.path.join(load_dir, filename)
                try:
                    with open(path, "rb") as f:
                        self.models[feature_name] = pickle.load(f)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def create_forecasts(self, until_date=None):
        self.forecasts.clear()
        for feat, model in self.models.items():
            last_date = model.history["ds"].max()
            if until_date:
                target = pd.to_datetime(until_date)
                periods = (target.year - last_date.year) * 12 + (target.month - last_date.month) + 1
                periods = max(periods, 36)  # default to at least 3 years
            else:
                periods = 36
            future = model.make_future_dataframe(periods=periods, freq="MS")
            forecast = model.predict(future)
            forecast["ds"] = pd.to_datetime(forecast["ds"])
            self.forecasts[feat] = forecast[["ds", "yhat"]].rename(columns={"yhat": feat})

    def predict_on_date(self, date_str):
        target_date = pd.to_datetime(date_str).date()
        preds = {}
        for feat, df in self.forecasts.items():
            row = df[df["ds"].dt.date == target_date]
            preds[feat] = float(row[feat].values[0]) if not row.empty else None
        return preds


# Initialize and pre-load forecaster
forecaster = ProphetForecaster()
forecaster.load_all_models(str(MODEL_DIR))

# Load historical data
df_data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
df_data["ds"] = pd.to_datetime(df_data["ds"])

# Response schema
class UnifiedPredictionResponse(BaseModel):
    date: str
    predictions: Dict[str, Optional[float]]
    source: str  # "historical data" or "forecast"


@router.get("/unified-predict", response_model=UnifiedPredictionResponse)
def unified_predict(date: str = Query(..., description="Date in format YYYY-MM-DD")):
    try:
        target_date = pd.to_datetime(date).normalize()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Check if data exists in CSV
    row = df_data[df_data["ds"].dt.date == target_date.date()]

    if not row.empty:
        # Use features from CSV
        features = {col: float(row.iloc[0][col]) for col in REGRESSORS}
        source = "historical data"
    else:
        # Ensure forecasts go far enough
        forecaster.create_forecasts(until_date=target_date)

        raw_preds = forecaster.predict_on_date(date)

        features = {}
        for k, v in raw_preds.items():
            mapped_name = FEATURE_NAME_MAP.get(k, k)
            if v is None:
                raise HTTPException(status_code=404, detail=f"Missing forecast for feature: {mapped_name}")
            features[mapped_name] = v
        source = "forecast"

    # Prepare input for inflation prediction
    input_df = pd.DataFrame([features])
    input_df["ds"] = [target_date]
    input_df[REGRESSORS] = scaler.transform(input_df[REGRESSORS])

    # Predict inflation
    forecast = inflation_model.predict(input_df).iloc[0]

    response = {
        "date": date,
        "predictions": {
            "inflation": forecast["yhat"],
            "inflation_lower": forecast["yhat_lower"],
            "inflation_upper": forecast["yhat_upper"],
            **features,
        },
        "source": source,
    }

    return response
