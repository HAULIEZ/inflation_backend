from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict
import os
import pandas as pd
import pickle
import re

# --- Your ProphetForecaster and sanitize function ---
class ProphetForecaster:
    def __init__(self, growth='linear', seasonality_mode='multiplicative'):
        self.models = {}
        self.forecasts = {}
        self.growth = growth
        self.seasonality_mode = seasonality_mode

    def load_all_models(self, load_dir):
        if not os.path.exists(load_dir):
            print(f"Model directory {load_dir} does not exist.")
            return

        files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
        if not files:
            print(f"No model files found in {load_dir}")
            return

        for filename in files:
            prefix = "prophet_model_"
            if filename.startswith(prefix):
                safe_feature_name = filename[len(prefix):-4]  # remove prefix and .pkl
                model_path = os.path.join(load_dir, filename)
                try:
                    with open(model_path, 'rb') as f:
                        self.models[safe_feature_name] = pickle.load(f)
                    print(f"Loaded model for feature: {safe_feature_name}")
                except Exception as e:
                    print(f"Failed to load model {filename}: {e}")
            else:
                print(f"Skipping unrecognized file: {filename}")

    def predict_on_date(self, date_str):
        target_date = pd.to_datetime(date_str).normalize()
        predictions = {}

        if not self.forecasts:
            for feature, model in self.models.items():
                future = model.make_future_dataframe(periods=12, freq='MS')
                forecast = model.predict(future)
                self.forecasts[feature] = forecast[['ds', 'yhat']].rename(columns={'yhat': feature})

        for feature, forecast_df in self.forecasts.items():
            row = forecast_df[forecast_df['ds'] == target_date]
            predictions[feature] = row[feature].values[0] if not row.empty else None

        return predictions

def sanitize_filename(name):
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)

# --- FastAPI setup ---
router = APIRouter()
MODEL_DIR = os.path.join("app", "model_api", "models")  # <-- relative path here
forecaster = ProphetForecaster()

feature_name_map = {
    "Bank_Rate__Policy_Rate": "Bank Rate/ Policy Rate",
    "ExchangeRate": "ExchangeRate",
    "FoodCPI": "FoodCPI",
    "FoodInflation": "FoodInflation",
    "Interest_Rate_Spread__base_lending_rate_less_3-month_deposit_rate_": "Interest Rate Spread (base lending rate less 3-month deposit rate)",
    "Lending_Rate": "Lending Rate",
    "NonFoodCPI": "NonFoodCPI",
    "NonFoodInflation": "NonFoodInflation",
    "OverallCPI": "OverallCPI",
}

class PredictionResponse(BaseModel):
    date: str
    predictions: Dict[str, float | None]

@router.on_event("startup")
def load_all_models_on_startup():
    forecaster.load_all_models(MODEL_DIR)

@router.get("/predict", response_model=PredictionResponse)
def get_predictions(date: str = Query(..., description="Date in format YYYY-MM-DD")):
    try:
        pd.to_datetime(date)  # validate date format
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    preds = forecaster.predict_on_date(date)

    if all(v is None for v in preds.values()):
        raise HTTPException(status_code=404, detail="Date outside forecast range or no predictions available.")

    mapped_preds = {feature_name_map.get(k, k): v for k, v in preds.items()}

    return PredictionResponse(date=date, predictions=mapped_preds)
