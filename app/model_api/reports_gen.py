from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Optional, List
import os
import pandas as pd
import pickle
from pathlib import Path

# === Paths ===
BASE_DIR = Path("app") / "model_api"
MODEL_PATH = BASE_DIR / "inflation_model" / "prophet_inflation_model_final.pkl"
SCALER_PATH = BASE_DIR / "inflation_model" / "regressor_scaler.pkl"
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "my_data.csv"

router = APIRouter()

# === Load model and scaler ===
with open(MODEL_PATH, "rb") as f:
    inflation_model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

REGRESSORS = [
    "OverallCPI", "FoodCPI", "FoodInflation", "NonFoodCPI", "NonFoodInflation",
    "ExchangeRate", "Bank Rate/ Policy Rate", "Lending Rate",
    "Interest Rate Spread (base lending rate less 3-month deposit rate)"
]

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
    "OverallCPI": "OverallCPI"
}

# === Forecaster Class ===
class ProphetForecaster:
    def __init__(self):
        self.models = {}
        self.forecasts = {}

    def load_all_models(self, load_dir):
        if not os.path.exists(load_dir):
            print(f"Model directory {load_dir} does not exist.")
            return
        for filename in os.listdir(load_dir):
            if filename.startswith("prophet_model_") and filename.endswith(".pkl"):
                feature_name = filename[len("prophet_model_"):-4]
                with open(os.path.join(load_dir, filename), "rb") as f:
                    self.models[feature_name] = pickle.load(f)

    def create_forecasts(self, until_date):
        self.forecasts.clear()
        for feat, model in self.models.items():
            last_date = model.history["ds"].max()
            target = pd.to_datetime(until_date)
            periods = max((target.year - last_date.year) * 12 + (target.month - last_date.month) + 1, 36)
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

# === Load data and models ===
forecaster = ProphetForecaster()
forecaster.load_all_models(str(MODEL_DIR))

df_data = pd.read_csv(DATA_PATH, parse_dates=["ds"])
df_data["ds"] = pd.to_datetime(df_data["ds"])

# === Response Models ===
class DatePrediction(BaseModel):
    date: str
    inflation: float
    inflation_lower: float
    inflation_upper: float
    source: str
    features: Dict[str, Optional[float]]

class RangePredictionResponse(BaseModel):
    start_date: str
    end_date: str
    results: List[DatePrediction]

# === Unified Report Endpoint ===
@router.get("/unified-report", response_model=RangePredictionResponse)
def generate_report(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD")
):
    try:
        # Adjust input dates to first day of the month
        start = pd.to_datetime(start_date).replace(day=1)
        end = pd.to_datetime(end_date).replace(day=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    if start > end:
        raise HTTPException(status_code=400, detail="Start date must be before or equal to end date.")

    date_range = pd.date_range(start=start, end=end, freq="MS")
    forecaster.create_forecasts(until_date=end)

    results = []

    for current_date in date_range:
        date_str = current_date.strftime("%Y-%m-%d")
        row = df_data[df_data["ds"].dt.date == current_date.date()]
        if not row.empty:
            features = {col: float(row.iloc[0][col]) for col in REGRESSORS}
            source = "historical data"
        else:
            raw_preds = forecaster.predict_on_date(date_str)
            features = {}
            for k, v in raw_preds.items():
                mapped_name = FEATURE_NAME_MAP.get(k, k)
                if v is None:
                    raise HTTPException(status_code=404, detail=f"Missing forecast for feature: {mapped_name}")
                features[mapped_name] = v
            source = "forecast"

        input_df = pd.DataFrame([features])
        input_df["ds"] = [current_date]
        input_df[REGRESSORS] = scaler.transform(input_df[REGRESSORS])
        forecast = inflation_model.predict(input_df).iloc[0]

        results.append(DatePrediction(
            date=date_str,
            inflation=forecast["yhat"],
            inflation_lower=forecast["yhat_lower"],
            inflation_upper=forecast["yhat_upper"],
            source=source,
            features=features
        ))

    return RangePredictionResponse(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        results=results
    )
