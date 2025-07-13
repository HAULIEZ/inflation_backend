from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal, List, Dict
import pandas as pd
import joblib
from datetime import datetime

# Import the reusable DB connection function
from app.util.database_connection import get_db_connection

router = APIRouter()

# Load models and weights (make sure paths are correct)
best_rf_model = joblib.load("app/model_api/models/best_rf_model.pkl")
best_xgb_model = joblib.load("app/model_api/models/best_xgb_model.pkl")
ensemble_data = joblib.load("app/model_api/models/ensemble_data.pkl")
weight_rf = ensemble_data["weight_rf"]
weight_xgb = ensemble_data["weight_xgb"]

# Map snake_case to training feature names for input construction
feature_name_map = {
    "advances_by_banks": "Advances by banks",
    "auto_sales": "Auto Sales",
    "consumer_confidence_index": "Consumer Confidence Index",
    "call_money_rate_end_of_period": "Call Money Rate (End of Period)",
    "imf_commodity_prices": "IMF Commodity Prices",
    "national_consumer_price_index": "National Consumer Price Index",
    "deposit_rate": "Deposit Rate",
    "economic_policy_uncertainty": "Economic Policy Uncertainty",
    "stock_exchange_100_index": "Stock Exchange 100 Index",
    "one_year_interest_rate": "1 Year Interest Rate",
    "lending_rate": "Lending Rate",
    "international_oil_prices": "International Oil Prices",
    "sbp_policy_rate": "SBP Policy Rate",
    "public_sector_borrowing": "Public Sector Borrowing",
    "real_output_quantum_index_of_large_scale_manufacturing": "Real Output - Quantum Index of Large-Scale Manufacturing Industries",
    "real_effective_exchange_rate": "Real Effective Exchange Rate",
    "interest_rate_spread": "Interest Rate Spread",
    "inflation_expectations": "Inflation Expectations",
    "banking_activity_index": "Banking Activity Index",
    "stock_market_volatility": "Stock Market Volatility",
    "commodity_price_index": "Commodity Price Index",
    "year": "Year",
    "month": "Month",
    "day": "Day",
}

class PredictionRequest(BaseModel):
    user_id: int
    model: Literal["rf", "xgb", "ensemble"]
    year: int

class MonthlyPrediction(BaseModel):
    month: str
    predicted_inflation: float
    confidence_interval: str  # e.g. "1.80 – 2.20"

class DetailedPredictionResponse(BaseModel):
    year: int
    predictions: List[MonthlyPrediction]

def get_latest_features_for_user_year(user_id: int, year: int) -> Dict:
    """Query DB for latest user features up to and including the specified year."""
    query = """
        SELECT *
        FROM user_feature_inputs
        WHERE user_id = %s AND year <= %s
        ORDER BY year DESC, month DESC, day DESC
        LIMIT 1
    """
    conn, cur = get_db_connection()
    if conn is None or cur is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur.execute(query, (user_id, year))
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="No feature data found for this user and year.")
        return dict(row)
    finally:
        cur.close()
        conn.close()

@router.post("/predict-detailed", response_model=DetailedPredictionResponse, tags=["Prediction based on features entered"])
def predict_detailed(request: PredictionRequest):
    # Step 1: Fetch latest features for user up to the year
    features_row = get_latest_features_for_user_year(request.user_id, request.year)
    
    # Step 2: Prepare monthly input data for the entire year
    monthly_data = []
    for month_num in range(1, 13):
        # Copy features, update year, month, day
        feature_copy = features_row.copy()
        feature_copy["year"] = request.year
        feature_copy["month"] = month_num
        feature_copy["day"] = 1  # just first day
        
        # Remove keys that are not model features
        for k in ["input_id", "user_id", "input_date"]:
            feature_copy.pop(k, None)
        
        # Map keys to original training feature names
        mapped_features = {feature_name_map[k]: feature_copy[k] for k in feature_name_map.keys()}
        
        monthly_data.append(mapped_features)
    
    # Build DataFrame and order columns like model expects
    columns_order = list(best_rf_model.feature_names_in_)
    input_df = pd.DataFrame(monthly_data)[columns_order]
    
    # Step 3: Predict based on model choice
    if request.model == "rf":
        preds = best_rf_model.predict(input_df)
    elif request.model == "xgb":
        preds = best_xgb_model.predict(input_df)
    elif request.model == "ensemble":
        rf_preds = best_rf_model.predict(input_df)
        xgb_preds = best_xgb_model.predict(input_df)
        total_weight = weight_rf + weight_xgb
        preds = (weight_rf * rf_preds + weight_xgb * xgb_preds) / total_weight
    else:
        raise HTTPException(status_code=400, detail="Invalid model selected.")

    # Step 4: Create confidence intervals (dummy ±0.15 for demo, replace with your logic)
    ci_margin = 0.15
    results = []
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i, pred in enumerate(preds):
        lower = round(pred - ci_margin, 2)
        upper = round(pred + ci_margin, 2)
        results.append(MonthlyPrediction(
            month=month_names[i],
            predicted_inflation=round(pred, 2),
            confidence_interval=f"{lower} – {upper}"
        ))

    return DetailedPredictionResponse(
        year=request.year,
        predictions=results
    )
