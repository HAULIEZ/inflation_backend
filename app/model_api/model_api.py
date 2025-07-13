from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Literal
import pandas as pd
import joblib

router = APIRouter()

# Load models
best_rf_model = joblib.load("app/model_api/models/best_rf_model.pkl")
best_xgb_model = joblib.load("app/model_api/models/best_xgb_model.pkl")
ensemble_data = joblib.load("app/model_api/models/ensemble_data.pkl")

weight_rf = ensemble_data["weight_rf"]
weight_xgb = ensemble_data["weight_xgb"]

class InflationFeatures(BaseModel):
    advances_by_banks: float
    auto_sales: float
    consumer_confidence_index: float
    call_money_rate_end_of_period: float
    imf_commodity_prices: float
    national_consumer_price_index: float
    deposit_rate: float
    economic_policy_uncertainty: float
    stock_exchange_100_index: float
    one_year_interest_rate: float
    lending_rate: float
    international_oil_prices: float
    sbp_policy_rate: float
    public_sector_borrowing: float
    real_output_quantum_index_of_large_scale_manufacturing_industries: float
    real_effective_exchange_rate: float
    interest_rate_spread: float
    inflation_expectations: float
    banking_activity_index: float
    stock_market_volatility: float
    commodity_price_index: float
    year: int
    month: int
    day: int

class PredictionRequest(BaseModel):
    model: Literal["rf", "xgb", "ensemble"]
    features: InflationFeatures

@router.post(
    "/predict",
    summary="Predict inflation using selected model",tags=["Inflation Prediction"],
    response_description="Predicted inflation value",
)
def predict_inflation(request: PredictionRequest):
    """
    Predict inflation based on provided features using one of the three models:

    - `rf`: Random Forest model
    - `xgb`: XGBoost model
    - `ensemble`: Weighted ensemble of RF and XGB models

    Provide all required features exactly as specified in the `features` object.

    **Example request JSON:**

    ```json
    {
      "model": "ensemble",
      "features": {
        "advances_by_banks": 0.12,
        "auto_sales": 0.45,
        "consumer_confidence_index": 0.67,
        "call_money_rate_end_of_period": 0.23,
        "imf_commodity_prices": 0.56,
        "national_consumer_price_index": 0.78,
        "deposit_rate": 0.34,
        "economic_policy_uncertainty": 0.89,
        "stock_exchange_100_index": 0.12,
        "one_year_interest_rate": 0.45,
        "lending_rate": 0.67,
        "international_oil_prices": 0.23,
        "sbp_policy_rate": 0.56,
        "public_sector_borrowing": 0.78,
        "real_output_quantum_index_of_large_scale_manufacturing_industries": 0.34,
        "real_effective_exchange_rate": 0.89,
        "interest_rate_spread": 0.12,
        "inflation_expectations": 0.45,
        "banking_activity_index": 0.67,
        "stock_market_volatility": 0.23,
        "commodity_price_index": 0.56,
        "year": 2025,
        "month": 7,
        "day": 9
      }
    }
    ```

    The response will indicate which model was used and the predicted inflation value.
    """
    try:
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
            "real_output_quantum_index_of_large_scale_manufacturing_industries": "Real Output - Quantum Index of Large-Scale Manufacturing Industries",
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

        input_dict = {feature_name_map[k]: v for k, v in request.features.dict().items()}

        columns_order = list(best_rf_model.feature_names_in_)

        input_df = pd.DataFrame([{col: input_dict[col] for col in columns_order}])

        if request.model == "rf":
            prediction = best_rf_model.predict(input_df)[0]
        elif request.model == "xgb":
            prediction = best_xgb_model.predict(input_df)[0]
        elif request.model == "ensemble":
            rf_pred = best_rf_model.predict(input_df)[0]
            xgb_pred = best_xgb_model.predict(input_df)[0]
            total_weight = weight_rf + weight_xgb
            prediction = (weight_rf * rf_pred + weight_xgb * xgb_pred) / total_weight
        else:
            raise HTTPException(status_code=400, detail="Invalid model selected.")

        return {
            "model_used": request.model,
            "predicted_inflation": round(float(prediction), 4)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
