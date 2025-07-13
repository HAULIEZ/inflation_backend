from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib
import math

# Import your utility function here
from app.util.database_connection import get_db_connection

router = APIRouter()

# Load models and weights once at startup
rf_model = joblib.load("app/model_api/models/best_rf_model.pkl")
xgb_model = joblib.load("app/model_api/models/best_xgb_model.pkl")
ensemble_data = joblib.load("app/model_api/models/ensemble_data.pkl")

weight_rf = ensemble_data["weight_rf"]
weight_xgb = ensemble_data["weight_xgb"]

def clean_nan(obj):
    """Recursively replace NaN floats with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        else:
            return obj
    else:
        return obj

@router.get("/feature-analysis/{user_id}", tags=["Feature Analysis using models"])
def analyze_features_and_predict(user_id: int):
    try:
        conn, cur = get_db_connection()
        if conn is None or cur is None:
            raise HTTPException(status_code=500, detail="Database connection failed")

        # Fetch last 12 feature input records for user
        cur.execute("""
            SELECT * FROM user_feature_inputs
            WHERE user_id = %s
            ORDER BY input_date DESC
            LIMIT 12;
        """, (user_id,))
        records = cur.fetchall()

        if not records:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="No feature data found for this user.")

        df = pd.DataFrame(records)

        db_to_model_feature_map = {
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
            "day": "Day"
        }

        renamed_df = df.rename(columns=db_to_model_feature_map)

        model_columns = list(rf_model.feature_names_in_)
        feature_df = renamed_df.loc[:, renamed_df.columns.isin(model_columns)].copy()
        feature_df = feature_df[model_columns]  # reorder columns

        desc_stats = feature_df.describe().T.reset_index()
        desc_stats.columns = ["feature", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        desc_stats = desc_stats.round(2).to_dict(orient="records")
        desc_stats = clean_nan(desc_stats)

        recent_trend = renamed_df[model_columns + ["Year", "Month", "Day"]].to_dict(orient="records")
        recent_trend = clean_nan(recent_trend)

        rf_preds = rf_model.predict(feature_df).round(4).tolist()
        xgb_preds = xgb_model.predict(feature_df).round(4).tolist()
        ensemble_preds = [
            round((weight_rf * rf + weight_xgb * xgb) / (weight_rf + weight_xgb), 4)
            for rf, xgb in zip(rf_preds, xgb_preds)
        ]

        cur.close()
        conn.close()

        return {
            "summary_statistics": desc_stats,
            "recent_feature_trend": recent_trend,
            "predictions": {
                "random_forest": rf_preds,
                "xgboost": xgb_preds,
                "ensemble": ensemble_preds
            },
            "total_entries": len(records)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
