from fastapi import APIRouter, HTTPException
from psycopg2.extras import RealDictCursor
from app.util.database_connection import get_db_connection

router = APIRouter()

@router.get("/feature-inputs/latest/{user_id}",tags=["User Management"])
def get_latest_feature_input_for_user(user_id: int):
    query = """
        SELECT
            input_id,
            user_id,
            input_date,
            year,
            month,
            day,
            advances_by_banks,
            auto_sales,
            consumer_confidence_index,
            call_money_rate_end_of_period,
            imf_commodity_prices,
            national_consumer_price_index,
            deposit_rate,
            economic_policy_uncertainty,
            stock_exchange_100_index,
            one_year_interest_rate,
            lending_rate,
            international_oil_prices,
            sbp_policy_rate,
            public_sector_borrowing,
            real_output_quantum_index_of_large_scale_manufacturing,
            real_effective_exchange_rate,
            interest_rate_spread,
            inflation_expectations,
            banking_activity_index,
            stock_market_volatility,
            commodity_price_index
        FROM user_feature_inputs
        WHERE user_id = %s
        ORDER BY input_date DESC
        LIMIT 1;
    """

    conn, cur = get_db_connection()
    if conn is None or cur is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur.execute(query, (user_id,))
        row = cur.fetchone()
        if row:
            return row
        else:
            raise HTTPException(status_code=404, detail="No feature input found for this user.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        cur.close()
        conn.close()
