from fastapi import APIRouter, HTTPException, status
import psycopg2
from pydantic import BaseModel, Field, conint
from typing import Optional
from psycopg2.extras import RealDictCursor
from app.util.database_connection import get_db_connection

router = APIRouter()

# Pydantic model for input validation matching DB columns
class FeatureInput(BaseModel):
    user_id: int = Field(..., gt=0)
    year: conint(ge=1900, le=2100)  # type: ignore
    month: conint(ge=1, le=12)  # type: ignore
    day: conint(ge=1, le=31)  # type: ignore

    advances_by_banks: Optional[float] = None
    auto_sales: Optional[float] = None
    consumer_confidence_index: Optional[float] = None
    call_money_rate_end_of_period: Optional[float] = None
    imf_commodity_prices: Optional[float] = None
    national_consumer_price_index: Optional[float] = None
    deposit_rate: Optional[float] = None
    economic_policy_uncertainty: Optional[float] = None
    stock_exchange_100_index: Optional[float] = None
    one_year_interest_rate: Optional[float] = None
    lending_rate: Optional[float] = None
    international_oil_prices: Optional[float] = None
    sbp_policy_rate: Optional[float] = None
    public_sector_borrowing: Optional[float] = None
    real_output_quantum_index_of_large_scale_manufacturing: Optional[float] = None
    real_effective_exchange_rate: Optional[float] = None

    interest_rate_spread: Optional[float] = None
    inflation_expectations: Optional[float] = None
    banking_activity_index: Optional[float] = None
    stock_market_volatility: Optional[float] = None
    commodity_price_index: Optional[float] = None

@router.post("/feature-inputs/", status_code=status.HTTP_201_CREATED, tags=["user Management"])
def insert_feature_input(input_data: FeatureInput):
    insert_sql = """
        INSERT INTO user_feature_inputs (
            user_id, year, month, day,
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
        )
        VALUES (
            %(user_id)s, %(year)s, %(month)s, %(day)s,
            %(advances_by_banks)s,
            %(auto_sales)s,
            %(consumer_confidence_index)s,
            %(call_money_rate_end_of_period)s,
            %(imf_commodity_prices)s,
            %(national_consumer_price_index)s,
            %(deposit_rate)s,
            %(economic_policy_uncertainty)s,
            %(stock_exchange_100_index)s,
            %(one_year_interest_rate)s,
            %(lending_rate)s,
            %(international_oil_prices)s,
            %(sbp_policy_rate)s,
            %(public_sector_borrowing)s,
            %(real_output_quantum_index_of_large_scale_manufacturing)s,
            %(real_effective_exchange_rate)s,
            %(interest_rate_spread)s,
            %(inflation_expectations)s,
            %(banking_activity_index)s,
            %(stock_market_volatility)s,
            %(commodity_price_index)s
        )
        RETURNING input_id;
    """

    conn, cur = get_db_connection()
    if conn is None or cur is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        cur.execute(insert_sql, input_data.dict())
        inserted = cur.fetchone()
        conn.commit()
        return {"message": "Feature input inserted successfully", "input_id": inserted["input_id"]}
    except psycopg2.IntegrityError as e:
        raise HTTPException(status_code=400, detail=f"Database integrity error: {e.pgerror}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        cur.close()
        conn.close()
