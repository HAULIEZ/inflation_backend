# routes/db_test.py
from fastapi import APIRouter
from app.util.database_connection import get_db_connection




router = APIRouter(prefix="/db", tags=["Database-Test"])



@router.get("/test", summary="Test DB Connection")
def test_db_connection():
    conn, cur = get_db_connection()
    if conn and cur:
        try:
            cur.execute("SELECT 1;")
            result = cur.fetchone()
            cur.close()
            conn.close()
            return {
                "success": True,
                "message": "Database connection successful.",
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Query failed: {e}"
            }
    else:
        return {
            "success": False,
            "message": "Failed to connect to the database."
        }
