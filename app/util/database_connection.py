# app/util/database_connection.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    conn_str = os.getenv("DATABASE_URL")
    if not conn_str:
        raise ValueError("DATABASE_URL environment variable is not set")
    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        return conn, cur
    except Exception as e:
        print(f"❌ Error connecting to the database: {e}")
        return None, None
