# routes/users.py
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from app.util.database_connection import get_db_connection

router = APIRouter(prefix="/users", tags=["Authentication"])

# Response models
class UserInfo(BaseModel):
    user_id: int
    first_name: str
    last_name: str
    email: EmailStr

class UserResponse(BaseModel):
    success: bool
    message: str
    user: UserInfo

@router.get(
    "/{user_id}",
    response_model=UserResponse,
    summary="Get user info by ID",
    status_code=status.HTTP_200_OK,
)
def get_user_by_id(user_id: int):
    conn, cur = get_db_connection()
    if not conn or not cur:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        print(f"🔎 Fetching user with ID: {user_id}")

        cur.execute(
            """
            SELECT user_id, first_name, last_name, email
            FROM users
            WHERE user_id = %s
            """,
            (user_id,),
        )
        user = cur.fetchone()
        print("📦 User fetched:", user)

        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        return UserResponse(
            success=True,
            message="User retrieved successfully",
            user=UserInfo(
                user_id=user["user_id"],
                first_name=user["first_name"],
                last_name=user["last_name"],
                email=user["email"],
            ),
        )

    except Exception as e:
        print("❌ Exception occurred:", repr(e))
        raise HTTPException(status_code=500, detail=f"Error fetching user: {repr(e)}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
