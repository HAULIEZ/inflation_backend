from fastapi import APIRouter, HTTPException, Response, status
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt
from app.util.database_connection import get_db_connection

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request body model
class SignInRequest(BaseModel):
    email: EmailStr
    password: str

# Response models
class UserInfo(BaseModel):
    user_id: int
    first_name: str | None
    last_name: str | None
    email: EmailStr

class SignInResponse(BaseModel):
    success: bool
    message: str
    user: UserInfo

@router.post(
    "/signin",
    response_model=SignInResponse,
    summary="Sign in and return user info",
    status_code=status.HTTP_200_OK,
)
def signin(response: Response, credentials: SignInRequest):
    conn, cur = get_db_connection()
    if not conn or not cur:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        email = credentials.email.lower()

        cur.execute(
            """
            SELECT user_id, email, first_name, last_name, password_hash, is_active
            FROM users
            WHERE email = %s
            """,
            (email,),
        )
        user = cur.fetchone()

        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        if not user["is_active"]:
            raise HTTPException(status_code=403, detail="User account is inactive")

        if not bcrypt.verify(credentials.password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        # Removed last_login update because it does not exist in the schema

        return SignInResponse(
            success=True,
            message="Login successful",
            user=UserInfo(
                user_id=user["user_id"],
                email=user["email"],
                first_name=user["first_name"],
                last_name=user["last_name"],
            ),
        )

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error during login: {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
