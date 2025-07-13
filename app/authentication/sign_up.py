from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr, constr
from passlib.hash import bcrypt
from app.util.database_connection import get_db_connection

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request model
class SignupRequest(BaseModel):
    first_name: constr(min_length=1)  # type: ignore
    last_name: constr(min_length=1)   # type: ignore
    email: EmailStr
    password: constr(min_length=6)    # type: ignore

# Response model
class UserInfo(BaseModel):
    user_id: int
    email: EmailStr
    first_name: str
    last_name: str

class SignupResponse(BaseModel):
    success: bool
    message: str
    user: UserInfo

@router.post(
    "/signup",
    response_model=SignupResponse,
    summary="Register a new user",
    status_code=status.HTTP_201_CREATED,
)
def signup(request: SignupRequest):
    conn, cur = get_db_connection()
    if not conn or not cur:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        email = request.email.lower()

        # Check if email is already registered
        cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered")

        # Hash password
        hashed_password = bcrypt.hash(request.password)

        # Insert new user
        cur.execute(
            """
            INSERT INTO users (first_name, last_name, email, password_hash)
            VALUES (%s, %s, %s, %s)
            RETURNING user_id, email, first_name, last_name
            """,
            (
                request.first_name.strip(),
                request.last_name.strip(),
                email,
                hashed_password,
            ),
        )
        new_user = cur.fetchone()
        conn.commit()

        if not new_user:
            raise HTTPException(status_code=500, detail="User creation failed")

        return SignupResponse(
            success=True,
            message="User registered successfully",
            user=UserInfo(
                user_id=new_user["user_id"],
                email=new_user["email"],
                first_name=new_user["first_name"],
                last_name=new_user["last_name"],
            ),
        )

    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Error during registration: {e}")

    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
