from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.Database import test_connection
from app.authentication import get_user, sign_in, sign_up
from app.management import feature_insert, get_features
from app.model_api import analysis, model_api, model_new_feature

# Initialize FastAPI app
app = FastAPI(
    title="Inflation FastAPI App",
    description="Provides analytical inflation insights",
    version="1.0.0"
)

# ✅ CORS settings
origins = [
    "http://localhost:3000",     # your frontend dev server
    "http://127.0.0.1:3000",     # alternative local frontend
    "https://r3lbzzj7-5000.uks1.devtunnels.ms/"  # production frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,         # ✅ This enables sending/receiving cookies
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the inflation FastAPI app!"}

# Database connection test route
app.include_router(test_connection.router)

# Authentication routes
app.include_router(sign_in.router)
app.include_router(sign_up.router)
app.include_router(get_user.router)

# Feature data management routes
app.include_router(feature_insert.router)
app.include_router(get_features.router)

# model API routes 
app.include_router(model_api.router)
app.include_router(model_new_feature.router)
app.include_router(analysis.router)