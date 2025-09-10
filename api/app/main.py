from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routers import prediction

app = FastAPI(
    title="Hoax Detection API",
    description="API for detecting hoaxes in Indonesian text using IndoBERT/XLM-R",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Include routers
app.include_router(prediction.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Hoax Detection API",
        "docs": "/docs",
        "endpoints": {
            "predict": "/predict"
        }
    }
