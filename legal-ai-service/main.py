"""
NyayaDoot Legal AI Service — slim entrypoint.
All logic lives in services/ and routes/ for easy debugging.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config import CORS_ORIGINS, HOST, PORT

# Import routes (this also triggers model loading via services.model_loader)
from routes.legal import router as legal_router

app = FastAPI(title="NyayaDoot Legal AI", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(legal_router)


if __name__ == "__main__":
    uvicorn.run("main:app", host=HOST, port=PORT, reload=True)