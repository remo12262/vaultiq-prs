from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers import prs
import logging

logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("PRS Service avviato")
    yield
    logging.info("PRS Service spento")

app = FastAPI(
    title="VaultIQ PRS Engine",
    description="Personal Risk Score - algoritmo proprietario",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prs.router, prefix="/api/v1/prs", tags=["PRS"])

@app.get("/health")
async def health():
    return {"status": "ok", "service": "PRS Engine", "version": "1.0.0"}

@app.get("/ping")
async def ping():
    return {"pong": True}
