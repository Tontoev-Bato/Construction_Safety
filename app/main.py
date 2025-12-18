from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.routers import detection
from app.config import settings
from app.models import safety_monitor
import uvicorn
import os

# Создаем FastAPI приложение
app = FastAPI(
    title=settings.app_name,
    description="API for monitoring safety on construction sites",
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутер
app.include_router(detection.router)

# Создаем папку для временных файлов 
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    """API Root Page"""
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.api_version,
        "docs": "/docs",
        "health_check": "/api/v1/detection/health"
    }

@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения"""
    print(f"Starting {settings.app_name} v{settings.api_version}")
    print(f"Model path: {settings.model_path}")
    # Проверяем загрузку модели
    if safety_monitor.is_model_loaded():
        print(" Model loaded successfully")
    else:
        print(" Model failed to load")

if __name__ == "__main__":
    print(" tarting server on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )