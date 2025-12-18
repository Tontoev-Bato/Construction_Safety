import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Safety Monitoring API"
    model_path: str = os.getenv("MODEL_PATH", "weights/best.pt")
    confidence_threshold: float = 0.3
    api_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"

settings = Settings()