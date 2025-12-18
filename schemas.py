from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class DetectionItem(BaseModel):
    class_name: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class SafetyStatus(BaseModel):
    person_detected: bool
    has_helmet: bool
    has_vest: bool
    is_compliant: bool
    violations: List[str]
    persons_count: int
    helmets_count: int
    vests_count: int
    no_helmets_count: int
    no_vests_count: int

# Базовый ответ с JSON
class DetectionResponse(BaseModel):
    status: str
    message: str
    detections: List[DetectionItem]
    safety_status: SafetyStatus
    frame_size: Optional[Dict[str, int]] = None
    inference_time: Optional[float] = None

# Ответ с изображением в base64
class DetectionResponseWithImage(BaseModel):
    status: str
    message: str
    detections: List[DetectionItem]
    safety_status: SafetyStatus
    frame_size: Optional[Dict[str, int]] = None
    inference_time: Optional[float] = None
    image_base64: str  # Добавляем поле для изображения

class ImageBase64(BaseModel):
    image_base64: str
    confidence_threshold: Optional[float] = 0.3

class HealthCheck(BaseModel):
    status: str
    version: str
    model_loaded: bool
