import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Dict, Any
from app.config import settings


class SafetyMonitor:
    def __init__(self):
        self.model = None
        self.classes = ['helmet', 'no-helmet', 'no-vest', 'person', 'vest']
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        try:
            print(f"Loading model from: {settings.model_path}")
            self.model = YOLO(settings.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def check_safety_compliance(self, detections: List[Dict], confidence_threshold: float = None):
        """Check safety compliance"""
        if confidence_threshold is None:
            confidence_threshold = settings.confidence_threshold
        
        safety_status = {
            'person_detected': False,
            'has_helmet': False,
            'has_vest': False,
            'is_compliant': False,
            'violations': [],
            'persons_count': 0,
            'helmets_count': 0,
            'vests_count': 0,
            'no_helmets_count': 0,
            'no_vests_count': 0
        }
        
        # Process detections
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            if confidence < confidence_threshold:
                continue
                
            if class_name == 'person':
                safety_status['person_detected'] = True
                safety_status['persons_count'] += 1
            elif class_name == 'helmet':
                safety_status['has_helmet'] = True
                safety_status['helmets_count'] += 1
            elif class_name == 'vest':
                safety_status['has_vest'] = True
                safety_status['vests_count'] += 1
            elif class_name == 'no-helmet':
                safety_status['no_helmets_count'] += 1
            elif class_name == 'no-vest':
                safety_status['no_vests_count'] += 1
        
        # Check violations
        if safety_status['person_detected']:
            if safety_status['no_helmets_count'] > 0 or not safety_status['has_helmet']:
                safety_status['violations'].append('No helmet')
            if safety_status['no_vests_count'] > 0 or not safety_status['has_vest']:
                safety_status['violations'].append('No vest')
        
        # Check compliance
        if safety_status['person_detected']:
            safety_status['is_compliant'] = (
                safety_status['has_helmet'] and
                safety_status['has_vest'] and
                len(safety_status['violations']) == 0
            )
        
        return safety_status
    
    def predict(self, image: np.ndarray, confidence_threshold: float = None):
        """Perform detection on image"""
        if confidence_threshold is None:
            confidence_threshold = settings.confidence_threshold
        
        start_time = time.time()
        
        # Perform prediction
        results = self.model(image, conf=confidence_threshold, verbose=False)
        
        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    confidence = boxes.conf[i].item()
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    
                    detections.append({
                        'class_id': class_id,
                        'class_name': self.classes[class_id],
                        'confidence': float(confidence),
                        'bbox': bbox
                    })
        
        inference_time = time.time() - start_time
        
        # Safety check
        safety_status = self.check_safety_compliance(detections, confidence_threshold)
        
        return {
            'detections': detections,
            'safety_status': safety_status,
            'inference_time': inference_time,
            'frame_size': {
                'height': image.shape[0],
                'width': image.shape[1],
                'channels': image.shape[2] if len(image.shape) > 2 else 1
            }
        }
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.model is not None


safety_monitor = SafetyMonitor()
