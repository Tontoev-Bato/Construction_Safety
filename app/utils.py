import base64
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import json

def base64_to_image(image_base64: str) -> np.ndarray:
    """Conversation base64 to numpy array"""
    try:
       
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decoding base64
        image_data = base64.b64decode(image_base64)
        
        # Conversation to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
           
            image = Image.open(BytesIO(image_data))
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        return image
    except Exception as e:
        raise ValueError(f"Error converting base64 to image: {e}")

def image_to_base64(image: np.ndarray) -> str:
    """Confersation numpy array to base64"""
    try:
        # Conversation BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Конвертация в base64
        _, buffer = cv2.imencode('.jpg', image_rgb)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return f"data:image/jpeg;base64,{image_base64}"
    except Exception as e:
        raise ValueError(f"Error converting image to base64: {e}")

def draw_detections(image: np.ndarray, detections: list, safety_status: dict) -> np.ndarray:
    """Rendering detections on an image"""
    # Copying image
    image_with_boxes = image.copy()
    
    
    colors = {
        'person': (0, 255, 0),      # green
        'helmet': (255, 0, 0),      # blue
        'vest': (0, 0, 255),        # red
        'no-helmet': (0, 255, 255), # yellow
        'no-vest': (255, 0, 255)    # perple
    }
    
    # bounding boxes
    for det in detections:
        class_name = det['class_name']
        confidence = det['confidence']
        bbox = det['bbox']
        
        color = colors.get(class_name, (255, 255, 255))
        
        
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(image_with_boxes, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    y_offset = 30
    
    # Status
    if safety_status['person_detected']:
        status_text = "[SAFE]" if safety_status['is_compliant'] else "[VIOLATION]"
        color = (0, 255, 0) if safety_status['is_compliant'] else (0, 0, 255)
    else:
        status_text = "[NO PEOPLE]"
        color = (255, 255, 255)
    
    cv2.putText(image_with_boxes, status_text, (10, y_offset),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    y_offset += 30
    
    if safety_status['person_detected']:
        cv2.putText(image_with_boxes, f"People: {safety_status['persons_count']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25
        cv2.putText(image_with_boxes, f"Helmets: {safety_status['helmets_count']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25
        cv2.putText(image_with_boxes, f"Vests: {safety_status['vests_count']}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y_offset += 25
    
    for violation in safety_status['violations']:
        cv2.putText(image_with_boxes, f"[!] {violation}",
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        y_offset += 25
    
    return image_with_boxes
