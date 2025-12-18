import requests
import base64
import json
import cv2
import numpy as np

def test_api():
    # URL API
    BASE_URL = "http://localhost:8000"
    
    # 1. Checking health

    print("1. Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/v1/detection/health")
    print(f"Health check: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    
    # 2. testing image
    print("\n2. Testing detection with uploaded image...")
    
    # Reading image
    image_path = "test_image.jpg"  
    
    with open(image_path, "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/detection/detect/upload",
            files=files,
            params={"confidence_threshold": 0.3}
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Detection successful!")
        print(f"People detected: {result['safety_status']['persons_count']}")
        print(f"Violations: {result['safety_status']['violations']}")
        print(f"Inference time: {result['inference_time']:.3f}s")
        
        # Saving annotation
        if 'annotated_image' in result:
            image_data = result['annotated_image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            
            with open('annotated_result.jpg', 'wb') as f:
                f.write(image_bytes)
            print("Annotated image saved as 'annotated_result.jpg'")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_api()