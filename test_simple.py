
import requests
import base64
import json
import time

def simple_test():
    print("🚀 Starting simple API test")
    print("=" * 50)
    
    BASE_URL = "http://localhost:8000"
    
    # 1. Health check
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/detection/health", timeout=5)
        print(f"✅ Status: {response.status_code}")
        
        health_data = response.json()
        print(f"   API Version: {health_data.get('version')}")
        print(f"   Model loaded: {health_data.get('model_loaded')}")
        print(f"   Status: {health_data.get('status')}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("   Make sure FastAPI server is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    # 2. Test with a simple image
    print("\n2. Testing with simple request...")
    
    # Создаем простой base64 изображение (маленькое черное)
    small_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/detection/detect/image",
            json={
                "image_base64": f"data:image/png;base64,{small_image}",
                "confidence_threshold": 0.3
            },
            timeout=10
        )
        
        print(f"✅ Detection response: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            print(f"   People detected: {result['safety_status']['persons_count']}")
            print(f"   Inference time: {result.get('inference_time', 0):.3f}s")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    
    # 3. Test file upload 
    print("\n3. Testing file upload...")
    
    try:
        with open('test_image.jpg', 'rb') as f:
            files = {'file': ('test.jpg', f, 'image/jpeg')}
            response = requests.post(
                f"{BASE_URL}/api/v1/detection/detect/upload",
                files=files,
                params={'confidence_threshold': 0.3},
                timeout=15
            )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ File upload successful!")
            print(f"   People: {result['safety_status']['persons_count']}")
            print(f"   Helmets: {result['safety_status']['helmets_count']}")
            print(f"   Vests: {result['safety_status']['vests_count']}")
            
            if result['safety_status']['violations']:
                print(f"   ⚠ Violations: {', '.join(result['safety_status']['violations'])}")
            else:
                print("   ✅ No violations detected")
                
            print(f"   ⏱ Time: {result['inference_time']:.3f}s")
            
            # Saving result
            if 'annotated_image' in result:
                try:
                    image_data = result['annotated_image'].split(',')[1]
                    with open('result_annotated.jpg', 'wb') as f:
                        f.write(base64.b64decode(image_data))
                    print("   💾 Annotated image saved as 'result_annotated.jpg'")
                except:
                    print("   ℹ Could not save annotated image")
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            
    except FileNotFoundError:
        print("ℹ No test_image.jpg found, skipping file upload test")
    except Exception as e:
        print(f"❌ Upload error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    return True

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    time.sleep(2)  
    simple_test()