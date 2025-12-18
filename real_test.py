import requests
import base64
import json
import cv2
import numpy as np

def real_world_test(image_path="real_test.jpg"):
    print("🔍 Real World API Test with Actual Image")
    print("=" * 60)
    
    BASE_URL = "http://localhost:8000"
    
    
    print("\n📡 1. Checking API health...")
    try:
        health = requests.get(f"{BASE_URL}/api/v1/detection/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ API unhealthy: {health.status_code}")
            return
        
        health_data = health.json()
        print(f"   ✅ {health_data['status'].upper()}")
        print(f"   📋 Version: {health_data['version']}")
        print(f"   🤖 Model: {'LOADED ✅' if health_data['model_loaded'] else 'NOT LOADED ❌'}")
        
        if not health_data['model_loaded']:
            print("   ⚠ Cannot proceed - model not loaded")
            return
            
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure server is running: python -m app.main")
        return
    
    
    print(f"\n📷 2. Analyzing image: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/jpeg")}
            
            print("   ⏳ Sending to safety monitoring API...")
            response = requests.post(
                f"{BASE_URL}/api/v1/detection/detect/upload",
                files=files,
                params={"confidence_threshold": 0.3},
                timeout=30  
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Analysis complete! ({result['inference_time']:.2f}s)")
            
            # Вывод результатов
            print(f"\n📊 3. Safety Analysis Results:")
            print(f"   {'─' * 40}")
            
            stats = result['safety_status']
            
            print(f"   👥 Persons detected: {stats['persons_count']}")
            print(f"   🪖 Safety helmets: {stats['helmets_count']}")
            print(f"   🦺 Safety vests: {stats['vests_count']}")
            print(f"   ⚠ No helmets: {stats['no_helmets_count']}")
            print(f"   ⚠ No vests: {stats['no_vests_count']}")
            
            print(f"\n   🎯 COMPLIANCE STATUS:")
            if stats['persons_count'] == 0:
                print("      ⚪ No persons in frame")
            elif stats['is_compliant']:
                print("      ✅ SAFE: All persons comply with safety rules!")
            else:
                print("      🚨 VIOLATION: Safety rules not followed!")
            
            if stats['violations']:
                print(f"\n   🚨 VIOLATIONS FOUND:")
                for violation in stats['violations']:
                    print(f"      • {violation}")
            
            # Детекции
            print(f"\n   🔍 DETECTED OBJECTS ({len(result['detections'])} total):")
            for det in result['detections'][:10]:  # Показываем первые 10
                print(f"      • {det['class_name']}: {det['confidence']:.1%}")
            
            if len(result['detections']) > 10:
                print(f"      ... and {len(result['detections']) - 10} more")
            
            
            if 'annotated_image' in result:
                try:
                    img_data = result['annotated_image'].split(',')[1]
                    output_file = "safety_analysis_result.jpg"
                    
                    with open(output_file, 'wb') as f:
                        f.write(base64.b64decode(img_data))
                    
                    print(f"\n   💾 Annotated image saved as: {output_file}")
                    print(f"   👀 Open this file to see detection boxes")
                except:
                    print("\n   ⚠ Could not save annotated image")
            
            print(f"\n{'=' * 60}")
            print("🎉 TEST COMPLETE! Your safety monitoring API is working!")
            
            return True
            
        else:
            print(f"❌ API Error {response.status_code}: {response.text[:200]}")
            
    except FileNotFoundError:
        print(f"❌ Image file not found: {image_path}")
        print("   Please provide a real image of a construction site or workers")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    return False

if __name__ == "__main__":
    
    import os
    
    test_images = ["real_test.jpg", "test_image.jpg", "construction.jpg"]
    
    for img in test_images:
        if os.path.exists(img):
            print(f"Found test image: {img}")
            real_world_test(img)
            break
    else:
        print("❌ No test image found!")
        print("Please add an image file named:")
        print("  - real_test.jpg (recommended)")
        print("  - test_image.jpg")
        print("  - construction.jpg")
        print("\nOr specify your own: python real_test.py your_image.jpg")