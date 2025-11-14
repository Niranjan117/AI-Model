import requests
import time

def test_api():
    url = "http://localhost:8000/analyze"
    
    # Test with any image file you have
    image_path = input("Enter path to test image: ")
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            
            print("Sending image for analysis...")
            start_time = time.time()
            
            response = requests.post(url, files=files)
            
            end_time = time.time()
            
            if response.status_code == 200:
                data = response.json()
                print(f"\n✅ Analysis completed in {end_time - start_time:.2f} seconds")
                print(f"Server processing time: {data.get('processing_time_seconds', 'N/A')} seconds")
                print(f"Yield prediction: {data['yield_prediction']:.2f} tons/hectare")
                print("\nLand use percentages:")
                for land_type, percentage in data['land_use_percentages'].items():
                    print(f"  {land_type}: {percentage:.1f}%")
                print(f"\nNDVI: {data['vegetation_health']['ndvi']:.3f}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(response.text)
                
    except FileNotFoundError:
        print("❌ Image file not found")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_api()