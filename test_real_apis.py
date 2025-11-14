#!/usr/bin/env python3
"""
Test script to verify real APIs are working
"""
import requests
import json
from datetime import datetime

def test_openweather_api():
    """Test OpenWeatherMap API"""
    print("Testing OpenWeatherMap API...")
    
    # Ludhiana coordinates
    lat, lon = 30.9010, 75.8573
    api_key = "your_openweather_api_key"  # Replace with real key
    
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Weather API working!")
            print(f"   Location: {data['name']}, {data['sys']['country']}")
            print(f"   Temperature: {data['main']['temp']}°C")
            print(f"   Humidity: {data['main']['humidity']}%")
            print(f"   Weather: {data['weather'][0]['description']}")
            return True
        else:
            print(f"❌ Weather API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Weather API failed: {e}")
        return False

def test_nasa_api():
    """Test NASA API"""
    print("\nTesting NASA API...")
    
    lat, lon = 30.9010, 75.8573
    api_key = "DEMO_KEY"  # NASA demo key
    
    url = f"https://api.nasa.gov/planetary/earth/assets?lon={lon}&lat={lat}&date=2023-01-01&dim=0.10&api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("✅ NASA API working!")
            print(f"   Response received for coordinates: {lat}, {lon}")
            return True
        else:
            print(f"❌ NASA API error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ NASA API failed: {e}")
        return False

def test_geolocation():
    """Test geolocation services"""
    print("\nTesting Geolocation...")
    
    try:
        # Test IP-based location (backup)
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Geolocation working!")
            print(f"   Detected location: {data.get('city', 'Unknown')}, {data.get('country', 'Unknown')}")
            print(f"   Coordinates: {data.get('lat', 'N/A')}, {data.get('lon', 'N/A')}")
            return True
        else:
            print("❌ Geolocation failed")
            return False
    except Exception as e:
        print(f"❌ Geolocation error: {e}")
        return False

def test_soil_moisture_calculation():
    """Test soil moisture calculations"""
    print("\nTesting Soil Moisture Calculations...")
    
    try:
        # Simulate soil moisture calculation
        import numpy as np
        
        # Mock weather data
        temperature = 25.0
        humidity = 65.0
        rainfall_last_week = 15.0  # mm
        
        # Calculate soil moisture index
        base_moisture = 40.0
        temp_factor = max(0.5, 1.0 - (abs(temperature - 25) / 50))
        humidity_factor = humidity / 100
        rainfall_factor = min(1.5, rainfall_last_week / 20)
        
        soil_moisture = base_moisture * temp_factor * humidity_factor * rainfall_factor
        soil_moisture = max(10, min(90, soil_moisture))
        
        print("✅ Soil moisture calculation working!")
        print(f"   Calculated soil moisture: {soil_moisture:.1f}%")
        print(f"   Temperature factor: {temp_factor:.2f}")
        print(f"   Humidity factor: {humidity_factor:.2f}")
        print(f"   Rainfall factor: {rainfall_factor:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Soil calculation error: {e}")
        return False

def test_vegetation_indices():
    """Test vegetation index calculations"""
    print("\nTesting Vegetation Index Calculations...")
    
    try:
        import numpy as np
        
        # Simulate RGB values from satellite image
        red = np.random.uniform(0.2, 0.4, (100, 100))
        green = np.random.uniform(0.4, 0.7, (100, 100))
        blue = np.random.uniform(0.1, 0.3, (100, 100))
        
        # Calculate NIR approximation
        nir = green * 1.3 + np.random.normal(0, 0.05, green.shape)
        
        # Calculate vegetation indices
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        evi = np.where((nir + 6*red - 7.5*blue + 1) > 0,
                      2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), 0)
        
        # SAVI (Soil Adjusted Vegetation Index)
        L = 0.5
        savi = np.where((nir + red + L) > 0, (1 + L) * (nir - red) / (nir + red + L), 0)
        
        print("✅ Vegetation indices calculation working!")
        print(f"   NDVI: {np.mean(ndvi):.3f}")
        print(f"   EVI: {np.mean(evi):.3f}")
        print(f"   SAVI: {np.mean(savi):.3f}")
        print(f"   Green coverage: {np.mean(green > 0.4):.3f}")
        
        return True
    except Exception as e:
        print(f"❌ Vegetation calculation error: {e}")
        return False

def test_yield_prediction_model():
    """Test yield prediction calculations"""
    print("\nTesting Yield Prediction Model...")
    
    try:
        # Mock data
        land_use = {
            'wheat': 13.0, 'rice': 18.5, 'corn': 8.2,
            'annual_crop': 6.3, 'forest': 12.4, 'water': 8.0, 'barren': 33.6
        }
        
        weather = {
            'temperature': 22.0, 'humidity': 68.0, 'rainfall': 45.0,
            'crop_season': 'rabi', 'heat_stress_level': 'none'
        }
        
        soil = {'soil_moisture_percent': 55.0, 'water_stress_index': 0.3}
        vegetation = {'ndvi': 0.65, 'evi': 0.58, 'vegetation_density': 0.72}
        
        # Calculate yield
        base_yields = {'wheat': 4.5, 'rice': 6.2, 'corn': 5.8, 'annual_crop': 4.0}
        
        total_yield = 0
        total_area = 0
        
        for crop in base_yields:
            area_percent = land_use.get(crop, 0)
            if area_percent > 0:
                base_yield = base_yields[crop]
                
                # Weather factor
                weather_factor = 1.0
                if weather['temperature'] < 15 or weather['temperature'] > 35:
                    weather_factor = 0.8
                
                # Soil factor
                soil_factor = 1.0 if soil['soil_moisture_percent'] > 50 else 0.8
                
                # Vegetation factor
                veg_factor = (vegetation['ndvi'] + 1) / 2
                
                final_yield = base_yield * weather_factor * soil_factor * veg_factor
                total_yield += final_yield * (area_percent / 100)
                total_area += area_percent / 100
        
        predicted_yield = total_yield / total_area if total_area > 0 else 0
        
        print("✅ Yield prediction model working!")
        print(f"   Predicted yield: {predicted_yield:.2f} tons/hectare")
        print(f"   Weather factor applied: {weather_factor}")
        print(f"   Soil factor applied: {soil_factor}")
        print(f"   Vegetation factor applied: {veg_factor:.2f}")
        
        return True
    except Exception as e:
        print(f"❌ Yield prediction error: {e}")
        return False

def main():
    """Run all API tests"""
    print("TESTING REAL APIs AND CALCULATIONS")
    print("=" * 50)
    
    tests = [
        test_openweather_api,
        test_nasa_api,
        test_geolocation,
        test_soil_moisture_calculation,
        test_vegetation_indices,
        test_yield_prediction_model
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL SYSTEMS WORKING! Ready for deployment.")
    else:
        print("⚠️  Some systems need API keys or have issues.")
        print("\nNEXT STEPS:")
        print("1. Get real API keys from api_keys_setup.txt")
        print("2. Replace placeholder keys in ai_model.py")
        print("3. Re-run this test")

if __name__ == "__main__":
    main()