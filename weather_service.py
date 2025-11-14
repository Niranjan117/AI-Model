"""
Weather Service Module for Agricultural Analysis
"""
import requests
import json
import numpy as np
from datetime import datetime, timedelta
import logging
from config import API_KEYS, LOCATION

class WeatherService:
    def __init__(self):
        self.api_key = API_KEYS['OPENWEATHER_API_KEY']
        self.lat = LOCATION['LATITUDE']
        self.lon = LOCATION['LONGITUDE']
        self.logger = logging.getLogger(__name__)
        
    def get_current_weather(self):
        """Get current weather conditions"""
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_current_weather(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Weather API request failed: {e}")
            return self._get_fallback_weather()
    
    def get_weather_forecast(self, days=5):
        """Get weather forecast"""
        url = f"http://api.openweathermap.org/data/2.5/forecast"
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric',
            'cnt': days * 8  # 8 forecasts per day (3-hour intervals)
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return self._parse_forecast_data(data)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Forecast API request failed: {e}")
            return self._get_fallback_forecast()
    
    def get_historical_weather(self, days_back=7):
        """Get historical weather data"""
        historical_data = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i+1)
            timestamp = int(date.timestamp())
            
            url = f"http://api.openweathermap.org/data/2.5/onecall/timemachine"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'dt': timestamp,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            try:
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    historical_data.append(self._parse_historical_data(data, date))
                    
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Historical weather request failed for {date}: {e}")
                continue
        
        return historical_data
    
    def _parse_current_weather(self, data):
        """Parse current weather API response"""
        return {
            'temperature': data['main']['temp'],
            'feels_like': data['main']['feels_like'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'wind_speed': data['wind']['speed'],
            'wind_direction': data['wind'].get('deg', 0),
            'cloud_cover': data['clouds']['all'],
            'visibility': data.get('visibility', 10000) / 1000,
            'weather_main': data['weather'][0]['main'],
            'weather_description': data['weather'][0]['description'],
            'sunrise': datetime.fromtimestamp(data['sys']['sunrise']),
            'sunset': datetime.fromtimestamp(data['sys']['sunset']),
            'uv_index': self._estimate_uv_index(data['weather'][0]['id']),
            'timestamp': datetime.now()
        }
    
    def _parse_forecast_data(self, data):
        """Parse forecast API response"""
        forecasts = []
        
        for item in data['list']:
            forecast = {
                'datetime': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'pressure': item['main']['pressure'],
                'wind_speed': item['wind']['speed'],
                'cloud_cover': item['clouds']['all'],
                'weather_main': item['weather'][0]['main'],
                'precipitation': item.get('rain', {}).get('3h', 0) + item.get('snow', {}).get('3h', 0)
            }
            forecasts.append(forecast)
        
        return forecasts
    
    def _parse_historical_data(self, data, date):
        """Parse historical weather data"""
        current = data['current']
        
        return {
            'date': date,
            'temperature': current['temp'],
            'humidity': current['humidity'],
            'pressure': current['pressure'],
            'wind_speed': current['wind_speed'],
            'cloud_cover': current['clouds'],
            'weather_main': current['weather'][0]['main'],
            'precipitation': sum([h.get('rain', {}).get('1h', 0) for h in data.get('hourly', [])])
        }
    
    def calculate_growing_degree_days(self, temperature_data, base_temp=10):
        """Calculate Growing Degree Days (GDD)"""
        gdd_values = []
        
        for temp_record in temperature_data:
            if isinstance(temp_record, dict):
                temp = temp_record.get('temperature', 0)
            else:
                temp = temp_record
            
            gdd = max(0, temp - base_temp)
            gdd_values.append(gdd)
        
        return {
            'daily_gdd': gdd_values,
            'cumulative_gdd': sum(gdd_values),
            'average_gdd': np.mean(gdd_values) if gdd_values else 0
        }
    
    def calculate_evapotranspiration(self, weather_data):
        """Calculate reference evapotranspiration using Penman-Monteith equation"""
        temp = weather_data['temperature']
        humidity = weather_data['humidity']
        wind_speed = weather_data['wind_speed']
        pressure = weather_data['pressure']
        
        # Simplified Penman-Monteith calculation
        # Saturation vapor pressure
        es = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
        
        # Actual vapor pressure
        ea = es * humidity / 100
        
        # Slope of saturation vapor pressure curve
        delta = 4098 * es / ((temp + 237.3) ** 2)
        
        # Psychrometric constant
        gamma = 0.665 * pressure / 1013.25
        
        # Wind speed at 2m height
        u2 = wind_speed * 4.87 / np.log(67.8 * 10 - 5.42)
        
        # Reference evapotranspiration (mm/day)
        et0 = (0.408 * delta * temp + gamma * 900 / (temp + 273) * u2 * (es - ea)) / (delta + gamma * (1 + 0.34 * u2))
        
        return max(0, et0)
    
    def assess_crop_stress(self, weather_data, crop_type='wheat'):
        """Assess weather-related crop stress"""
        temp = weather_data['temperature']
        humidity = weather_data['humidity']
        wind_speed = weather_data['wind_speed']
        
        stress_factors = {}
        
        # Temperature stress
        if crop_type == 'wheat':
            optimal_range = (15, 25)
        elif crop_type == 'rice':
            optimal_range = (20, 35)
        else:
            optimal_range = (18, 30)
        
        if temp < optimal_range[0]:
            stress_factors['cold_stress'] = (optimal_range[0] - temp) / 10
        elif temp > optimal_range[1]:
            stress_factors['heat_stress'] = (temp - optimal_range[1]) / 15
        else:
            stress_factors['temperature_stress'] = 0
        
        # Humidity stress
        if humidity < 40:
            stress_factors['drought_stress'] = (40 - humidity) / 40
        elif humidity > 85:
            stress_factors['excess_moisture'] = (humidity - 85) / 15
        else:
            stress_factors['humidity_stress'] = 0
        
        # Wind stress
        if wind_speed > 10:
            stress_factors['wind_stress'] = (wind_speed - 10) / 20
        else:
            stress_factors['wind_stress'] = 0
        
        # Overall stress index (0-1, where 1 is maximum stress)
        overall_stress = np.mean(list(stress_factors.values()))
        stress_factors['overall_stress'] = min(1.0, overall_stress)
        
        return stress_factors
    
    def get_seasonal_analysis(self):
        """Get seasonal weather analysis"""
        current_month = datetime.now().month
        
        # Define seasons for Punjab region
        if current_month in [12, 1, 2]:
            season = 'winter'
            crop_season = 'rabi'
        elif current_month in [3, 4, 5]:
            season = 'spring'
            crop_season = 'rabi_harvest'
        elif current_month in [6, 7, 8, 9]:
            season = 'monsoon'
            crop_season = 'kharif'
        else:
            season = 'post_monsoon'
            crop_season = 'kharif_harvest'
        
        return {
            'season': season,
            'crop_season': crop_season,
            'month': current_month,
            'recommended_crops': self._get_seasonal_crops(crop_season)
        }
    
    def _get_seasonal_crops(self, crop_season):
        """Get recommended crops for current season"""
        seasonal_crops = {
            'rabi': ['wheat', 'barley', 'mustard', 'gram', 'pea'],
            'rabi_harvest': ['wheat', 'barley', 'mustard'],
            'kharif': ['rice', 'cotton', 'sugarcane', 'maize', 'bajra'],
            'kharif_harvest': ['rice', 'cotton', 'maize']
        }
        
        return seasonal_crops.get(crop_season, [])
    
    def _estimate_uv_index(self, weather_id):
        """Estimate UV index based on weather conditions"""
        # Clear sky
        if weather_id in [800]:
            return np.random.uniform(6, 10)
        # Partly cloudy
        elif weather_id in [801, 802]:
            return np.random.uniform(4, 7)
        # Cloudy/overcast
        elif weather_id in [803, 804]:
            return np.random.uniform(2, 5)
        # Rain/storm
        else:
            return np.random.uniform(1, 3)
    
    def _get_fallback_weather(self):
        """Fallback weather data when API fails"""
        month = datetime.now().month
        
        # Seasonal averages for Ludhiana
        if month in [12, 1, 2]:  # Winter
            base_temp = 12
        elif month in [3, 4, 5]:  # Spring
            base_temp = 25
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 30
        else:  # Post-monsoon
            base_temp = 22
        
        return {
            'temperature': base_temp + np.random.uniform(-3, 3),
            'feels_like': base_temp + np.random.uniform(-2, 4),
            'humidity': 65 + np.random.uniform(-15, 15),
            'pressure': 1013 + np.random.uniform(-10, 10),
            'wind_speed': 3.5 + np.random.uniform(-1, 2),
            'wind_direction': np.random.uniform(0, 360),
            'cloud_cover': 40 + np.random.uniform(-20, 30),
            'visibility': 8.0,
            'weather_main': 'Clear',
            'weather_description': 'clear sky',
            'uv_index': 5.0,
            'timestamp': datetime.now()
        }
    
    def _get_fallback_forecast(self):
        """Fallback forecast data"""
        forecasts = []
        base_weather = self._get_fallback_weather()
        
        for i in range(40):  # 5 days * 8 forecasts per day
            forecast_time = datetime.now() + timedelta(hours=i*3)
            temp_variation = np.random.uniform(-2, 2)
            
            forecast = {
                'datetime': forecast_time,
                'temperature': base_weather['temperature'] + temp_variation,
                'humidity': base_weather['humidity'] + np.random.uniform(-5, 5),
                'pressure': base_weather['pressure'] + np.random.uniform(-2, 2),
                'wind_speed': base_weather['wind_speed'] + np.random.uniform(-1, 1),
                'cloud_cover': base_weather['cloud_cover'] + np.random.uniform(-10, 10),
                'weather_main': base_weather['weather_main'],
                'precipitation': np.random.uniform(0, 2) if np.random.random() > 0.7 else 0
            }
            forecasts.append(forecast)
        
        return forecasts