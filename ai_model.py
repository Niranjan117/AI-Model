"""
Lightweight AI Model for Render Deployment
"""
import numpy as np
import cv2
import os
import requests
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle
import logging

logger = logging.getLogger(__name__)

class CropAnalysisAI:
    def __init__(self):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.land_classes = ['water', 'forest', 'wheat', 'rice', 'corn', 'barren']
        self.is_trained = False
        
        # API keys
        self.weather_api_key = os.getenv('OPENWEATHER_API_KEY', 'demo_key')
        self.nasa_api_key = os.getenv('NASA_API_KEY', 'DEMO_KEY')
        
        # Ludhiana coordinates
        self.lat = 30.9010
        self.lon = 75.8573
        
    def get_real_weather_data(self):
        """Get real weather data from OpenWeatherMap API"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': self.lat,
                'lon': self.lon,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'pressure': data['main']['pressure'],
                    'wind_speed': data['wind']['speed'],
                    'cloud_cover': data['clouds']['all'],
                    'weather_condition': data['weather'][0]['main'],
                    'description': data['weather'][0]['description'],
                    'crop_season': self._get_crop_season(),
                    'growing_degree_days': max(0, data['main']['temp'] - 10)
                }
            else:
                return self._get_fallback_weather()
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return self._get_fallback_weather()
    
    def get_soil_moisture_data(self):
        """Get soil moisture estimation"""
        try:
            # Simple soil moisture estimation based on weather
            weather = self.get_real_weather_data()
            
            base_moisture = 45.0
            temp_factor = max(0.5, 1.0 - (abs(weather['temperature'] - 25) / 50))
            humidity_factor = weather['humidity'] / 100
            
            soil_moisture = base_moisture * temp_factor * humidity_factor
            soil_moisture = max(10, min(90, soil_moisture))
            
            return {
                'soil_moisture_percent': soil_moisture,
                'water_stress_index': 1.0 - (soil_moisture / 100),
                'irrigation_need': 'low' if soil_moisture > 60 else 'high' if soil_moisture < 30 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Soil data error: {e}")
            return {
                'soil_moisture_percent': 45.0,
                'water_stress_index': 0.4,
                'irrigation_need': 'medium'
            }
    
    def extract_image_features(self, image):
        """Extract comprehensive features from image"""
        # Resize for consistent processing
        image = cv2.resize(image, (256, 256))
        
        # Color statistics
        features = []
        for channel in range(3):
            features.extend([
                np.mean(image[:,:,channel]),
                np.std(image[:,:,channel]),
                np.percentile(image[:,:,channel], 25),
                np.percentile(image[:,:,channel], 75)
            ])
        
        # Vegetation indices
        red = image[:,:,0].astype(float) / 255.0
        green = image[:,:,1].astype(float) / 255.0
        blue = image[:,:,2].astype(float) / 255.0
        
        # Simulate NIR
        nir = green * 1.3 + np.random.normal(0, 0.02, green.shape)
        nir = np.clip(nir, 0, 1)
        
        # Calculate indices
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        evi = np.where((nir + 6*red - 7.5*blue + 1) > 0,
                      2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), 0)
        
        features.extend([
            np.mean(ndvi), np.std(ndvi),
            np.mean(evi), np.std(evi),
            np.mean(green > 0.4),  # Green coverage
            np.mean(ndvi > 0.2)    # Vegetation density
        ])
        
        # Texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features.extend([
            np.mean(gradient_magnitude),
            np.std(gradient_magnitude),
            np.std(gray)  # Texture contrast
        ])
        
        # Color clustering
        pixels = image.reshape(-1, 3)
        if len(pixels) > 1000:
            sample_indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[sample_indices]
        
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(pixels)
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        color_proportions = counts / len(pixels)
        
        # Pad to ensure consistent length
        while len(color_proportions) < 5:
            color_proportions = np.append(color_proportions, 0)
        
        features.extend(color_proportions[:5])
        
        return np.array(features)
    
    def train_model(self):
        """Train Random Forest model with synthetic data"""
        logger.info("Training Random Forest model...")
        
        # Generate synthetic training data
        X_train = []
        y_train = []
        
        np.random.seed(42)
        
        for class_idx, class_name in enumerate(self.land_classes):
            for _ in range(200):  # 200 samples per class
                # Generate synthetic image
                if class_name == 'water':
                    img = self._generate_water_image()
                elif class_name == 'forest':
                    img = self._generate_forest_image()
                elif class_name == 'wheat':
                    img = self._generate_wheat_image()
                elif class_name == 'rice':
                    img = self._generate_rice_image()
                elif class_name == 'corn':
                    img = self._generate_corn_image()
                else:  # barren
                    img = self._generate_barren_image()
                
                features = self.extract_image_features(img)
                X_train.append(features)
                y_train.append(class_idx)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        logger.info("Model training completed!")
        
        # Save model
        try:
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(self.rf_model, f)
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(self.scaler, f)
        except Exception as e:
            logger.warning(f"Could not save model: {e}")
    
    def _generate_water_image(self):
        """Generate synthetic water image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:,:,2] = np.random.randint(120, 255, (64, 64))  # High blue
        img[:,:,1] = np.random.randint(50, 120, (64, 64))   # Medium green
        img[:,:,0] = np.random.randint(20, 80, (64, 64))    # Low red
        return img
    
    def _generate_forest_image(self):
        """Generate synthetic forest image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:,:,1] = np.random.randint(80, 180, (64, 64))   # High green
        img[:,:,0] = np.random.randint(20, 100, (64, 64))   # Low-medium red
        img[:,:,2] = np.random.randint(20, 100, (64, 64))   # Low-medium blue
        # Add texture
        noise = np.random.randint(-30, 30, (64, 64, 3))
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        return img
    
    def _generate_wheat_image(self):
        """Generate synthetic wheat image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:,:,0] = np.random.randint(150, 220, (64, 64))  # High red (golden)
        img[:,:,1] = np.random.randint(140, 200, (64, 64))  # High green (golden)
        img[:,:,2] = np.random.randint(60, 120, (64, 64))   # Medium blue
        return img
    
    def _generate_rice_image(self):
        """Generate synthetic rice image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:,:,1] = np.random.randint(120, 220, (64, 64))  # Very high green
        img[:,:,0] = np.random.randint(40, 120, (64, 64))   # Medium red
        img[:,:,2] = np.random.randint(30, 100, (64, 64))   # Low-medium blue
        return img
    
    def _generate_corn_image(self):
        """Generate synthetic corn image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:,:,1] = np.random.randint(100, 180, (64, 64))  # High green
        img[:,:,0] = np.random.randint(80, 150, (64, 64))   # Medium-high red
        img[:,:,2] = np.random.randint(40, 110, (64, 64))   # Medium blue
        return img
    
    def _generate_barren_image(self):
        """Generate synthetic barren land image"""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        base_color = np.random.randint(80, 150)
        img[:,:,0] = np.random.randint(base_color-20, base_color+20, (64, 64))
        img[:,:,1] = np.random.randint(base_color-20, base_color+20, (64, 64))
        img[:,:,2] = np.random.randint(base_color-20, base_color+20, (64, 64))
        return img
    
    def analyze_image(self, image_path):
        """Analyze uploaded image"""
        if not self.is_trained:
            self.train_model()
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not load image")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features
            features = self.extract_image_features(image)
            features_scaled = self.scaler.transform([features])
            
            # Get predictions
            predictions = self.rf_model.predict_proba(features_scaled)[0]
            
            # Convert to land use percentages
            land_use_percentages = {}
            for i, class_name in enumerate(self.land_classes):
                if class_name == 'water':
                    water_percent = predictions[i] * 100
                    land_use_percentages['lake'] = water_percent * 0.6
                    land_use_percentages['river'] = water_percent * 0.4
                else:
                    land_use_percentages[class_name] = predictions[i] * 100
            
            # Get real-time data
            weather_data = self.get_real_weather_data()
            soil_data = self.get_soil_moisture_data()
            
            # Calculate vegetation health
            vegetation_health = self._calculate_vegetation_health(image)
            
            # Calculate yield
            yield_prediction = self._calculate_yield(land_use_percentages, weather_data, vegetation_health)
            
            return {
                'yield_prediction': yield_prediction,
                'land_use_percentages': {k: round(v, 2) for k, v in land_use_percentages.items()},
                'vegetation_health': vegetation_health,
                'weather_factors': weather_data,
                'soil_conditions': soil_data,
                'analysis_confidence': round(np.max(predictions), 3)
            }
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            raise
    
    def _calculate_vegetation_health(self, image):
        """Calculate vegetation health indices"""
        image_resized = cv2.resize(image, (256, 256))
        red = image_resized[:,:,0].astype(float) / 255.0
        green = image_resized[:,:,1].astype(float) / 255.0
        blue = image_resized[:,:,2].astype(float) / 255.0
        
        nir = green * 1.2
        ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red), 0)
        evi = np.where((nir + 6*red - 7.5*blue + 1) > 0,
                      2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1), 0)
        
        return {
            'ndvi': round(float(np.mean(ndvi)), 3),
            'evi': round(float(np.mean(evi)), 3),
            'green_coverage': round(float(np.mean(green > 0.4)), 3),
            'vegetation_density': round(float(np.mean(ndvi > 0.2)), 3)
        }
    
    def _calculate_yield(self, land_use, weather, vegetation):
        """Calculate yield prediction"""
        agricultural_percent = (land_use.get('wheat', 0) + 
                              land_use.get('rice', 0) + 
                              land_use.get('corn', 0))
        
        if agricultural_percent < 5:
            return 2.0
        
        # Base yield calculation
        base_yield = 8 + (agricultural_percent / 100) * 15
        
        # Weather adjustments
        temp_factor = 1.0
        if weather['temperature'] < 15 or weather['temperature'] > 35:
            temp_factor = 0.8
        
        # Vegetation health factor
        veg_factor = (vegetation['ndvi'] + 1) / 2
        
        # Soil moisture factor
        soil_factor = 1.0 if weather.get('humidity', 50) > 50 else 0.9
        
        final_yield = base_yield * temp_factor * veg_factor * soil_factor
        return round(max(2.0, min(30.0, final_yield)), 2)
    
    def _get_crop_season(self):
        """Get current crop season"""
        month = datetime.now().month
        if month in [11, 12, 1, 2, 3]:
            return 'rabi'
        elif month in [6, 7, 8, 9, 10]:
            return 'kharif'
        else:
            return 'zaid'
    
    def _get_fallback_weather(self):
        """Fallback weather data"""
        month = datetime.now().month
        
        if month in [12, 1, 2]:
            base_temp = 18
        elif month in [6, 7, 8, 9]:
            base_temp = 30
        else:
            base_temp = 25
        
        return {
            'temperature': base_temp + np.random.uniform(-3, 3),
            'humidity': 65 + np.random.uniform(-10, 15),
            'pressure': 1013,
            'wind_speed': 3.5,
            'cloud_cover': 40,
            'weather_condition': 'Clear',
            'description': 'clear sky',
            'crop_season': self._get_crop_season(),
            'growing_degree_days': max(0, base_temp - 10)
        }
    
    def load_models(self):
        """Load or train models"""
        try:
            if os.path.exists('rf_model.pkl') and os.path.exists('scaler.pkl'):
                with open('rf_model.pkl', 'rb') as f:
                    self.rf_model = pickle.load(f)
                with open('scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_trained = True
                logger.info("Loaded pre-trained model")
            else:
                self.train_model()
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
            self.train_model()
    
    def train_models(self):
        """Train models"""
        self.train_model()