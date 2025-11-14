"""
Configuration settings for Crop Analysis AI System
"""
import os
from datetime import datetime

# API Configuration
API_KEYS = {
    'OPENWEATHER_API_KEY': os.getenv('OPENWEATHER_API_KEY', 'your_openweather_key'),
    'NASA_API_KEY': os.getenv('NASA_API_KEY', 'DEMO_KEY'),
    'SENTINEL_API_KEY': os.getenv('SENTINEL_API_KEY', 'your_sentinel_key'),
    'USGS_API_KEY': os.getenv('USGS_API_KEY', 'your_usgs_key')
}

# Geographic Configuration - Ludhiana Region
LOCATION = {
    'LATITUDE': 30.9010,
    'LONGITUDE': 75.8573,
    'REGION_NAME': 'Ludhiana District',
    'STATE': 'Punjab',
    'COUNTRY': 'India',
    'TIMEZONE': 'Asia/Kolkata',
    'ELEVATION': 247  # meters above sea level
}

# Model Configuration
MODEL_CONFIG = {
    'INPUT_SIZE': (256, 256, 3),
    'BATCH_SIZE': 16,
    'EPOCHS': 30,
    'LEARNING_RATE': 0.001,
    'VALIDATION_SPLIT': 0.2,
    'MODEL_VERSION': '2.1.0'
}

# Crop Configuration
CROP_SEASONS = {
    'RABI': {
        'months': [11, 12, 1, 2, 3],
        'crops': ['wheat', 'barley', 'mustard', 'gram'],
        'optimal_temp': (15, 25),
        'sowing_period': 'October-December',
        'harvest_period': 'March-May'
    },
    'KHARIF': {
        'months': [6, 7, 8, 9, 10],
        'crops': ['rice', 'cotton', 'sugarcane', 'maize'],
        'optimal_temp': (20, 35),
        'sowing_period': 'June-July',
        'harvest_period': 'September-October'
    },
    'ZAID': {
        'months': [4, 5, 6],
        'crops': ['fodder', 'watermelon', 'cucumber'],
        'optimal_temp': (25, 40),
        'sowing_period': 'March-April',
        'harvest_period': 'June-July'
    }
}

# Yield Benchmarks (tons per hectare)
YIELD_BENCHMARKS = {
    'wheat': {'min': 2.5, 'avg': 4.5, 'max': 7.0},
    'rice': {'min': 3.0, 'avg': 6.2, 'max': 9.5},
    'corn': {'min': 2.8, 'avg': 5.8, 'max': 8.2},
    'cotton': {'min': 1.2, 'avg': 2.1, 'max': 3.5},
    'sugarcane': {'min': 45, 'avg': 75, 'max': 110}
}

# Analysis Thresholds
THRESHOLDS = {
    'NDVI': {'poor': 0.2, 'moderate': 0.5, 'good': 0.7},
    'SOIL_MOISTURE': {'dry': 30, 'optimal': 60, 'wet': 80},
    'TEMPERATURE_STRESS': {'cold': 10, 'optimal_min': 15, 'optimal_max': 30, 'hot': 40}
}

# File Paths
PATHS = {
    'MODEL_DIR': 'models/',
    'DATA_DIR': 'data/',
    'LOGS_DIR': 'logs/',
    'TEMP_DIR': 'temp/',
    'RESULTS_DIR': 'results/'
}

# Logging Configuration
LOGGING = {
    'LEVEL': 'INFO',
    'FORMAT': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'FILE': f'logs/crop_analysis_{datetime.now().strftime("%Y%m%d")}.log'
}

# API Rate Limits
RATE_LIMITS = {
    'OPENWEATHER': 1000,  # calls per day
    'NASA': 2000,
    'SENTINEL': 500
}