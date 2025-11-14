"""
Utility Functions for Crop Analysis AI System
"""
import os
import hashlib
import uuid
from datetime import datetime, timedelta
import numpy as np
import cv2
from PIL import Image
import json
import requests
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def generate_session_id():
    """Generate unique session ID"""
    return str(uuid.uuid4())

def calculate_file_hash(file_path):
    """Calculate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError as e:
        logger.error(f"Error calculating file hash: {e}")
        return None

def validate_image_file(file_path, max_size_mb=10):
    """Validate uploaded image file"""
    try:
        # Check file exists
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
        if file_size > max_size_mb:
            return False, f"File too large: {file_size:.1f}MB (max: {max_size_mb}MB)"
        
        # Check if it's a valid image
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception:
            return False, "Invalid image file"
        
        # Check image dimensions
        img = cv2.imread(file_path)
        if img is None:
            return False, "Cannot read image file"
        
        height, width = img.shape[:2]
        if width < 100 or height < 100:
            return False, f"Image too small: {width}x{height} (min: 100x100)"
        
        if width > 5000 or height > 5000:
            return False, f"Image too large: {width}x{height} (max: 5000x5000)"
        
        return True, "Valid image file"
        
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return False, f"Validation error: {str(e)}"

def resize_image_smart(image, target_size=(256, 256)):
    """Smart image resizing with aspect ratio preservation"""
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    # Calculate scaling factor
    scale_w = target_width / width
    scale_h = target_height / height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Create canvas and center the image
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate position to center the image
    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    
    canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return canvas

def calculate_image_statistics(image):
    """Calculate comprehensive image statistics"""
    stats = {}
    
    # Basic statistics
    stats['mean_rgb'] = np.mean(image, axis=(0, 1)).tolist()
    stats['std_rgb'] = np.std(image, axis=(0, 1)).tolist()
    stats['min_rgb'] = np.min(image, axis=(0, 1)).tolist()
    stats['max_rgb'] = np.max(image, axis=(0, 1)).tolist()
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    stats['mean_hsv'] = np.mean(hsv, axis=(0, 1)).tolist()
    stats['mean_lab'] = np.mean(lab, axis=(0, 1)).tolist()
    
    # Brightness and contrast
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    stats['brightness'] = float(np.mean(gray))
    stats['contrast'] = float(np.std(gray))
    
    # Color distribution
    stats['dominant_color'] = get_dominant_color(image)
    
    return stats

def get_dominant_color(image, k=5):
    """Get dominant colors in image using K-means"""
    try:
        # Reshape image to be a list of pixels
        pixels = image.reshape(-1, 3)
        
        # Apply K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get cluster centers (dominant colors)
        colors = kmeans.cluster_centers_.astype(int)
        
        # Get cluster sizes
        labels = kmeans.labels_
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant_colors = []
        for i in sorted_indices:
            color = colors[i].tolist()
            percentage = (counts[i] / len(pixels)) * 100
            dominant_colors.append({
                'color_rgb': color,
                'percentage': round(percentage, 2)
            })
        
        return dominant_colors
        
    except Exception as e:
        logger.error(f"Error calculating dominant colors: {e}")
        return []

def format_coordinates(lat, lon):
    """Format coordinates for display"""
    lat_dir = 'N' if lat >= 0 else 'S'
    lon_dir = 'E' if lon >= 0 else 'W'
    
    return f"{abs(lat):.4f}°{lat_dir}, {abs(lon):.4f}°{lon_dir}"

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    from math import radians, cos, sin, asin, sqrt
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r

def get_season_from_date(date=None):
    """Get agricultural season from date"""
    if date is None:
        date = datetime.now()
    
    month = date.month
    
    if month in [11, 12, 1, 2, 3]:
        return 'rabi'  # Winter crops
    elif month in [6, 7, 8, 9, 10]:
        return 'kharif'  # Monsoon crops
    else:
        return 'zaid'  # Summer crops

def format_file_size(size_bytes):
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def create_thumbnail(image_path, output_path, size=(150, 150)):
    """Create thumbnail of image"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(output_path, "JPEG", quality=85)
        return True
    except Exception as e:
        logger.error(f"Error creating thumbnail: {e}")
        return False

def validate_api_key(api_key, service='openweather'):
    """Validate API key format"""
    if not api_key or api_key in ['your_api_key', 'DEMO_KEY']:
        return False, "Invalid or placeholder API key"
    
    if service == 'openweather':
        if len(api_key) != 32:
            return False, "OpenWeather API key should be 32 characters"
    elif service == 'nasa':
        if len(api_key) < 10:
            return False, "NASA API key too short"
    
    return True, "Valid API key format"

def rate_limit_check(client_ip, endpoint, max_requests=100, time_window=3600):
    """Simple rate limiting check"""
    # This is a simplified version - in production, use Redis or similar
    current_time = datetime.now()
    
    # For demo purposes, always return True
    # In production, implement proper rate limiting with Redis/database
    return True, 0

def sanitize_filename(filename):
    """Sanitize filename for safe storage"""
    import re
    
    # Remove path components
    filename = os.path.basename(filename)
    
    # Replace unsafe characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    name, ext = os.path.splitext(filename)
    if len(name) > 50:
        name = name[:50]
    
    return name + ext

def create_response_metadata(processing_time, confidence_score, model_version="2.1.0"):
    """Create standardized response metadata"""
    return {
        'processing_time_seconds': round(processing_time, 3),
        'confidence_score': round(confidence_score, 3),
        'model_version': model_version,
        'timestamp': datetime.now().isoformat(),
        'api_version': '1.0'
    }

def log_api_usage(endpoint, method, client_ip, processing_time, status_code):
    """Log API usage for analytics"""
    usage_data = {
        'timestamp': datetime.now().isoformat(),
        'endpoint': endpoint,
        'method': method,
        'client_ip': client_ip,
        'processing_time': processing_time,
        'status_code': status_code
    }
    
    # In production, send to analytics service
    logger.info(f"API Usage: {json.dumps(usage_data)}")

def check_system_health():
    """Check system health status"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage('/')
        free_percent = (free / total) * 100
        
        health_status['checks']['disk_space'] = {
            'status': 'ok' if free_percent > 10 else 'warning',
            'free_percent': round(free_percent, 2)
        }
    except Exception as e:
        health_status['checks']['disk_space'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Check memory usage
    try:
        import psutil
        memory = psutil.virtual_memory()
        
        health_status['checks']['memory'] = {
            'status': 'ok' if memory.percent < 80 else 'warning',
            'usage_percent': memory.percent
        }
    except ImportError:
        health_status['checks']['memory'] = {
            'status': 'unknown',
            'error': 'psutil not available'
        }
    
    # Overall status
    check_statuses = [check['status'] for check in health_status['checks'].values()]
    if 'error' in check_statuses:
        health_status['status'] = 'unhealthy'
    elif 'warning' in check_statuses:
        health_status['status'] = 'degraded'
    
    return health_status

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def cleanup_temp_files(temp_dir='temp', max_age_hours=24):
    """Clean up temporary files older than specified age"""
    if not os.path.exists(temp_dir):
        return
    
    current_time = datetime.now()
    max_age = timedelta(hours=max_age_hours)
    
    cleaned_count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if current_time - file_time > max_age:
                    os.remove(file_path)
                    cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} temporary files")
        
    except Exception as e:
        logger.error(f"Error cleaning temp files: {e}")

def get_client_info(request):
    """Extract client information from request"""
    return {
        'ip_address': request.client.host if hasattr(request, 'client') else 'unknown',
        'user_agent': request.headers.get('user-agent', 'unknown'),
        'referer': request.headers.get('referer', ''),
        'timestamp': datetime.now().isoformat()
    }