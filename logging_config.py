"""
Logging Configuration for Crop Analysis AI System
"""
import logging
import logging.handlers
import os
from datetime import datetime
from config import LOGGING, PATHS

def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create logs directory
    os.makedirs(PATHS['LOGS_DIR'], exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOGGING['LEVEL']))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for all logs
    log_file = os.path.join(PATHS['LOGS_DIR'], f"crop_analysis_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Error file handler
    error_file = os.path.join(PATHS['LOGS_DIR'], f"errors_{datetime.now().strftime('%Y%m%d')}.log")
    error_handler = logging.handlers.RotatingFileHandler(
        error_file, maxBytes=5*1024*1024, backupCount=3
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_handler)
    
    # API access log
    api_logger = logging.getLogger('api_access')
    api_file = os.path.join(PATHS['LOGS_DIR'], f"api_access_{datetime.now().strftime('%Y%m%d')}.log")
    api_handler = logging.handlers.RotatingFileHandler(
        api_file, maxBytes=20*1024*1024, backupCount=7
    )
    api_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(message)s'
    ))
    api_logger.addHandler(api_handler)
    api_logger.setLevel(logging.INFO)
    
    # Model training log
    training_logger = logging.getLogger('model_training')
    training_file = os.path.join(PATHS['LOGS_DIR'], f"training_{datetime.now().strftime('%Y%m%d')}.log")
    training_handler = logging.FileHandler(training_file)
    training_handler.setFormatter(detailed_formatter)
    training_logger.addHandler(training_handler)
    training_logger.setLevel(logging.DEBUG)
    
    # Weather API log
    weather_logger = logging.getLogger('weather_api')
    weather_file = os.path.join(PATHS['LOGS_DIR'], f"weather_{datetime.now().strftime('%Y%m%d')}.log")
    weather_handler = logging.FileHandler(weather_file)
    weather_handler.setFormatter(detailed_formatter)
    weather_logger.addHandler(weather_handler)
    weather_logger.setLevel(logging.INFO)
    
    logging.info("Logging system initialized")

class APILogger:
    """Custom logger for API requests"""
    
    def __init__(self):
        self.logger = logging.getLogger('api_access')
    
    def log_request(self, method, endpoint, client_ip, user_agent, file_size=None):
        """Log API request"""
        message = f"{method} {endpoint} - IP: {client_ip} - Agent: {user_agent}"
        if file_size:
            message += f" - FileSize: {file_size}MB"
        self.logger.info(message)
    
    def log_response(self, endpoint, status_code, processing_time, response_size=None):
        """Log API response"""
        message = f"Response {endpoint} - Status: {status_code} - Time: {processing_time:.2f}s"
        if response_size:
            message += f" - Size: {response_size}KB"
        self.logger.info(message)
    
    def log_error(self, endpoint, error_message, client_ip):
        """Log API error"""
        message = f"ERROR {endpoint} - IP: {client_ip} - Error: {error_message}"
        self.logger.error(message)

class PerformanceLogger:
    """Logger for performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
        
        # Performance log file
        perf_file = os.path.join(PATHS['LOGS_DIR'], f"performance_{datetime.now().strftime('%Y%m%d')}.log")
        perf_handler = logging.FileHandler(perf_file)
        perf_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(message)s'
        ))
        self.logger.addHandler(perf_handler)
        self.logger.setLevel(logging.INFO)
    
    def log_analysis_performance(self, filename, processing_time, image_size, model_inference_time):
        """Log image analysis performance"""
        message = (f"Analysis - File: {filename} - "
                  f"Total: {processing_time:.2f}s - "
                  f"ImageSize: {image_size}KB - "
                  f"ModelTime: {model_inference_time:.2f}s")
        self.logger.info(message)
    
    def log_api_performance(self, endpoint, response_time, memory_usage=None):
        """Log API endpoint performance"""
        message = f"API - {endpoint} - Time: {response_time:.2f}s"
        if memory_usage:
            message += f" - Memory: {memory_usage}MB"
        self.logger.info(message)
    
    def log_database_performance(self, operation, execution_time, records_affected=None):
        """Log database operation performance"""
        message = f"DB - {operation} - Time: {execution_time:.3f}s"
        if records_affected:
            message += f" - Records: {records_affected}"
        self.logger.info(message)

class SecurityLogger:
    """Logger for security events"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
        
        # Security log file
        security_file = os.path.join(PATHS['LOGS_DIR'], f"security_{datetime.now().strftime('%Y%m%d')}.log")
        security_handler = logging.FileHandler(security_file)
        security_handler.setFormatter(logging.Formatter(
            '%(asctime)s - SECURITY - %(message)s'
        ))
        self.logger.addHandler(security_handler)
        self.logger.setLevel(logging.WARNING)
    
    def log_suspicious_activity(self, client_ip, activity_description):
        """Log suspicious activity"""
        message = f"Suspicious activity from {client_ip}: {activity_description}"
        self.logger.warning(message)
    
    def log_rate_limit_exceeded(self, client_ip, endpoint, request_count):
        """Log rate limit violations"""
        message = f"Rate limit exceeded - IP: {client_ip} - Endpoint: {endpoint} - Requests: {request_count}"
        self.logger.warning(message)
    
    def log_invalid_file_upload(self, client_ip, filename, reason):
        """Log invalid file upload attempts"""
        message = f"Invalid file upload - IP: {client_ip} - File: {filename} - Reason: {reason}"
        self.logger.warning(message)
    
    def log_authentication_failure(self, client_ip, attempted_key):
        """Log authentication failures"""
        message = f"Authentication failure - IP: {client_ip} - Key: {attempted_key[:10]}..."
        self.logger.error(message)

def get_logger(name):
    """Get logger instance with proper configuration"""
    return logging.getLogger(name)

def log_system_startup():
    """Log system startup information"""
    logger = logging.getLogger('system')
    logger.info("=" * 50)
    logger.info("CROP ANALYSIS AI SYSTEM STARTUP")
    logger.info("=" * 50)
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info(f"Log Level: {LOGGING['LEVEL']}")
    logger.info(f"Logs Directory: {PATHS['LOGS_DIR']}")
    logger.info("System initialized successfully")
    logger.info("=" * 50)

def log_system_shutdown():
    """Log system shutdown"""
    logger = logging.getLogger('system')
    logger.info("=" * 50)
    logger.info("CROP ANALYSIS AI SYSTEM SHUTDOWN")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("System shutdown completed")
    logger.info("=" * 50)