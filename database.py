"""
Database Module for Crop Analysis Results Storage
"""
import sqlite3
import json
from datetime import datetime
import logging
import os
from config import PATHS

class CropAnalysisDB:
    def __init__(self, db_path='crop_analysis.db'):
        self.db_path = os.path.join(PATHS.get('DATA_DIR', ''), db_path)
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialize database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Analysis results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS analysis_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT NOT NULL,
                        analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        yield_prediction REAL,
                        land_use_percentages TEXT,
                        vegetation_health TEXT,
                        weather_factors TEXT,
                        soil_conditions TEXT,
                        satellite_metadata TEXT,
                        confidence_score REAL,
                        processing_time REAL,
                        location_lat REAL,
                        location_lon REAL
                    )
                ''')
                
                # Weather history table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS weather_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        temperature REAL,
                        humidity REAL,
                        pressure REAL,
                        wind_speed REAL,
                        cloud_cover REAL,
                        weather_condition TEXT,
                        precipitation REAL,
                        location_lat REAL,
                        location_lon REAL
                    )
                ''')
                
                # Model performance table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_version TEXT,
                        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        accuracy REAL,
                        loss REAL,
                        validation_accuracy REAL,
                        validation_loss REAL,
                        training_samples INTEGER,
                        epochs INTEGER
                    )
                ''')
                
                # User sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT UNIQUE,
                        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        end_time TIMESTAMP,
                        total_analyses INTEGER DEFAULT 0,
                        user_agent TEXT,
                        ip_address TEXT
                    )
                ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization failed: {e}")
            raise
    
    def store_analysis_result(self, result_data):
        """Store analysis result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO analysis_results (
                        filename, yield_prediction, land_use_percentages,
                        vegetation_health, weather_factors, soil_conditions,
                        satellite_metadata, confidence_score, processing_time,
                        location_lat, location_lon
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result_data.get('filename', ''),
                    result_data.get('yield_prediction', 0),
                    json.dumps(result_data.get('land_use_percentages', {})),
                    json.dumps(result_data.get('vegetation_health', {})),
                    json.dumps(result_data.get('weather_factors', {})),
                    json.dumps(result_data.get('soil_conditions', {})),
                    json.dumps(result_data.get('satellite_metadata', {})),
                    result_data.get('analysis_confidence', 0),
                    result_data.get('processing_time_seconds', 0),
                    result_data.get('location', {}).get('lat', 30.9010),
                    result_data.get('location', {}).get('lon', 75.8573)
                ))
                
                conn.commit()
                analysis_id = cursor.lastrowid
                self.logger.info(f"Analysis result stored with ID: {analysis_id}")
                return analysis_id
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store analysis result: {e}")
            return None
    
    def store_weather_data(self, weather_data):
        """Store weather data in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO weather_history (
                        temperature, humidity, pressure, wind_speed,
                        cloud_cover, weather_condition, precipitation,
                        location_lat, location_lon
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    weather_data.get('temperature', 0),
                    weather_data.get('humidity', 0),
                    weather_data.get('pressure', 0),
                    weather_data.get('wind_speed', 0),
                    weather_data.get('cloud_cover', 0),
                    weather_data.get('weather_condition', ''),
                    weather_data.get('precipitation', 0),
                    30.9010,  # Ludhiana lat
                    75.8573   # Ludhiana lon
                ))
                
                conn.commit()
                self.logger.info("Weather data stored successfully")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store weather data: {e}")
    
    def get_analysis_history(self, limit=50):
        """Get recent analysis history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM analysis_results 
                    ORDER BY analysis_date DESC 
                    LIMIT ?
                ''', (limit,))
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    result = dict(zip(columns, row))
                    
                    # Parse JSON fields
                    for json_field in ['land_use_percentages', 'vegetation_health', 'weather_factors', 'soil_conditions', 'satellite_metadata']:
                        if result[json_field]:
                            try:
                                result[json_field] = json.loads(result[json_field])
                            except json.JSONDecodeError:
                                result[json_field] = {}
                    
                    results.append(result)
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get analysis history: {e}")
            return []
    
    def get_weather_trends(self, days=30):
        """Get weather trends for specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM weather_history 
                    WHERE date >= datetime('now', '-{} days')
                    ORDER BY date DESC
                '''.format(days))
                
                columns = [description[0] for description in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get weather trends: {e}")
            return []
    
    def get_yield_statistics(self):
        """Get yield prediction statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT 
                        AVG(yield_prediction) as avg_yield,
                        MIN(yield_prediction) as min_yield,
                        MAX(yield_prediction) as max_yield,
                        COUNT(*) as total_analyses,
                        AVG(confidence_score) as avg_confidence
                    FROM analysis_results
                    WHERE yield_prediction > 0
                ''')
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        'average_yield': round(result[0] or 0, 2),
                        'minimum_yield': round(result[1] or 0, 2),
                        'maximum_yield': round(result[2] or 0, 2),
                        'total_analyses': result[3] or 0,
                        'average_confidence': round(result[4] or 0, 2)
                    }
                
                return {}
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get yield statistics: {e}")
            return {}
    
    def get_land_use_trends(self):
        """Get land use distribution trends"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT land_use_percentages FROM analysis_results
                    WHERE land_use_percentages IS NOT NULL
                    ORDER BY analysis_date DESC
                    LIMIT 100
                ''')
                
                results = cursor.fetchall()
                land_use_totals = {}
                count = 0
                
                for row in results:
                    try:
                        land_use_data = json.loads(row[0])
                        for land_type, percentage in land_use_data.items():
                            if land_type not in land_use_totals:
                                land_use_totals[land_type] = 0
                            land_use_totals[land_type] += percentage
                        count += 1
                    except json.JSONDecodeError:
                        continue
                
                # Calculate averages
                if count > 0:
                    land_use_averages = {
                        land_type: round(total / count, 2)
                        for land_type, total in land_use_totals.items()
                    }
                    return land_use_averages
                
                return {}
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get land use trends: {e}")
            return {}
    
    def store_model_performance(self, performance_data):
        """Store model training performance"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO model_performance (
                        model_version, accuracy, loss, validation_accuracy,
                        validation_loss, training_samples, epochs
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance_data.get('model_version', '1.0'),
                    performance_data.get('accuracy', 0),
                    performance_data.get('loss', 0),
                    performance_data.get('validation_accuracy', 0),
                    performance_data.get('validation_loss', 0),
                    performance_data.get('training_samples', 0),
                    performance_data.get('epochs', 0)
                ))
                
                conn.commit()
                self.logger.info("Model performance data stored")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to store model performance: {e}")
    
    def create_session(self, session_id, user_agent='', ip_address=''):
        """Create new user session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO user_sessions (
                        session_id, user_agent, ip_address
                    ) VALUES (?, ?, ?)
                ''', (session_id, user_agent, ip_address))
                
                conn.commit()
                self.logger.info(f"Session created: {session_id}")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to create session: {e}")
    
    def update_session_analysis_count(self, session_id):
        """Update analysis count for session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE user_sessions 
                    SET total_analyses = total_analyses + 1
                    WHERE session_id = ?
                ''', (session_id,))
                
                conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to update session analysis count: {e}")
    
    def get_system_statistics(self):
        """Get overall system statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total analyses
                cursor.execute('SELECT COUNT(*) FROM analysis_results')
                total_analyses = cursor.fetchone()[0]
                
                # Analyses today
                cursor.execute('''
                    SELECT COUNT(*) FROM analysis_results 
                    WHERE date(analysis_date) = date('now')
                ''')
                analyses_today = cursor.fetchone()[0]
                
                # Average processing time
                cursor.execute('SELECT AVG(processing_time) FROM analysis_results')
                avg_processing_time = cursor.fetchone()[0] or 0
                
                # Total sessions
                cursor.execute('SELECT COUNT(*) FROM user_sessions')
                total_sessions = cursor.fetchone()[0]
                
                return {
                    'total_analyses': total_analyses,
                    'analyses_today': analyses_today,
                    'average_processing_time': round(avg_processing_time, 2),
                    'total_sessions': total_sessions,
                    'database_size_mb': self._get_database_size()
                }
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to get system statistics: {e}")
            return {}
    
    def _get_database_size(self):
        """Get database file size in MB"""
        try:
            if os.path.exists(self.db_path):
                size_bytes = os.path.getsize(self.db_path)
                size_mb = size_bytes / (1024 * 1024)
                return round(size_mb, 2)
            return 0
        except OSError:
            return 0
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old data to maintain database size"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old analysis results
                cursor.execute('''
                    DELETE FROM analysis_results 
                    WHERE analysis_date < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                # Delete old weather data
                cursor.execute('''
                    DELETE FROM weather_history 
                    WHERE date < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                # Delete old sessions
                cursor.execute('''
                    DELETE FROM user_sessions 
                    WHERE start_time < datetime('now', '-{} days')
                '''.format(days_to_keep))
                
                conn.commit()
                
                # Vacuum database to reclaim space
                cursor.execute('VACUUM')
                
                self.logger.info(f"Cleaned up data older than {days_to_keep} days")
                
        except sqlite3.Error as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
    
    def export_data(self, table_name, output_file):
        """Export table data to JSON file"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute(f'SELECT * FROM {table_name}')
                columns = [description[0] for description in cursor.description]
                
                data = []
                for row in cursor.fetchall():
                    data.append(dict(zip(columns, row)))
                
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                
                self.logger.info(f"Exported {len(data)} records from {table_name} to {output_file}")
                return True
                
        except (sqlite3.Error, IOError) as e:
            self.logger.error(f"Failed to export data: {e}")
            return False