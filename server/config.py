import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    # Flask server configuration
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    PORT = int(os.getenv('PORT', 5000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Fall detection configuration
    FALL_THRESHOLD = float(os.getenv('FALL_THRESHOLD', 0.05))
    MOTION_THRESHOLD = float(os.getenv('MOTION_THRESHOLD', 0.03))
    VELOCITY_HISTORY_SIZE = int(os.getenv('VELOCITY_HISTORY_SIZE', 10))
    IMMEDIATE_FALL_THRESHOLD = float(os.getenv('IMMEDIATE_FALL_THRESHOLD', 0.9))
    NOTIFICATION_THRESHOLD = int(os.getenv('NOTIFICATION_THRESHOLD', 10))
    VIDEO_SOURCE = os.getenv('VIDEO_SOURCE', 'pan1.mp4')
    
    # Firebase configuration for push notifications
    FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY', '')
    FIREBASE_AUTH_DOMAIN = os.getenv('FIREBASE_AUTH_DOMAIN', '')
    FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID', '')
    FIREBASE_STORAGE_BUCKET = os.getenv('FIREBASE_STORAGE_BUCKET', '')
    FIREBASE_MESSAGING_SENDER_ID = os.getenv('FIREBASE_MESSAGING_SENDER_ID', '')
    FIREBASE_APP_ID = os.getenv('FIREBASE_APP_ID', '')
    
    # Paths configuration
    SNAPSHOTS_DIR = os.getenv('SNAPSHOTS_DIR', 'static/snapshots')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    VIDEOS_DIR = os.getenv('VIDEOS_DIR', 'recorded_videos')
    
    # Security configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    
    @staticmethod
    def get_firebase_config():
        """Return Firebase configuration as a dictionary"""
        return {
            "apiKey": Config.FIREBASE_API_KEY,
            "authDomain": Config.FIREBASE_AUTH_DOMAIN,
            "projectId": Config.FIREBASE_PROJECT_ID,
            "storageBucket": Config.FIREBASE_STORAGE_BUCKET,
            "messagingSenderId": Config.FIREBASE_MESSAGING_SENDER_ID,
            "appId": Config.FIREBASE_APP_ID
        }

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    # In production, ensure you set a strong SECRET_KEY in environment variables
    
class TestingConfig(Config):
    TESTING = True
    DEBUG = True
    # Use test video or webcam for testing
    VIDEO_SOURCE = os.getenv('TEST_VIDEO_SOURCE', 'fall1.mp4')

# Set the active configuration based on environment variable
config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

active_config = config_map.get(os.getenv('FLASK_ENV', 'development'), DevelopmentConfig)