import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    
    # Camera settings
    CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', 0))
    
    # Model settings
    MODEL_PATH = os.environ.get('MODEL_PATH') or 'sign_language_model.h5'
    DATA_DIR = os.environ.get('DATA_DIR') or 'Data'
    
    # Detection settings
    CONFIDENCE_THRESHOLD = float(os.environ.get('CONFIDENCE_THRESHOLD', 0.7))
    
    # Server settings
    HOST = os.environ.get('HOST') or '0.0.0.0'
    PORT = int(os.environ.get('PORT', 5000))
    DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'