import os

class Config:
    
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = True
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    CRIME_CSV_PATH = os.path.join(DATA_DIR, 'crime.csv')
    MODEL_PATH = os.path.join(PROCESSED_DIR, 'crime_model.pkl')
    SCALER_PATH = os.path.join(PROCESSED_DIR, 'scaler.pkl')
    
    DELHI_BOUNDS = {
        'north': 28.88,
        'south': 28.40,
        'east': 77.35,
        'west': 76.85
    }
    
    ML_CONFIG = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    
    CRIME_WEIGHTS = {
        'murder': 20,
        'rape': 18,
        'gangrape': 18,
        'robbery': 5,
        'theft': 4,
        'assualt_murders': 19,
        'sexual_harassement': 16
    }
    
    ROUTE_CONFIG = {
        'risk_search_radius': 0.01,
        'risk_weight_factor': 2.0,
        'balanced_distance_weight': 0.6,
        'balanced_safety_weight': 0.4
    }
    
    OSM_CONFIG = {
        'network_type': 'drive',
        'simplify': True,
        'retain_all': False
    }
    
    CORS_ORIGINS = '*'
    
    CACHE_ENABLED = True
    MAX_ROUTE_SEGMENTS = 1000
    
    @classmethod
    def get_crime_csv_path(cls):
        return cls.CRIME_CSV_PATH
    
    @classmethod
    def get_model_paths(cls):
        return cls.MODEL_PATH, cls.SCALER_PATH
    
    @classmethod
    def ensure_directories(cls):
        os.makedirs(cls.DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
    
    @classmethod
    def get_delhi_bounds(cls):
        return (
            cls.DELHI_BOUNDS['north'],
            cls.DELHI_BOUNDS['south'],
            cls.DELHI_BOUNDS['east'],
            cls.DELHI_BOUNDS['west']
        )


class DevelopmentConfig(Config):
    FLASK_DEBUG = True


class ProductionConfig(Config):
    FLASK_DEBUG = False
    CORS_ORIGINS = [
        'https://yourdomain.com',
        'https://www.yourdomain.com'
    ]


class TestingConfig(Config):
    DELHI_BOUNDS = {
        'north': 28.70,
        'south': 28.55,
        'east': 77.30,
        'west': 77.10
    }
    ML_CONFIG = {
        'n_estimators': 50,
        'max_depth': 5,
        'random_state': 42,
        'n_jobs': 1
    }


config_map = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}

def get_config(env='development'):
    return config_map.get(env, DevelopmentConfig)