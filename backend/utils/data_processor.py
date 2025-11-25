import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

class CrimeDataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.scaler = StandardScaler()
        self.model = None
        
    def load_and_preprocess_data(self):
        df = pd.read_csv(self.csv_path)
        
        df = df.drop_duplicates(subset=['nm_pol', 'lat', 'long'], keep='first')
        
        df['crime_severity'] = (
            df['murder'] * 0.20 + 
            df['rape'] * 0.18 + 
            df['gangrape'] * 0.18 + 
            df['robbery'] * 0.05 + 
            df['theft'] * 0.04 + 
            df['assualt murders'] * 0.19 + 
            df['sexual harassement'] * 0.16
        )  
        
        df['crime_density'] = df['totalcrime'] / df['totarea']
        
        df['crime_risk_score'] = (df['crime_density'] - df['crime_density'].min()) / \
                                  (df['crime_density'].max() - df['crime_density'].min()) * 100
        df.to_csv("data/preprocessed.csv")
        return df
    
    def train_model(self, df):
        feature_cols = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 
                       'assualt murders', 'sexual harassement', 'totalcrime', 
                       'totarea', 'crime_density', 'crime_severity']
        
        X = df[feature_cols]
        y = df['crime_risk_score']
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestRegressor(
            n_estimators=50,
            max_depth=8,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        
        return self.model
    
    def save_model(self, model_dir='data/processed'):
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_dir, 'crime_model.pkl'))
        joblib.dump(self.scaler, os.path.join(model_dir, 'scaler.pkl'))
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir='data/processed'):
        model_path = os.path.join(model_dir, 'crime_model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded cached model from {model_dir}")
            return True
        print(f"No cached model found at {model_dir}")
        return False