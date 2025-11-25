import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

class CrimeRiskAnalyzer:
    def __init__(self, crime_df):
        self.crime_df = crime_df
        self.build_spatial_index()
        
    def build_spatial_index(self):
        coords = self.crime_df[['lat', 'long']].values
        self.tree = cKDTree(coords)
        
    def get_risk_score(self, lat, lon, radius=0.01):
        
        indices = self.tree.query_ball_point([lat, lon], radius)
        
        if not indices:
            _, idx = self.tree.query([lat, lon])
            return self.crime_df.iloc[idx]['crime_risk_score']
        
        nearby_data = self.crime_df.iloc[indices]
        distances = np.sqrt(
            (nearby_data['lat'] - lat)**2 + 
            (nearby_data['long'] - lon)**2
        )
        
        # Inverse distance weighting
        weights = 1 / (distances + 0.0001)
        weights = weights / weights.sum()
        
        weighted_risk = (nearby_data['crime_risk_score'] * weights).sum()
        
        return weighted_risk
    
    def get_route_segments_risk(self, route_coords):
        risks = []
        for lat, lon in route_coords:
            risk = self.get_risk_score(lat, lon)
            risks.append(risk)
        
        return risks
    
    def calculate_route_safety_score(self, route_coords):
        segment_risks = self.get_route_segments_risk(route_coords)
        
        return {
            'average_risk': np.mean(segment_risks),
            'max_risk': np.max(segment_risks),
            'min_risk': np.min(segment_risks),
            'safety_score': 100 - np.mean(segment_risks),
            'risk_variance': np.var(segment_risks)
        }