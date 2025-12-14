from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
import sys
import traceback

print("="*60)
print("STARTING SAFE ROUTING SYSTEM")
print("="*60)
from models import crime_model, route_optimizer
from utils import data_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.crime_model import CrimeRiskAnalyzer
    from models.route_optimizer import SafeRouteOptimizer
    from utils.data_processor import CrimeDataProcessor
    print("Modules imported")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

app = Flask(__name__, static_folder='../frontend')
CORS(app)

crime_df = None
crime_analyzer = None
route_optimizer = None
initialized = False

def initialize_system():
    global crime_df, crime_analyzer, route_optimizer, initialized
    
    print("\n" + "="*60)
    print("INITIALIZATION STARTING")
    print("="*60 + "\n")
    
    try:
        csv_path = 'data/preprocessed.csv'
        if not os.path.exists(csv_path):
            print(f"Crime data not found: {csv_path}")
            return False
        
        print("Loading crime data...")
        processor = CrimeDataProcessor(csv_path)
        crime_df = processor.load_and_preprocess_data()
        print(f"Loaded {len(crime_df)} records")
        
        if processor.load_model():
            print("Model loaded from cache")
        else:
            print("Training model")
            processor.train_model(crime_df)
            processor.save_model()
            print("Model trained and saved")
        
        print("Initializing crime analyzer...")
        crime_analyzer = CrimeRiskAnalyzer(crime_df)
        print("Analyzer ready")
        
        print("Initializing route optimizer...")
        route_optimizer = SafeRouteOptimizer(crime_analyzer)
        
        print("\nLoading road network...")
        print("Area: Central Delhi (28.50-28.75 N, 77.05-77.35 E)")
        success = route_optimizer.load_graph(
            north=28.75, south=28.50, east=77.35, west=77.05
        )
        
        if not success:
            print("Failed to load road network")
            return False
        
        print("\nAdding crime weights...")
        route_optimizer.add_crime_weights()
        
        initialized = True
        print("\n" + "="*60)
        print("INITIALIZATION COMPLETE")
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        traceback.print_exc()
        print("\n" + "="*60)
        print("INITIALIZATION FAILED")
        print("="*60 + "\n")
        return False

@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'initialized': initialized,
        'message': 'Ready' if initialized else 'Not initialized'
    })

@app.route('/api/initialize', methods=['POST'])
def init_system():
    global initialized
    
    if initialized:
        return jsonify({'success': True, 'message': 'Already initialized'})
    
    success = initialize_system()
    return jsonify({
        'success': success,
        'message': 'Initialized' if success else 'Failed'
    })

@app.route('/api/route', methods=['POST'])
def get_route():
    if not initialized:
        return jsonify({'error': 'Not initialized'}), 400
    
    data = request.json
    start_lat = data.get('start_lat')
    start_lon = data.get('start_lon')
    end_lat = data.get('end_lat')
    end_lon = data.get('end_lon')
    
    if None in [start_lat, start_lon, end_lat, end_lon]:
        return jsonify({'error': 'Missing coordinates'}), 400
    
    try:
        start_lat = float(start_lat)
        start_lon = float(start_lon)
        end_lat = float(end_lat)
        end_lon = float(end_lon)
    except:
        return jsonify({'error': 'Invalid coordinates'}), 400
    
    if not (28.50 <= start_lat <= 28.75 and 77.05 <= start_lon <= 77.35):
        return jsonify({'error': 'Start outside area (28.50-28.75 N, 77.05-77.35 E)'}), 400
    
    if not (28.50 <= end_lat <= 28.75 and 77.05 <= end_lon <= 77.35):
        return jsonify({'error': 'End outside area (28.50-28.75 N, 77.05-77.35 E)'}), 400
    
    try:
        print(f"\nFinding routes: ({start_lat},{start_lon}) to ({end_lat},{end_lon})")
        
        routes = route_optimizer.find_routes(start_lat, start_lon, end_lat, end_lon)
        
        if not routes:
            return jsonify({'error': 'No routes found'}), 404
        
        for route_type, route_data in routes.items():
            if route_data:
                risk = route_data['avg_crime_risk']
                if risk < 25:
                    level = 'Low'
                    
                elif risk < 50:
                    level = 'Medium'
                    
                elif risk < 75:
                    level = 'High'
                    
                else:
                    level = 'Very High'
                    
                route_data['risk_level'] = level
                
                print(f"  {route_type}: {route_data['distance_km']:.2f}km, safety={route_data['safety_score']:.1f}")
        
        return jsonify({
            'success': True,
            'routes': routes,
            'start': {'lat': start_lat, 'lon': start_lon},
            'end': {'lat': end_lat, 'lon': end_lon}
        })
        
    except Exception as e:
        print(f"Route error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/crime-risk', methods=['POST'])
def get_risk():
    if not initialized:
        return jsonify({'error': 'Not initialized'}), 400
    
    data = request.json
    lat = data.get('lat')
    lon = data.get('lon')
    
    if lat is None or lon is None:
        return jsonify({'error': 'Missing coordinates'}), 400
    
    try:
        risk = crime_analyzer.get_risk_score(lat, lon)
        return jsonify({
            'lat': lat,
            'lon': lon,
            'risk_score': float(risk),
            'safety_score': float(100 - risk)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/crime-data', methods=['GET'])
def get_crime_data():
    global crime_df
    if crime_df is None:
        return jsonify({'error': 'No crime data loaded'}), 400
    
    try:
        crime_columns = ['murder', 'rape', 'gangrape', 'robbery', 'theft', 'assualt murders', 'sexual harassement']
        
        crime_stats = {}
        for col in crime_columns:
            if col in crime_df.columns:
                total = int(crime_df[col].sum())
                crime_stats[col.replace('assualt murders', 'Assault Murders')
                                       .replace('sexual harassement', 'Sexual Harassment')
                                       .replace('gangrape', 'Gang Rape')
                                       .title()] = total
        
        print("\n" + "="*60)
        print("ACTUAL CRIME DATA FROM DATASET")
        print("="*60)
        for crime_type, count in sorted(crime_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {crime_type}: {count}")
        print("="*60 + "\n")
        
        return jsonify({
            'success': True,
            'crime_data': crime_stats,
            'total_records': len(crime_df),
            'total_crimes': int(crime_df[crime_columns].sum().sum())
        })
    except Exception as e:
        print(f"Error getting crime data: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\nFlask ready")
    print("\nNEXT STEPS:")
    print("1. Open: http://localhost:5000")
    print("2. Click: Initialize System")
    print("3. Wait: 2-3 min (first time) or 5 sec (later)")
    print("4. Enter coordinates and find routes")
    print("\nSample coordinates:")
    print("  Start: 28.6328, 77.2197")
    print("  End:   28.6129, 77.2295")
    print("\n" + "="*60 + "\n")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)