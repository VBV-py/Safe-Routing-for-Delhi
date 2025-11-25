import osmnx as ox
import networkx as nx
import numpy as np
import pickle
import os

class SafeRouteOptimizer:
    def __init__(self, crime_analyzer):
        self.crime_analyzer = crime_analyzer
        self.graph = None
        self.graph_cache_file = 'data/processed/osm_graph.pkl'
        
    def load_graph(self, north, south, east, west):
        
        # Try to load cached graph first
        if os.path.exists(self.graph_cache_file):
            try:
                print(f"Loading cached OSM graph...")
                with open(self.graph_cache_file, 'rb') as f:
                    self.graph = pickle.load(f)
                print(f"Loaded cached graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
                return True
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print("Downloading fresh graph...")
        
        try:
            # Download street network
            print("Downloading OSM data (first time only, 2-3 minutes)...")
            self.graph = ox.graph_from_bbox(north, south, east, west, network_type='drive', simplify=True)
            print(f"Downloaded graph: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
            # Cache the graph
            try:
                os.makedirs(os.path.dirname(self.graph_cache_file), exist_ok=True)
                with open(self.graph_cache_file, 'wb') as f:
                    pickle.dump(self.graph, f)
                print(f"Cached graph for faster initialization next time")
            except Exception as e:
                print(f"Failed to cache: {e}")
            
            return True
        except Exception as e:
            print(f"Error loading graph: {e}")
            return False
    
    def add_crime_weights(self):
        if self.graph is None:
            return False
        
        # Check if already weighted
        sample_edge = list(self.graph.edges(keys=True, data=True))[0]
        if 'crime_risk' in sample_edge[3]:
            print("Crime weights already in cached graph")
            return True
        
        print("Adding crime weights to roads...")
        edges = list(self.graph.edges(keys=True, data=True))
        total = len(edges)
        
        for idx, (u, v, key, data) in enumerate(edges):
            # Get edge midpoint
            u_lat = self.graph.nodes[u]['y']
            u_lon = self.graph.nodes[u]['x']
            v_lat = self.graph.nodes[v]['y']
            v_lon = self.graph.nodes[v]['x']
            
            mid_lat = (u_lat + v_lat) / 2
            mid_lon = (u_lon + v_lon) / 2
            
            # Get crime risk
            risk_score = self.crime_analyzer.get_risk_score(mid_lat, mid_lon)
            
            # Calculate weights
            length = data.get('length', 100)
            risk_factor = 1 + (risk_score / 100) * 2
            
            data['crime_risk'] = risk_score
            data['safe_weight'] = length * risk_factor
            data['original_length'] = length
            
            if (idx + 1) % 2000 == 0:
                print(f"  Processed {idx + 1}/{total} edges...")
        
        print(f"Added weights to {total} edges")
        
        # Update cache
        try:
            with open(self.graph_cache_file, 'wb') as f:
                pickle.dump(self.graph, f)
            print("Updated cache")
        except:
            pass
        
        return True
    
    def find_routes(self, start_lat, start_lon, end_lat, end_lon):
        if self.graph is None:
            return None
        
        try:
            # Find nearest nodes
            start_node = ox.distance.nearest_nodes(self.graph, start_lon, start_lat)
            end_node = ox.distance.nearest_nodes(self.graph, end_lon, end_lat)
            
            if start_node == end_node:
                print("Start and end are same location")
                return None
            
            routes = {}
            
            # Shortest route
            try:
                shortest = nx.shortest_path(self.graph, start_node, end_node, weight='length')
                routes['shortest'] = self.process_route(shortest, 'shortest')
            except nx.NetworkXNoPath:
                routes['shortest'] = None
            
            # Safest route
            try:
                safest = nx.shortest_path(self.graph, start_node, end_node, weight='safe_weight')
                routes['safest'] = self.process_route(safest, 'safest')
            except nx.NetworkXNoPath:
                routes['safest'] = None
            
            # Balanced route
            try:
                for u, v, key, data in self.graph.edges(keys=True, data=True):
                    data['balanced_weight'] = (
                        0.6 * data.get('original_length', 100) + 
                        0.4 * data.get('safe_weight', 100)
                    )
                
                balanced = nx.shortest_path(self.graph, start_node, end_node, weight='balanced_weight')
                routes['balanced'] = self.process_route(balanced, 'balanced')
            except nx.NetworkXNoPath:
                routes['balanced'] = None
            
            return routes
            
        except Exception as e:
            print(f"Error finding routes: {e}")
            return None
    
    def process_route(self, node_list, route_type):
        if not node_list:
            return None
        
        # Get coordinates
        coords = []
        for node in node_list:
            coords.append({
                'lat': self.graph.nodes[node]['y'],
                'lon': self.graph.nodes[node]['x']
            })
        
        # Calculate metrics
        total_distance = 0
        segment_risks = []
        
        for i in range(len(node_list) - 1):
            u = node_list[i]
            v = node_list[i + 1]
            
            edge_data = None
            if self.graph.is_multigraph():
                edges = self.graph.get_edge_data(u, v)
                if edges:
                    edge_data = list(edges.values())[0]
            else:
                edge_data = self.graph.get_edge_data(u, v)
            
            if edge_data:
                total_distance += edge_data.get('original_length', 0)
                segment_risks.append(edge_data.get('crime_risk', 0))
        
        avg_risk = np.mean(segment_risks) if segment_risks else 0
        
        return {
            'type': route_type,
            'coordinates': coords,
            'distance_km': total_distance / 1000,
            'avg_crime_risk': float(avg_risk),
            'safety_score': float(100 - avg_risk),
            'max_crime_risk': float(max(segment_risks)) if segment_risks else 0,
            'num_segments': len(node_list) - 1
        }