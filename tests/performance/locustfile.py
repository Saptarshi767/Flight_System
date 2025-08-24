"""
Performance tests for Flight Scheduling Analysis System using Locust
"""

import json
import random
from datetime import datetime, timedelta
from locust import HttpUser, task, between

class FlightAnalysisUser(HttpUser):
    """Simulated user for performance testing"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        # Health check to ensure system is ready
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception(f"System not healthy: {response.status_code}")
    
    @task(3)
    def get_flights(self):
        """Test flight data retrieval"""
        params = {
            'limit': random.randint(10, 100),
            'offset': random.randint(0, 1000)
        }
        
        with self.client.get("/api/v1/data/flights", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'flights' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def get_delay_analysis(self):
        """Test delay analysis endpoint"""
        params = {
            'airport': random.choice(['BOM', 'DEL']),
            'date_from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
            'date_to': datetime.now().strftime('%Y-%m-%d')
        }
        
        with self.client.get("/api/v1/analysis/delays", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'analysis' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(2)
    def get_congestion_analysis(self):
        """Test congestion analysis endpoint"""
        params = {
            'airport': random.choice(['BOM', 'DEL']),
            'time_range': random.choice(['1h', '6h', '24h'])
        }
        
        with self.client.get("/api/v1/analysis/congestion", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'congestion' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def post_nlp_query(self):
        """Test natural language query endpoint"""
        queries = [
            "What's the best time to fly from Mumbai to Delhi?",
            "Which flights have the most delays?",
            "Show me congestion patterns for today",
            "What are the peak hours at Delhi airport?",
            "Which airlines have the best on-time performance?"
        ]
        
        payload = {
            'query': random.choice(queries),
            'context': {
                'airport': random.choice(['BOM', 'DEL']),
                'date': datetime.now().strftime('%Y-%m-%d')
            }
        }
        
        with self.client.post("/api/v1/nlp/query", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'response' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 429:
                # Rate limited - this is expected behavior
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_schedule_impact(self):
        """Test schedule impact analysis"""
        payload = {
            'flight_id': f'AI{random.randint(100, 999)}',
            'proposed_time': (datetime.now() + timedelta(hours=random.randint(1, 24))).isoformat(),
            'analysis_type': random.choice(['delay_impact', 'congestion_impact', 'network_impact'])
        }
        
        with self.client.post("/api/v1/analysis/schedule-impact", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'impact_analysis' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_cascading_impact(self):
        """Test cascading impact analysis"""
        params = {
            'flight_id': f'AI{random.randint(100, 999)}',
            'delay_minutes': random.randint(15, 180)
        }
        
        with self.client.get("/api/v1/analysis/cascading-impact", params=params, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'cascading_analysis' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                # Prometheus format should contain metric names
                if 'app_requests_total' in response.text:
                    response.success()
                else:
                    response.failure("Invalid metrics format")
            else:
                response.failure(f"HTTP {response.status_code}")

class AdminUser(HttpUser):
    """Admin user for testing administrative endpoints"""
    
    wait_time = between(2, 5)
    weight = 1  # Lower weight than regular users
    
    @task(1)
    def upload_flight_data(self):
        """Test file upload endpoint"""
        # Simulate CSV upload
        csv_data = """flight_number,origin,destination,scheduled_departure,actual_departure
AI101,BOM,DEL,2024-01-01T10:00:00,2024-01-01T10:15:00
AI102,DEL,BOM,2024-01-01T14:00:00,2024-01-01T14:05:00"""
        
        files = {'file': ('test_data.csv', csv_data, 'text/csv')}
        
        with self.client.post("/api/v1/data/upload", files=files, catch_response=True) as response:
            if response.status_code in [200, 201]:
                response.success()
            elif response.status_code == 413:
                # File too large - acceptable for test
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def get_system_status(self):
        """Test system status endpoint"""
        with self.client.get("/api/v1/system/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if 'status' in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            else:
                response.failure(f"HTTP {response.status_code}")

class DashboardUser(HttpUser):
    """User accessing the Streamlit dashboard"""
    
    host = "http://localhost:8501"  # Streamlit default port
    wait_time = between(3, 8)
    weight = 2
    
    @task(1)
    def access_dashboard(self):
        """Test dashboard access"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                if 'streamlit' in response.text.lower():
                    response.success()
                else:
                    response.failure("Not a Streamlit page")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Test dashboard health check"""
        with self.client.get("/_stcore/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

# Custom test scenarios
class StressTestUser(HttpUser):
    """High-load stress testing user"""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    weight = 1
    
    @task
    def rapid_requests(self):
        """Make rapid requests to test system limits"""
        endpoints = [
            "/health",
            "/api/v1/data/flights?limit=10",
            "/metrics"
        ]
        
        endpoint = random.choice(endpoints)
        with self.client.get(endpoint, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 429:
                # Rate limited - expected under stress
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

# Test configuration for different scenarios
class WebsiteUser(HttpUser):
    """Regular website user behavior"""
    tasks = [FlightAnalysisUser]
    min_wait = 1000
    max_wait = 3000

class PowerUser(HttpUser):
    """Power user with more complex queries"""
    tasks = [FlightAnalysisUser, AdminUser]
    min_wait = 500
    max_wait = 2000
    weight = 3

class CasualUser(HttpUser):
    """Casual user with simple queries"""
    tasks = [FlightAnalysisUser]
    min_wait = 3000
    max_wait = 10000
    weight = 5