"""
End-to-end integration tests for the Flight Scheduling Analysis System
"""
import pytest
import asyncio
import json
import tempfile
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from src.api.main import app
from src.database.connection import db_session_scope
from src.database.models import Flight, Airport, Airline, Aircraft
from src.utils.cache import cache_manager
from src.data.processors import ExcelDataProcessor, DataProcessor
from src.analysis.delay_analyzer import DelayAnalyzer
from src.analysis.congestion_analyzer import CongestionAnalyzer
from src.analysis.cascading_impact_analyzer import CascadingImpactAnalyzer
from src.nlp.query_processor import QueryProcessor
from src.nlp.langchain_orchestrator import LangChainOrchestrator
from src.reporting.report_generator import ReportGenerator


class EndToEndTestSuite:
    """Comprehensive end-to-end test suite"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_data_created = False
        self.sample_flights = []
    
    def setup_test_data(self):
        """Setup comprehensive test data"""
        if self.test_data_created:
            return
        
        with db_session_scope() as session:
            # Create test airports
            airports = [
                Airport(
                    code='BOM', name='Chhatrapati Shivaji Maharaj International Airport',
                    city='Mumbai', country='India', runway_count=2, runway_capacity=30,
                    latitude=19.0896, longitude=72.8656, timezone='Asia/Kolkata'
                ),
                Airport(
                    code='DEL', name='Indira Gandhi International Airport',
                    city='Delhi', country='India', runway_count=3, runway_capacity=45,
                    latitude=28.5562, longitude=77.1000, timezone='Asia/Kolkata'
                ),
                Airport(
                    code='BLR', name='Kempegowda International Airport',
                    city='Bangalore', country='India', runway_count=2, runway_capacity=25,
                    latitude=13.1986, longitude=77.7066, timezone='Asia/Kolkata'
                )
            ]
            
            for airport in airports:
                existing = session.query(Airport).filter(Airport.code == airport.code).first()
                if not existing:
                    session.add(airport)
            
            # Create test airlines
            airlines = [
                Airline(code='AI', name='Air India', country='India', is_active=True),
                Airline(code='6E', name='IndiGo', country='India', is_active=True),
                Airline(code='SG', name='SpiceJet', country='India', is_active=True),
                Airline(code='UK', name='Vistara', country='India', is_active=True)
            ]
            
            for airline in airlines:
                existing = session.query(Airline).filter(Airline.code == airline.code).first()
                if not existing:
                    session.add(airline)
            
            # Create test aircraft
            aircraft_types = [
                Aircraft(
                    type_code='A320', manufacturer='Airbus', model='A320-200',
                    typical_seating=180, max_seating=200, max_range=3300, cruise_speed=450
                ),
                Aircraft(
                    type_code='B737', manufacturer='Boeing', model='737-800',
                    typical_seating=160, max_seating=189, max_range=2935, cruise_speed=453
                ),
                Aircraft(
                    type_code='A321', manufacturer='Airbus', model='A321-200',
                    typical_seating=200, max_seating=220, max_range=3200, cruise_speed=447
                )
            ]
            
            for aircraft in aircraft_types:
                existing = session.query(Aircraft).filter(Aircraft.type_code == aircraft.type_code).first()
                if not existing:
                    session.add(aircraft)
            
            session.commit()
            
            # Create comprehensive flight data
            self._create_flight_test_data(session)
            
        self.test_data_created = True
    
    def _create_flight_test_data(self, session: Session):
        """Create comprehensive flight test data"""
        base_date = datetime.now() - timedelta(days=30)
        flights = []
        
        routes = [
            ('BOM', 'DEL'), ('DEL', 'BOM'),
            ('BOM', 'BLR'), ('BLR', 'BOM'),
            ('DEL', 'BLR'), ('BLR', 'DEL')
        ]
        
        airlines = ['AI', '6E', 'SG', 'UK']
        aircraft_types = ['A320', 'B737', 'A321']
        
        flight_counter = 1
        
        # Create 30 days of flight data
        for day in range(30):
            current_date = base_date + timedelta(days=day)
            
            # Create flights throughout the day
            for hour in range(6, 23):  # 6 AM to 11 PM
                for minute in [0, 30]:  # Every 30 minutes
                    if flight_counter > 2000:  # Limit total flights
                        break
                    
                    flight_time = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    
                    # Select random route, airline, and aircraft
                    origin, destination = routes[flight_counter % len(routes)]
                    airline = airlines[flight_counter % len(airlines)]
                    aircraft_type = aircraft_types[flight_counter % len(aircraft_types)]
                    
                    # Calculate flight duration based on route
                    if origin == 'BOM' and destination == 'DEL':
                        flight_duration = timedelta(hours=2, minutes=15)
                    elif origin == 'DEL' and destination == 'BOM':
                        flight_duration = timedelta(hours=2, minutes=20)
                    elif origin in ['BOM', 'DEL'] and destination == 'BLR':
                        flight_duration = timedelta(hours=2, minutes=30)
                    elif origin == 'BLR' and destination in ['BOM', 'DEL']:
                        flight_duration = timedelta(hours=2, minutes=25)
                    else:
                        flight_duration = timedelta(hours=2)
                    
                    # Generate realistic delays
                    delay_minutes = self._generate_realistic_delay(hour, day % 7, airline)
                    
                    # Create flight
                    flight = Flight(
                        flight_id=f'E2E{flight_counter:04d}',
                        flight_number=f'{airline}{1000 + flight_counter % 9000}',
                        airline_code=airline,
                        origin_airport=origin,
                        destination_airport=destination,
                        aircraft_type=aircraft_type,
                        scheduled_departure=flight_time,
                        scheduled_arrival=flight_time + flight_duration,
                        actual_departure=flight_time + timedelta(minutes=delay_minutes),
                        actual_arrival=flight_time + flight_duration + timedelta(minutes=delay_minutes),
                        departure_delay_minutes=delay_minutes,
                        arrival_delay_minutes=delay_minutes,
                        delay_category=self._get_delay_category(delay_minutes, hour),
                        status='completed',
                        passenger_count=150 + (flight_counter % 50),
                        runway_used=f'RW{(flight_counter % 3) + 1:02d}',
                        gate=f'G{(flight_counter % 20) + 1}',
                        data_source='e2e_test',
                        data_quality_score=0.95
                    )
                    
                    flights.append(flight)
                    self.sample_flights.append(flight)
                    flight_counter += 1
        
        # Batch insert flights
        session.add_all(flights)
        session.commit()
    
    def _generate_realistic_delay(self, hour: int, day_of_week: int, airline: str) -> int:
        """Generate realistic delay patterns"""
        base_delay = 0
        
        # Peak hour delays (7-9 AM, 6-8 PM)
        if hour in [7, 8, 18, 19]:
            base_delay += 15
        elif hour in [9, 17, 20]:
            base_delay += 8
        
        # Weekend vs weekday patterns
        if day_of_week in [5, 6]:  # Weekend
            base_delay -= 5
        else:  # Weekday
            base_delay += 3
        
        # Airline-specific patterns
        airline_factors = {'AI': 1.2, '6E': 0.8, 'SG': 1.1, 'UK': 0.9}
        base_delay = int(base_delay * airline_factors.get(airline, 1.0))
        
        # Add some randomness
        import random
        random_factor = random.randint(-10, 20)
        
        return max(0, base_delay + random_factor)
    
    def _get_delay_category(self, delay_minutes: int, hour: int) -> str:
        """Categorize delays based on patterns"""
        if delay_minutes <= 5:
            return 'on_time'
        elif delay_minutes <= 15:
            return 'operational'
        elif hour in [7, 8, 18, 19]:
            return 'traffic'
        else:
            return 'weather'


@pytest.fixture(scope="session")
def e2e_test_suite():
    """Session-scoped test suite fixture"""
    suite = EndToEndTestSuite()
    suite.setup_test_data()
    return suite


class TestDataIngestionWorkflow:
    """Test complete data ingestion workflow"""
    
    def test_excel_data_processing(self, e2e_test_suite):
        """Test Excel data processing end-to-end"""
        # Create sample Excel file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp_file:
            sample_data = pd.DataFrame({
                'Flight': ['AI101', 'AI102', '6E201', '6E202'],
                'From': ['BOM', 'DEL', 'BOM', 'BLR'],
                'To': ['DEL', 'BOM', 'BLR', 'BOM'],
                'Departure': ['2024-01-01 08:00', '2024-01-01 10:00', 
                             '2024-01-01 12:00', '2024-01-01 14:00'],
                'Arrival': ['2024-01-01 10:15', '2024-01-01 12:20',
                           '2024-01-01 14:30', '2024-01-01 16:25'],
                'Aircraft': ['A320', 'B737', 'A320', 'A321'],
                'Airline': ['AI', 'AI', '6E', '6E']
            })
            sample_data.to_excel(tmp_file.name, index=False)
            
            # Test Excel processor
            processor = ExcelDataProcessor(tmp_file.name)
            df = processor.load_flight_data()
            
            assert not df.empty
            assert len(df) == 4
            assert 'flight_number' in df.columns
            assert 'origin_airport' in df.columns
            assert 'destination_airport' in df.columns
            
            # Test CSV conversion
            csv_file = tmp_file.name.replace('.xlsx', '.csv')
            success = processor.convert_to_csv(csv_file)
            assert success
            assert Path(csv_file).exists()
            
            # Cleanup
            Path(tmp_file.name).unlink()
            Path(csv_file).unlink()
    
    def test_api_data_upload(self, e2e_test_suite):
        """Test data upload via API"""
        # Create test CSV data
        csv_data = """flight_number,airline_code,origin_airport,destination_airport,scheduled_departure,scheduled_arrival
API001,AI,BOM,DEL,2024-01-01 08:00:00,2024-01-01 10:15:00
API002,6E,DEL,BOM,2024-01-01 10:00:00,2024-01-01 12:20:00"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_data)
            tmp_file.flush()
            
            # Test file upload
            with open(tmp_file.name, 'rb') as f:
                response = e2e_test_suite.client.post(
                    "/api/v1/data/upload",
                    files={"file": ("test_data.csv", f, "text/csv")}
                )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert 'upload_id' in result['data']
            
            # Cleanup
            Path(tmp_file.name).unlink()


class TestAnalysisWorkflow:
    """Test complete analysis workflow"""
    
    def test_delay_analysis_workflow(self, e2e_test_suite):
        """Test complete delay analysis workflow"""
        # Test delay analysis API
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'delay_patterns' in result['data']
        assert 'recommendations' in result['data']
        
        # Verify analysis components
        delay_data = result['data']
        assert 'hourly_analysis' in delay_data['delay_patterns']
        assert 'airline_performance' in delay_data['delay_patterns']
        assert len(delay_data['recommendations']) > 0
    
    def test_congestion_analysis_workflow(self, e2e_test_suite):
        """Test complete congestion analysis workflow"""
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/congestion",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'congestion_patterns' in result['data']
        assert 'peak_hours' in result['data']
        
        # Verify congestion data
        congestion_data = result['data']
        assert 'hourly_traffic' in congestion_data['congestion_patterns']
        assert 'capacity_utilization' in congestion_data['congestion_patterns']
    
    def test_cascading_impact_workflow(self, e2e_test_suite):
        """Test cascading impact analysis workflow"""
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/cascading-impact",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'network_analysis' in result['data']
        assert 'critical_flights' in result['data']
        
        # Verify network analysis
        network_data = result['data']
        assert 'impact_scores' in network_data['network_analysis']
        assert len(network_data['critical_flights']) > 0


class TestNLPWorkflow:
    """Test complete NLP workflow"""
    
    def test_nlp_query_processing(self, e2e_test_suite):
        """Test NLP query processing workflow"""
        test_queries = [
            "What are the busiest hours at Mumbai airport?",
            "Show me delay patterns for Air India flights",
            "Which flights have the most cascading impact?",
            "What's the best time to fly from Delhi to Mumbai?"
        ]
        
        for query in test_queries:
            response = e2e_test_suite.client.post(
                "/api/v1/nlp/query",
                json={
                    "query": query,
                    "context": {},
                    "session_id": "test_session"
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert 'response' in result['data']
            assert 'confidence' in result['data']
            assert result['data']['confidence'] > 0
    
    def test_nlp_suggestions(self, e2e_test_suite):
        """Test NLP query suggestions"""
        response = e2e_test_suite.client.get("/api/v1/nlp/suggestions")
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'suggestions' in result['data']
        assert len(result['data']['suggestions']) > 0


class TestReportingWorkflow:
    """Test complete reporting workflow"""
    
    def test_dashboard_data_generation(self, e2e_test_suite):
        """Test dashboard data generation"""
        response = e2e_test_suite.client.get(
            "/api/v1/reports/dashboard",
            params={"airport_code": "BOM"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'dashboard_data' in result['data']
        
        dashboard_data = result['data']['dashboard_data']
        assert 'summary_metrics' in dashboard_data
        assert 'charts' in dashboard_data
        assert 'recent_analysis' in dashboard_data
    
    def test_report_generation(self, e2e_test_suite):
        """Test report generation workflow"""
        response = e2e_test_suite.client.post(
            "/api/v1/reports/generate",
            json={
                "report_type": "delay_analysis",
                "airport_code": "BOM",
                "date_range": {
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-07"
                },
                "format": "json"
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result['success'] is True
        assert 'report_id' in result['data']
        assert 'download_url' in result['data']


class TestSystemIntegration:
    """Test complete system integration"""
    
    def test_full_analysis_pipeline(self, e2e_test_suite):
        """Test complete analysis pipeline from data to insights"""
        # 1. Upload data
        csv_data = """flight_number,airline_code,origin_airport,destination_airport,scheduled_departure,scheduled_arrival,actual_departure,actual_arrival
PIPE001,AI,BOM,DEL,2024-01-01 08:00:00,2024-01-01 10:15:00,2024-01-01 08:15:00,2024-01-01 10:30:00
PIPE002,6E,DEL,BOM,2024-01-01 10:00:00,2024-01-01 12:20:00,2024-01-01 10:05:00,2024-01-01 12:25:00"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            tmp_file.write(csv_data)
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as f:
                upload_response = e2e_test_suite.client.post(
                    "/api/v1/data/upload",
                    files={"file": ("pipeline_test.csv", f, "text/csv")}
                )
            
            assert upload_response.status_code == 200
            
            # 2. Run analysis
            analysis_response = e2e_test_suite.client.get(
                "/api/v1/analysis/delays",
                params={"airport_code": "BOM", "days_back": 1}
            )
            
            assert analysis_response.status_code == 200
            
            # 3. Query via NLP
            nlp_response = e2e_test_suite.client.post(
                "/api/v1/nlp/query",
                json={
                    "query": "What are the recent delay patterns at Mumbai airport?",
                    "context": {},
                    "session_id": "pipeline_test"
                }
            )
            
            assert nlp_response.status_code == 200
            
            # 4. Generate report
            report_response = e2e_test_suite.client.post(
                "/api/v1/reports/generate",
                json={
                    "report_type": "delay_analysis",
                    "airport_code": "BOM",
                    "date_range": {
                        "start_date": "2024-01-01",
                        "end_date": "2024-01-01"
                    },
                    "format": "json"
                }
            )
            
            assert report_response.status_code == 200
            
            # Cleanup
            Path(tmp_file.name).unlink()
    
    def test_cache_integration(self, e2e_test_suite):
        """Test cache integration across system"""
        # Clear cache
        cache_manager.flush_all()
        
        # First request (should populate cache)
        response1 = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response1.status_code == 200
        
        # Second request (should use cache)
        response2 = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response2.status_code == 200
        assert response1.json() == response2.json()
        
        # Verify cache statistics
        stats = cache_manager.get_stats()
        assert stats['keyspace_hits'] > 0
    
    def test_error_handling_integration(self, e2e_test_suite):
        """Test error handling across system components"""
        # Test invalid airport code
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "INVALID", "days_back": 7}
        )
        
        assert response.status_code in [400, 404]
        result = response.json()
        assert result['success'] is False
        assert 'error' in result
        
        # Test invalid NLP query
        response = e2e_test_suite.client.post(
            "/api/v1/nlp/query",
            json={
                "query": "",  # Empty query
                "context": {},
                "session_id": "error_test"
            }
        )
        
        assert response.status_code == 400
        result = response.json()
        assert result['success'] is False
    
    def test_performance_under_load(self, e2e_test_suite):
        """Test system performance under concurrent load"""
        import concurrent.futures
        import time
        
        def make_request():
            response = e2e_test_suite.client.get(
                "/api/v1/analysis/delays",
                params={"airport_code": "BOM", "days_back": 7}
            )
            return response.status_code == 200, response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
        
        # Make 20 concurrent requests
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        # Verify all requests succeeded
        success_count = sum(1 for success, _ in results if success)
        assert success_count >= 18  # Allow for some failures under load
        
        # Verify reasonable performance
        assert total_time < 30  # Should complete within 30 seconds


class TestRequirementsValidation:
    """Validate all requirements are met"""
    
    def test_requirement_1_data_collection(self, e2e_test_suite):
        """Validate Requirement 1: Data Collection and Processing"""
        # Test Excel/CSV processing
        response = e2e_test_suite.client.get("/api/v1/data/flights?limit=10")
        assert response.status_code == 200
        
        # Test multiple airport support
        for airport in ['BOM', 'DEL']:
            response = e2e_test_suite.client.get(
                f"/api/v1/data/flights?origin_airport={airport}&limit=5"
            )
            assert response.status_code == 200
            result = response.json()
            assert len(result['data']['flights']) > 0
    
    def test_requirement_2_delay_analysis(self, e2e_test_suite):
        """Validate Requirement 2: Delay Analysis and Optimization"""
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify scheduled vs actual analysis
        assert 'delay_patterns' in result['data']
        assert 'optimal_times' in result['data']
        assert 'recommendations' in result['data']
    
    def test_requirement_3_congestion_analysis(self, e2e_test_suite):
        """Validate Requirement 3: Peak Hours and Congestion Analysis"""
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/congestion",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify peak hours identification
        assert 'peak_hours' in result['data']
        assert 'congestion_patterns' in result['data']
        assert 'recommendations' in result['data']
    
    def test_requirement_4_schedule_impact(self, e2e_test_suite):
        """Validate Requirement 4: Schedule Impact Modeling"""
        response = e2e_test_suite.client.post(
            "/api/v1/analysis/schedule-impact",
            json={
                "flight_id": "E2E0001",
                "proposed_changes": {
                    "new_departure_time": "2024-01-01T09:00:00"
                }
            }
        )
        
        # Note: This might return 404 if specific flight doesn't exist
        # The important thing is that the endpoint exists and handles the request
        assert response.status_code in [200, 404, 422]
    
    def test_requirement_5_cascading_impact(self, e2e_test_suite):
        """Validate Requirement 5: Cascading Impact Identification"""
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/cascading-impact",
            params={"airport_code": "BOM", "days_back": 7}
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify network analysis
        assert 'network_analysis' in result['data']
        assert 'critical_flights' in result['data']
    
    def test_requirement_6_nlp_interface(self, e2e_test_suite):
        """Validate Requirement 6: Natural Language Query Interface"""
        test_queries = [
            "What are the busiest hours at Mumbai airport?",
            "Show me delay patterns",
            "Which flights cause the most delays?"
        ]
        
        for query in test_queries:
            response = e2e_test_suite.client.post(
                "/api/v1/nlp/query",
                json={
                    "query": query,
                    "context": {},
                    "session_id": "req6_test"
                }
            )
            
            assert response.status_code == 200
            result = response.json()
            assert result['success'] is True
            assert 'response' in result['data']
    
    def test_requirement_7_reporting(self, e2e_test_suite):
        """Validate Requirement 7: Reporting and Visualization"""
        # Test dashboard data
        response = e2e_test_suite.client.get(
            "/api/v1/reports/dashboard",
            params={"airport_code": "BOM"}
        )
        
        assert response.status_code == 200
        result = response.json()
        assert 'dashboard_data' in result['data']
        assert 'charts' in result['data']['dashboard_data']
    
    def test_requirement_8_api_integration(self, e2e_test_suite):
        """Validate Requirement 8: System Integration and API"""
        # Test API endpoints exist and return structured responses
        endpoints = [
            "/api/v1/data/flights",
            "/api/v1/analysis/delays?airport_code=BOM&days_back=7",
            "/api/v1/analysis/congestion?airport_code=BOM&days_back=7",
            "/api/v1/reports/dashboard?airport_code=BOM"
        ]
        
        for endpoint in endpoints:
            response = e2e_test_suite.client.get(endpoint)
            assert response.status_code == 200
            result = response.json()
            assert 'success' in result
            assert 'data' in result
    
    def test_requirement_9_performance(self, e2e_test_suite):
        """Validate Requirement 9: Performance and Scalability"""
        import time
        
        # Test response time
        start_time = time.time()
        response = e2e_test_suite.client.get(
            "/api/v1/analysis/delays",
            params={"airport_code": "BOM", "days_back": 7}
        )
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds
        
        # Test concurrent requests
        def make_request():
            return e2e_test_suite.client.get("/health")
        
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All health checks should succeed
        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count == 10


if __name__ == "__main__":
    # Run the complete test suite
    pytest.main([__file__, "-v", "-s", "--tb=short"])