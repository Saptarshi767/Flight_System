"""
Integration tests for data management endpoints
"""
import pytest
import io
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pandas as pd

from src.api.main import app
from src.database.connection import db_session_scope
from src.database.models import Flight, Airport, Airline

client = TestClient(app)


class TestDataEndpoints:
    """Test data management endpoints"""
    
    def test_data_root_endpoint(self):
        """Test data root endpoint returns available operations"""
        response = client.get("/api/v1/data/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "available_endpoints" in data["data"]
        assert len(data["data"]["available_endpoints"]) > 0
    
    def test_get_flights_empty_database(self):
        """Test getting flights from empty database"""
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_session.query.return_value.count.return_value = 0
            mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = []
            mock_get_db.return_value = mock_session
            
            response = client.get("/api/v1/data/flights")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["total"] == 0
            assert len(data["data"]) == 0
    
    def test_get_flights_with_pagination(self):
        """Test getting flights with pagination parameters"""
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_session.query.return_value.count.return_value = 100
            mock_session.query.return_value.offset.return_value.limit.return_value.all.return_value = []
            mock_get_db.return_value = mock_session
            
            response = client.get("/api/v1/data/flights?page=2&size=25")
            assert response.status_code == 200
            
            data = response.json()
            assert data["page"] == 2
            assert data["size"] == 25
            assert data["total"] == 100
            assert data["pages"] == 4
    
    def test_get_flights_with_filters(self):
        """Test getting flights with various filters"""
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_query = mock_session.query.return_value
            mock_query.filter.return_value = mock_query
            mock_query.count.return_value = 5
            mock_query.offset.return_value.limit.return_value.all.return_value = []
            mock_get_db.return_value = mock_session
            
            # Test with multiple filters
            response = client.get(
                "/api/v1/data/flights"
                "?airline=AI"
                "&origin_airport=BOM"
                "&destination_airport=DEL"
                "&min_delay=15"
            )
            assert response.status_code == 200
            
            # Verify filters were applied (mock was called)
            assert mock_query.filter.call_count >= 4
    
    def test_create_flight_success(self):
        """Test creating a new flight record"""
        flight_data = {
            "flight_number": "AI101",
            "airline": "AI",
            "origin_airport": "BOM",
            "destination_airport": "DEL",
            "scheduled_departure": "2024-01-15T10:00:00",
            "scheduled_arrival": "2024-01-15T12:30:00",
            "aircraft_type": "A320"
        }
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            
            # Mock successful flight creation
            mock_flight = MagicMock()
            mock_flight.flight_id = "AI101_20240115_1000"
            mock_repo.create_flight.return_value = mock_flight
            
            response = client.post("/api/v1/data/flights", json=flight_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "Flight created successfully" in data["message"]
    
    def test_create_flight_validation_error(self):
        """Test creating flight with invalid data"""
        invalid_data = {
            "flight_number": "",  # Empty flight number
            "airline": "AI"
            # Missing required fields
        }
        
        response = client.post("/api/v1/data/flights", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_flight_by_id_success(self):
        """Test getting a specific flight by ID"""
        flight_id = "AI101_20240115_1000"
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            
            # Mock flight found
            mock_flight = MagicMock()
            mock_flight.flight_id = flight_id
            mock_repo.get_flight_by_id.return_value = mock_flight
            
            response = client.get(f"/api/v1/data/flights/{flight_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
    
    def test_get_flight_by_id_not_found(self):
        """Test getting non-existent flight"""
        flight_id = "NONEXISTENT"
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            mock_repo.get_flight_by_id.return_value = None
            
            response = client.get(f"/api/v1/data/flights/{flight_id}")
            assert response.status_code == 404
    
    def test_update_flight_success(self):
        """Test updating a flight record"""
        flight_id = "AI101_20240115_1000"
        update_data = {
            "actual_departure": "2024-01-15T10:15:00",
            "passenger_count": 150
        }
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            
            # Mock successful update
            mock_flight = MagicMock()
            mock_repo.update_flight.return_value = mock_flight
            
            response = client.put(f"/api/v1/data/flights/{flight_id}", json=update_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "Flight updated successfully" in data["message"]
    
    def test_delete_flight_success(self):
        """Test deleting a flight record"""
        flight_id = "AI101_20240115_1000"
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            mock_repo.delete_flight.return_value = True
            
            response = client.delete(f"/api/v1/data/flights/{flight_id}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "Flight deleted successfully" in data["message"]
    
    def test_upload_csv_file(self):
        """Test uploading CSV flight data"""
        # Create sample CSV content
        csv_content = """Flight,Airline,From,To,Departure,Arrival,Aircraft
AI101,AI,BOM,DEL,2024-01-15 10:00:00,2024-01-15 12:30:00,A320
6E202,6E,DEL,BOM,2024-01-15 14:00:00,2024-01-15 16:30:00,A320"""
        
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            
            files = {"file": ("test_flights.csv", csv_content, "text/csv")}
            response = client.post("/api/v1/data/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test_flights.csv"
            assert data["records_processed"] == 2
    
    def test_upload_excel_file(self):
        """Test uploading Excel flight data"""
        # Create sample Excel content
        df = pd.DataFrame({
            'Flight': ['AI101', '6E202'],
            'Airline': ['AI', '6E'],
            'From': ['BOM', 'DEL'],
            'To': ['DEL', 'BOM'],
            'Departure': ['2024-01-15 10:00:00', '2024-01-15 14:00:00'],
            'Arrival': ['2024-01-15 12:30:00', '2024-01-15 16:30:00'],
            'Aircraft': ['A320', 'A320']
        })
        
        excel_buffer = io.BytesIO()
        df.to_excel(excel_buffer, index=False)
        excel_content = excel_buffer.getvalue()
        
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            
            files = {"file": ("test_flights.xlsx", excel_content, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
            response = client.post("/api/v1/data/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["filename"] == "test_flights.xlsx"
            assert data["records_processed"] == 2
    
    def test_upload_invalid_file_type(self):
        """Test uploading unsupported file type"""
        files = {"file": ("test.txt", "invalid content", "text/plain")}
        response = client.post("/api/v1/data/upload", files=files)
        
        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]
    
    def test_export_csv_format(self):
        """Test exporting flight data in CSV format"""
        with patch('src.api.routers.data.get_flights_dataframe') as mock_get_df:
            # Mock DataFrame with sample data
            mock_df = pd.DataFrame({
                'flight_id': ['AI101_20240115_1000'],
                'flight_number': ['AI101'],
                'airline_code': ['AI'],
                'origin_airport': ['BOM'],
                'destination_airport': ['DEL']
            })
            mock_get_df.return_value = mock_df
            
            response = client.get("/api/v1/data/export?format=csv")
            assert response.status_code == 200
            assert response.headers["content-type"] == "text/csv; charset=utf-8"
            assert "attachment" in response.headers["content-disposition"]
    
    def test_export_json_format(self):
        """Test exporting flight data in JSON format"""
        with patch('src.api.routers.data.get_flights_dataframe') as mock_get_df:
            mock_df = pd.DataFrame({
                'flight_id': ['AI101_20240115_1000'],
                'flight_number': ['AI101']
            })
            mock_get_df.return_value = mock_df
            
            response = client.get("/api/v1/data/export?format=json")
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json; charset=utf-8"
    
    def test_export_no_data(self):
        """Test exporting when no data matches criteria"""
        with patch('src.api.routers.data.get_flights_dataframe') as mock_get_df:
            mock_get_df.return_value = pd.DataFrame()  # Empty DataFrame
            
            response = client.get("/api/v1/data/export")
            assert response.status_code == 404
            assert "No flight data found" in response.json()["detail"]
    
    def test_get_statistics_success(self):
        """Test getting flight statistics"""
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo, \
             patch('src.api.routers.data.db_session_scope') as mock_scope:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            mock_scope.return_value.__enter__.return_value = mock_session
            
            # Mock statistics
            mock_stats = {
                'total_flights': 100,
                'delayed_flights': 25,
                'delay_percentage': 25.0,
                'avg_departure_delay_minutes': 12.5
            }
            mock_repo.get_flight_statistics.return_value = mock_stats
            
            # Mock additional queries
            mock_query = mock_session.query.return_value
            mock_query.filter.return_value = mock_query
            mock_query.with_entities.return_value.count.return_value = 10
            
            response = client.get("/api/v1/data/statistics")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert data["data"]["total_flights"] == 100
            assert data["data"]["delay_percentage"] == 25.0
    
    def test_get_statistics_with_filters(self):
        """Test getting statistics with airport and date filters"""
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo, \
             patch('src.api.routers.data.db_session_scope') as mock_scope:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            mock_scope.return_value.__enter__.return_value = mock_session
            
            mock_stats = {'total_flights': 50}
            mock_repo.get_flight_statistics.return_value = mock_stats
            
            # Mock additional queries
            mock_query = mock_session.query.return_value
            mock_query.filter.return_value = mock_query
            mock_query.with_entities.return_value.count.return_value = 5
            
            response = client.get(
                "/api/v1/data/statistics"
                "?airport_code=BOM"
                "&start_date=2024-01-01T00:00:00"
                "&end_date=2024-01-31T23:59:59"
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "analysis_period" in data["data"]
    
    def test_cors_options_request(self):
        """Test CORS OPTIONS request"""
        response = client.options("/api/v1/data/")
        assert response.status_code == 200
        assert response.json()["message"] == "OK"


class TestDataEndpointErrors:
    """Test error handling in data endpoints"""
    
    def test_database_connection_error(self):
        """Test handling of database connection errors"""
        with patch('src.api.routers.data.get_db') as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/v1/data/flights")
            assert response.status_code == 500
            assert "Failed to retrieve flights" in response.json()["detail"]
    
    def test_create_flight_database_error(self):
        """Test handling of database errors during flight creation"""
        flight_data = {
            "flight_number": "AI101",
            "airline": "AI",
            "origin_airport": "BOM",
            "destination_airport": "DEL",
            "scheduled_departure": "2024-01-15T10:00:00",
            "scheduled_arrival": "2024-01-15T12:30:00"
        }
        
        with patch('src.api.routers.data.get_db') as mock_get_db, \
             patch('src.api.routers.data.flight_repo') as mock_repo:
            
            mock_session = MagicMock()
            mock_get_db.return_value = mock_session
            mock_repo.create_flight.side_effect = Exception("Database error")
            
            response = client.post("/api/v1/data/flights", json=flight_data)
            assert response.status_code == 400
            assert "Failed to create flight" in response.json()["detail"]
    
    def test_export_processing_error(self):
        """Test handling of errors during data export"""
        with patch('src.api.routers.data.get_flights_dataframe') as mock_get_df:
            mock_get_df.side_effect = Exception("Data processing error")
            
            response = client.get("/api/v1/data/export")
            assert response.status_code == 500
            assert "Failed to export data" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__])