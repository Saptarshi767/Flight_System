"""
Tests for database functionality
"""
import pytest
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.database.models import Flight, Airport, Airline, Aircraft, AnalysisResult, DataIngestionLog
from src.database.operations import (
    flight_repo, airport_repo, airline_repo, aircraft_repo, 
    analysis_result_repo, ingestion_log_repo
)
from src.database.utils import db_utils
from src.database.connection import db_manager


class TestDatabaseModels:
    """Test database models"""
    
    def test_airport_model_creation(self):
        """Test Airport model creation"""
        airport_data = {
            'code': 'BOM',
            'name': 'Chhatrapati Shivaji Maharaj International Airport',
            'city': 'Mumbai',
            'country': 'India',
            'timezone': 'Asia/Kolkata',
            'runway_count': 2,
            'runway_capacity': 45,
            'latitude': 19.0896,
            'longitude': 72.8656
        }
        
        airport = Airport(**airport_data)
        assert airport.code == 'BOM'
        assert airport.name == 'Chhatrapati Shivaji Maharaj International Airport'
        assert airport.runway_count == 2
        assert airport.runway_capacity == 45
    
    def test_airline_model_creation(self):
        """Test Airline model creation"""
        airline_data = {
            'code': 'AI',
            'name': 'Air India',
            'country': 'India',
            'is_active': True
        }
        
        airline = Airline(**airline_data)
        assert airline.code == 'AI'
        assert airline.name == 'Air India'
        assert airline.is_active is True
    
    def test_aircraft_model_creation(self):
        """Test Aircraft model creation"""
        aircraft_data = {
            'type_code': 'A320',
            'manufacturer': 'Airbus',
            'model': 'A320',
            'typical_seating': 150,
            'max_seating': 180,
            'max_range': 3300,
            'cruise_speed': 450
        }
        
        aircraft = Aircraft(**aircraft_data)
        assert aircraft.type_code == 'A320'
        assert aircraft.manufacturer == 'Airbus'
        assert aircraft.typical_seating == 150
    
    def test_flight_model_creation(self):
        """Test Flight model creation"""
        flight_data = {
            'flight_id': 'AI101-20240101-BOM-DEL',
            'flight_number': 'AI101',
            'airline_code': 'AI',
            'origin_airport': 'BOM',
            'destination_airport': 'DEL',
            'aircraft_type': 'A320',
            'scheduled_departure': datetime(2024, 1, 1, 10, 0),
            'scheduled_arrival': datetime(2024, 1, 1, 12, 30),
            'actual_departure': datetime(2024, 1, 1, 10, 15),
            'actual_arrival': datetime(2024, 1, 1, 12, 45),
            'departure_delay_minutes': 15,
            'arrival_delay_minutes': 15,
            'delay_category': 'operational',
            'passenger_count': 150,
            'data_source': 'excel',
            'data_quality_score': 0.95
        }
        
        flight = Flight(**flight_data)
        assert flight.flight_id == 'AI101-20240101-BOM-DEL'
        assert flight.flight_number == 'AI101'
        assert flight.departure_delay_minutes == 15
        assert flight.data_quality_score == 0.95


class TestDatabaseOperations:
    """Test database operations"""
    
    @patch('src.database.operations.db_session_scope')
    def test_flight_repository_create(self, mock_session_scope):
        """Test flight repository create operation"""
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        flight_data = {
            'flight_id': 'AI101-20240101-BOM-DEL',
            'flight_number': 'AI101',
            'airline_code': 'AI',
            'origin_airport': 'BOM',
            'destination_airport': 'DEL',
            'scheduled_departure': datetime(2024, 1, 1, 10, 0),
            'scheduled_arrival': datetime(2024, 1, 1, 12, 30),
            'data_source': 'test'
        }
        
        # Mock the flight creation
        mock_flight = Flight(**flight_data)
        mock_session.add.return_value = None
        mock_session.flush.return_value = None
        
        # Test the repository method
        with patch.object(Flight, '__init__', return_value=None):
            with patch.object(flight_repo, 'logger'):
                result = flight_repo.create_flight(mock_session, flight_data)
                mock_session.add.assert_called_once()
                mock_session.flush.assert_called_once()
    
    @patch('src.database.operations.db_session_scope')
    def test_airport_repository_create(self, mock_session_scope):
        """Test airport repository create operation"""
        mock_session = MagicMock()
        mock_session_scope.return_value.__enter__.return_value = mock_session
        
        airport_data = {
            'code': 'BOM',
            'name': 'Mumbai Airport',
            'city': 'Mumbai',
            'country': 'India',
            'timezone': 'Asia/Kolkata',
            'runway_count': 2,
            'runway_capacity': 45
        }
        
        # Test the repository method
        with patch.object(Airport, '__init__', return_value=None):
            with patch.object(airport_repo, 'logger'):
                result = airport_repo.create_airport(mock_session, airport_data)
                mock_session.add.assert_called_once()
                mock_session.flush.assert_called_once()


class TestDatabaseUtils:
    """Test database utilities"""
    
    @patch('src.database.utils.db_manager')
    def test_database_utils_initialization(self, mock_db_manager):
        """Test database utils initialization"""
        mock_db_manager.create_tables.return_value = None
        
        result = db_utils.initialize_database(create_sample_data=False)
        mock_db_manager.create_tables.assert_called_once()
    
    @patch('src.database.utils.db_manager')
    def test_database_health_check(self, mock_db_manager):
        """Test database health check"""
        mock_db_manager.test_connection.return_value = True
        
        with patch('src.database.utils.db_session_scope') as mock_session_scope:
            mock_session = MagicMock()
            mock_session_scope.return_value.__enter__.return_value = mock_session
            
            # Mock inspector
            with patch('src.database.utils.inspect') as mock_inspect:
                mock_inspector = MagicMock()
                mock_inspector.get_table_names.return_value = ['flights', 'airports']
                mock_inspect.return_value = mock_inspector
                
                # Mock query results
                mock_result = MagicMock()
                mock_result.scalar.return_value = 100
                mock_session.execute.return_value = mock_result
                
                health_info = db_utils.check_database_health()
                
                assert 'connection_test' in health_info
                assert 'table_integrity' in health_info
                assert 'timestamp' in health_info
    
    def test_database_connection_info(self):
        """Test getting database connection info"""
        with patch.object(db_manager, 'get_connection_info') as mock_get_info:
            mock_get_info.return_value = {
                'host': 'localhost',
                'port': 5432,
                'database': 'flightdb',
                'username': 'user'
            }
            
            info = db_manager.get_connection_info()
            assert info['host'] == 'localhost'
            assert info['port'] == 5432
            assert info['database'] == 'flightdb'


class TestDatabaseConnection:
    """Test database connection management"""
    
    @patch('src.database.connection.create_engine')
    def test_database_manager_initialization(self, mock_create_engine):
        """Test DatabaseManager initialization"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        with patch('src.database.connection.sessionmaker') as mock_sessionmaker:
            with patch('src.database.connection.DatabaseManager._validate_database_url'):
                from src.database.connection import DatabaseManager
                
                db_mgr = DatabaseManager("postgresql://user:pass@localhost/test")
                
                mock_create_engine.assert_called_once()
                mock_sessionmaker.assert_called_once()
    
    def test_database_url_validation(self):
        """Test database URL validation"""
        from src.database.connection import DatabaseManager
        
        # Test valid URL
        valid_url = "postgresql://user:pass@localhost:5432/testdb"
        with patch.object(DatabaseManager, '_initialize_engine'):
            with patch.object(DatabaseManager, '_initialize_session_factory'):
                db_mgr = DatabaseManager(valid_url)
                # Should not raise exception
        
        # Test invalid URL
        invalid_url = "invalid://url"
        with pytest.raises(ValueError):
            with patch.object(DatabaseManager, '_initialize_engine'):
                with patch.object(DatabaseManager, '_initialize_session_factory'):
                    db_mgr = DatabaseManager(invalid_url)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])