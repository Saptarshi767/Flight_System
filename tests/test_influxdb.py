"""
Tests for InfluxDB time series functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd

from src.database.influxdb_client import InfluxDBManager, InfluxDBConfig
from src.database.time_series_models import (
    FlightDelayMetric, CongestionMetric, RunwayMetric, WeatherMetric,
    PerformanceMetric, TimeSeriesDataBuilder, DelayCategory, MeasurementType
)
from src.database.time_series_ingestion import TimeSeriesIngestionPipeline
from src.database.time_series_utils import TimeSeriesQueryUtils, AggregationConfig
from src.database.setup_influxdb import InfluxDBSetup


class TestInfluxDBManager:
    """Test InfluxDB manager functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Mock InfluxDB configuration."""
        return InfluxDBConfig(
            url="http://localhost:8086",
            token="test-token",
            org="test-org",
            bucket="test-bucket"
        )
    
    @pytest.fixture
    def mock_influxdb_client(self):
        """Mock InfluxDB client."""
        with patch('src.database.influxdb_client.InfluxDBClient') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            mock_instance.ping.return_value = True
            mock_instance.write_api.return_value = Mock()
            mock_instance.query_api.return_value = Mock()
            mock_instance.delete_api.return_value = Mock()
            yield mock_instance
    
    def test_influxdb_manager_initialization(self, mock_config, mock_influxdb_client):
        """Test InfluxDB manager initialization."""
        manager = InfluxDBManager(mock_config)
        
        assert manager.config == mock_config
        assert manager.client is not None
        mock_influxdb_client.ping.assert_called_once()
    
    def test_health_check(self, mock_config, mock_influxdb_client):
        """Test health check functionality."""
        manager = InfluxDBManager(mock_config)
        
        # Test successful health check
        mock_influxdb_client.ping.return_value = True
        assert manager.health_check() is True
        
        # Test failed health check
        mock_influxdb_client.ping.side_effect = Exception("Connection failed")
        assert manager.health_check() is False
    
    def test_write_flight_metrics(self, mock_config, mock_influxdb_client):
        """Test writing flight metrics."""
        manager = InfluxDBManager(mock_config)
        
        metrics = [
            {
                "measurement": "test_metrics",
                "tags": {"airport": "BOM"},
                "fields": {"delay": 15.0},
                "time": datetime.now()
            }
        ]
        
        # Mock successful write
        mock_write_api = Mock()
        manager.write_api = mock_write_api
        
        result = manager.write_flight_metrics(metrics)
        
        assert result is True
        mock_write_api.write.assert_called_once()
    
    def test_query_flight_metrics(self, mock_config, mock_influxdb_client):
        """Test querying flight metrics."""
        manager = InfluxDBManager(mock_config)
        
        # Mock query result
        mock_record = Mock()
        mock_record.get_time.return_value = datetime.now()
        mock_record.get_measurement.return_value = "test_metrics"
        mock_record.get_field.return_value = "delay"
        mock_record.get_value.return_value = 15.0
        mock_record.values = {"airport": "BOM"}
        
        mock_table = Mock()
        mock_table.records = [mock_record]
        
        mock_query_api = Mock()
        mock_query_api.query.return_value = [mock_table]
        manager.query_api = mock_query_api
        
        result = manager.query_flight_metrics(
            measurement="test_metrics",
            start_time=datetime.now() - timedelta(hours=1),
            end_time=datetime.now()
        )
        
        assert len(result) == 1
        assert result[0]["measurement"] == "test_metrics"
        assert result[0]["field"] == "delay"
        assert result[0]["value"] == 15.0


class TestTimeSeriesModels:
    """Test time series data models."""
    
    def test_flight_delay_metric_creation(self):
        """Test FlightDelayMetric creation."""
        scheduled_time = datetime.now()
        actual_time = scheduled_time + timedelta(minutes=15)
        
        metric = FlightDelayMetric(
            flight_id="AI101",
            airline="AI",
            aircraft_type="A320",
            origin_airport="BOM",
            destination_airport="DEL",
            runway="RW01",
            delay_minutes=15.0,
            delay_category=DelayCategory.TRAFFIC,
            scheduled_time=scheduled_time,
            actual_time=actual_time
        )
        
        assert metric.measurement == MeasurementType.DELAY_METRICS.value
        assert metric.tags["flight_id"] == "AI101"
        assert metric.tags["airline"] == "AI"
        assert metric.tags["delay_category"] == "traffic"
        assert metric.fields["delay_minutes"] == 15.0
        
        # Test conversion to InfluxDB format
        influx_dict = metric.to_influx_dict()
        assert "measurement" in influx_dict
        assert "tags" in influx_dict
        assert "fields" in influx_dict
        assert "time" in influx_dict
    
    def test_congestion_metric_creation(self):
        """Test CongestionMetric creation."""
        timestamp = datetime.now()
        
        metric = CongestionMetric(
            airport_code="BOM",
            runway="RW01",
            flight_count=10,
            congestion_score=75.0,
            capacity_utilization=85.0,
            average_delay=12.5,
            timestamp=timestamp,
            peak_hour=True
        )
        
        assert metric.measurement == MeasurementType.CONGESTION_METRICS.value
        assert metric.tags["airport_code"] == "BOM"
        assert metric.tags["peak_hour"] == "true"
        assert metric.fields["flight_count"] == 10
        assert metric.fields["congestion_score"] == 75.0
    
    def test_time_series_data_builder(self):
        """Test TimeSeriesDataBuilder functionality."""
        flight_data = {
            "flight_id": "AI101",
            "airline": "AI",
            "aircraft_type": "A320",
            "origin_airport": "BOM",
            "destination_airport": "DEL",
            "runway": "RW01",
            "scheduled_departure": datetime.now(),
            "actual_departure": datetime.now() + timedelta(minutes=15)
        }
        
        metric = TimeSeriesDataBuilder.create_flight_delay_metric(
            flight_data=flight_data,
            delay_minutes=15.0,
            delay_category="traffic"
        )
        
        assert isinstance(metric, FlightDelayMetric)
        assert metric.tags["flight_id"] == "AI101"
        assert metric.fields["delay_minutes"] == 15.0


class TestTimeSeriesIngestion:
    """Test time series ingestion pipeline."""
    
    @pytest.fixture
    def mock_influxdb_manager(self):
        """Mock InfluxDB manager for testing."""
        manager = Mock()
        manager.write_flight_metrics.return_value = True
        return manager
    
    @pytest.fixture
    def ingestion_pipeline(self, mock_influxdb_manager):
        """Create ingestion pipeline with mocked manager."""
        return TimeSeriesIngestionPipeline(mock_influxdb_manager)
    
    @pytest.mark.asyncio
    async def test_ingest_flight_data_batch(self, ingestion_pipeline):
        """Test batch ingestion of flight data."""
        flight_data = [
            {
                "flight_id": "AI101",
                "airline": "AI",
                "origin_airport": "BOM",
                "destination_airport": "DEL",
                "scheduled_departure": datetime.now(),
                "actual_departure": datetime.now() + timedelta(minutes=15),
                "delay_minutes": 15.0
            },
            {
                "flight_id": "6E202",
                "airline": "6E",
                "origin_airport": "DEL",
                "destination_airport": "BOM",
                "scheduled_departure": datetime.now(),
                "actual_departure": datetime.now() + timedelta(minutes=5),
                "delay_minutes": 5.0
            }
        ]
        
        result = await ingestion_pipeline.ingest_flight_data_batch(
            flight_data=flight_data,
            calculate_delays=True,
            include_congestion=False
        )
        
        assert result is True
        assert ingestion_pipeline.stats["total_processed"] == 2
        assert ingestion_pipeline.stats["successful_writes"] == 1
    
    @pytest.mark.asyncio
    async def test_ingest_real_time_flight(self, ingestion_pipeline):
        """Test real-time flight ingestion."""
        flight_data = {
            "flight_id": "AI101",
            "airline": "AI",
            "origin_airport": "BOM",
            "destination_airport": "DEL",
            "scheduled_departure": datetime.now(),
            "actual_departure": datetime.now() + timedelta(minutes=10),
            "delay_minutes": 10.0
        }
        
        result = await ingestion_pipeline.ingest_real_time_flight(flight_data)
        
        assert result is True
    
    def test_delay_calculation(self, ingestion_pipeline):
        """Test delay calculation logic."""
        # Test with actual and scheduled times
        flight_data = {
            "scheduled_departure": "2024-01-01 10:00:00",
            "actual_departure": "2024-01-01 10:15:00"
        }
        
        delay = ingestion_pipeline._calculate_delay(flight_data)
        assert delay == 15.0
        
        # Test with pre-calculated delay
        flight_data_with_delay = {
            "delay_minutes": 20.0
        }
        
        delay = ingestion_pipeline._calculate_delay(flight_data_with_delay)
        assert delay == 20.0
        
        # Test with insufficient data
        insufficient_data = {
            "flight_id": "AI101"
        }
        
        delay = ingestion_pipeline._calculate_delay(insufficient_data)
        assert delay is None
    
    def test_delay_categorization(self, ingestion_pipeline):
        """Test delay categorization logic."""
        # Test weather-related delay
        flight_data = {
            "weather_condition": "heavy rain"
        }
        
        category = ingestion_pipeline._categorize_delay(flight_data, 30.0)
        assert category == DelayCategory.WEATHER
        
        # Test traffic delay during peak hours
        flight_data = {
            "scheduled_departure": datetime.now().replace(hour=8)  # Peak hour
        }
        
        category = ingestion_pipeline._categorize_delay(flight_data, 20.0)
        assert category == DelayCategory.TRAFFIC
        
        # Test operational delay for long delays
        category = ingestion_pipeline._categorize_delay({}, 90.0)
        assert category == DelayCategory.OPERATIONAL


class TestTimeSeriesUtils:
    """Test time series utility functions."""
    
    @pytest.fixture
    def mock_influxdb_manager(self):
        """Mock InfluxDB manager for testing."""
        manager = Mock()
        manager.query_flight_metrics.return_value = [
            {
                "time": datetime.now(),
                "measurement": "delay_metrics",
                "field": "delay_minutes",
                "value": 15.0,
                "airport": "BOM"
            }
        ]
        return manager
    
    @pytest.fixture
    def query_utils(self, mock_influxdb_manager):
        """Create query utils with mocked manager."""
        return TimeSeriesQueryUtils(mock_influxdb_manager)
    
    @pytest.mark.asyncio
    async def test_get_delay_statistics(self, query_utils):
        """Test delay statistics query."""
        result = await query_utils.get_delay_statistics(
            airport_code="BOM",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
        
        assert isinstance(result.data, list)
        assert result.record_count >= 0
        assert result.execution_time >= 0
    
    @pytest.mark.asyncio
    async def test_get_congestion_trends(self, query_utils):
        """Test congestion trends query."""
        result = await query_utils.get_congestion_trends(
            airport_code="BOM",
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            aggregation=AggregationConfig(window="1h", function="mean")
        )
        
        assert isinstance(result.data, list)
        assert result.query.startswith("congestion_trends")
    
    def test_time_series_aggregator_moving_average(self):
        """Test moving average calculation."""
        from src.database.time_series_utils import TimeSeriesAggregator
        
        data = [
            {"time": datetime.now(), "value": 10.0},
            {"time": datetime.now(), "value": 15.0},
            {"time": datetime.now(), "value": 20.0},
            {"time": datetime.now(), "value": 25.0},
            {"time": datetime.now(), "value": 30.0}
        ]
        
        result = TimeSeriesAggregator.calculate_moving_average(
            data=data,
            window_size=3,
            value_field="value"
        )
        
        assert len(result) == len(data)
        assert "moving_average" in result[0]
    
    def test_time_series_aggregator_anomaly_detection(self):
        """Test anomaly detection."""
        from src.database.time_series_utils import TimeSeriesAggregator
        
        data = [
            {"time": datetime.now(), "value": 10.0},
            {"time": datetime.now(), "value": 12.0},
            {"time": datetime.now(), "value": 11.0},
            {"time": datetime.now(), "value": 100.0},  # Anomaly
            {"time": datetime.now(), "value": 13.0}
        ]
        
        result = TimeSeriesAggregator.detect_anomalies(
            data=data,
            value_field="value",
            threshold_std=2.0
        )
        
        assert len(result) == len(data)
        assert "is_anomaly" in result[0]
        assert "anomaly_score" in result[0]
        
        # Check that the anomaly is detected
        anomalies = [r for r in result if r["is_anomaly"]]
        assert len(anomalies) > 0


class TestInfluxDBSetup:
    """Test InfluxDB setup functionality."""
    
    @pytest.fixture
    def mock_influxdb_manager(self):
        """Mock InfluxDB manager for setup testing."""
        manager = Mock()
        manager.health_check.return_value = True
        manager.setup_retention_policies.return_value = True
        manager.get_bucket_info.return_value = {
            "id": "test-bucket-id",
            "name": "test-bucket",
            "retention_rules": [{"type": "expire", "every_seconds": 7776000}]
        }
        manager.write_flight_metrics.return_value = True
        manager.query_flight_metrics.return_value = []
        return manager
    
    @pytest.mark.asyncio
    async def test_influxdb_setup_initialization(self, mock_influxdb_manager):
        """Test InfluxDB setup initialization."""
        with patch('src.database.setup_influxdb.InfluxDBManager') as mock_manager_class:
            mock_manager_class.return_value = mock_influxdb_manager
            
            setup = InfluxDBSetup()
            result = await setup.initialize_database()
            
            assert result is True
            mock_influxdb_manager.health_check.assert_called()
            mock_influxdb_manager.setup_retention_policies.assert_called()
    
    @pytest.mark.asyncio
    async def test_setup_verification(self, mock_influxdb_manager):
        """Test setup verification."""
        with patch('src.database.setup_influxdb.InfluxDBManager') as mock_manager_class:
            mock_manager_class.return_value = mock_influxdb_manager
            
            setup = InfluxDBSetup()
            verification = await setup.verify_setup()
            
            assert verification["connection_healthy"] is True
            assert verification["bucket_exists"] is True
            assert verification["retention_policies"] is True
            assert isinstance(verification["sample_data_count"], int)
            assert isinstance(verification["measurements"], list)


if __name__ == "__main__":
    pytest.main([__file__])