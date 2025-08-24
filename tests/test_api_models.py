"""
Tests for Pydantic models used in the API.

This module contains tests for request/response models, validation,
and data serialization/deserialization.
"""

import pytest
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from pydantic import ValidationError

from src.api.models import (
    # Base models
    BaseResponse, ErrorResponse, PaginationParams, PaginatedResponse,
    # Flight data models
    FlightDataBase, FlightDataCreate, FlightDataUpdate, FlightData,
    FlightDataResponse, FlightDataListResponse, FlightDataFilter,
    # Analysis models
    DelayAnalysisRequest, DelayAnalysisResult,
    CongestionAnalysisRequest, CongestionAnalysisResult,
    ScheduleImpactRequest, ScheduleImpactResult,
    CascadingImpactRequest, CascadingImpactResult,
    # NLP models
    NLPQueryRequest, NLPQueryResponse, QuerySuggestion, QuerySuggestionsResponse,
    QueryFeedback,
    # Export models
    ExportRequest, ExportResponse,
    # Health models
    HealthStatus, ComponentHealth, HealthCheckResponse,
    # Enums
    DelayCategory, AnalysisType, ResponseFormat
)


class TestBaseModels:
    """Test cases for base models."""
    
    def test_base_response_creation(self):
        """Test BaseResponse model creation."""
        response = BaseResponse()
        assert response.success is True
        assert response.message == "Operation completed successfully"
        assert isinstance(response.timestamp, datetime)
        assert response.correlation_id is None
        
        # Test with custom values
        custom_response = BaseResponse(
            success=False,
            message="Custom message",
            correlation_id="test-id"
        )
        assert custom_response.success is False
        assert custom_response.message == "Custom message"
        assert custom_response.correlation_id == "test-id"
    
    def test_error_response_creation(self):
        """Test ErrorResponse model creation."""
        error = ErrorResponse(
            error="Test error",
            status_code=400
        )
        assert error.success is False
        assert error.error == "Test error"
        assert error.status_code == 400
        assert isinstance(error.timestamp, datetime)
    
    def test_pagination_params_validation(self):
        """Test PaginationParams validation."""
        # Valid parameters
        params = PaginationParams(page=2, size=25)
        assert params.page == 2
        assert params.size == 25
        assert params.offset == 25  # (2-1) * 25
        
        # Test defaults
        default_params = PaginationParams()
        assert default_params.page == 1
        assert default_params.size == 50
        assert default_params.offset == 0
        
        # Test validation errors
        with pytest.raises(ValidationError):
            PaginationParams(page=0)  # page must be >= 1
        
        with pytest.raises(ValidationError):
            PaginationParams(size=0)  # size must be >= 1
        
        with pytest.raises(ValidationError):
            PaginationParams(size=1001)  # size must be <= 1000
    
    def test_paginated_response_creation(self):
        """Test PaginatedResponse model creation."""
        response = PaginatedResponse(
            page=2,
            size=25,
            total=100,
            pages=4,
            has_next=True,
            has_prev=True
        )
        assert response.page == 2
        assert response.size == 25
        assert response.total == 100
        assert response.pages == 4
        assert response.has_next is True
        assert response.has_prev is True


class TestFlightDataModels:
    """Test cases for flight data models."""
    
    def test_flight_data_base_creation(self):
        """Test FlightDataBase model creation."""
        now = datetime.utcnow()
        flight_data = FlightDataBase(
            flight_number="AI101",
            airline="Air India",
            aircraft_type="A320",
            origin_airport="BOM",
            destination_airport="DEL",
            scheduled_departure=now,
            scheduled_arrival=now + timedelta(hours=2),
            passenger_count=150
        )
        
        assert flight_data.flight_number == "AI101"
        assert flight_data.airline == "Air India"
        assert flight_data.aircraft_type == "A320"
        assert flight_data.origin_airport == "BOM"
        assert flight_data.destination_airport == "DEL"
        assert flight_data.passenger_count == 150
    
    def test_flight_data_base_validation(self):
        """Test FlightDataBase validation."""
        now = datetime.utcnow()
        
        # Test required fields
        with pytest.raises(ValidationError):
            FlightDataBase()  # Missing required fields
        
        # Test passenger count validation
        with pytest.raises(ValidationError):
            FlightDataBase(
                flight_number="AI101",
                airline="Air India",
                origin_airport="BOM",
                destination_airport="DEL",
                scheduled_departure=now,
                scheduled_arrival=now + timedelta(hours=2),
                passenger_count=-1  # Must be >= 0
            )
    
    def test_flight_data_create_model(self):
        """Test FlightDataCreate model."""
        now = datetime.utcnow()
        flight_create = FlightDataCreate(
            flight_number="AI101",
            airline="Air India",
            origin_airport="BOM",
            destination_airport="DEL",
            scheduled_departure=now,
            scheduled_arrival=now + timedelta(hours=2)
        )
        
        assert isinstance(flight_create, FlightDataBase)
        assert flight_create.flight_number == "AI101"
    
    def test_flight_data_update_model(self):
        """Test FlightDataUpdate model."""
        now = datetime.utcnow()
        flight_update = FlightDataUpdate(
            actual_departure=now + timedelta(minutes=15),
            actual_arrival=now + timedelta(hours=2, minutes=20),
            passenger_count=145,
            runway_used="09L"
        )
        
        assert flight_update.actual_departure == now + timedelta(minutes=15)
        assert flight_update.passenger_count == 145
        assert flight_update.runway_used == "09L"
        
        # Test all fields are optional
        empty_update = FlightDataUpdate()
        assert empty_update.actual_departure is None
    
    def test_flight_data_delay_calculation(self):
        """Test FlightData delay calculation."""
        now = datetime.utcnow()
        flight_data = FlightData(
            id=uuid4(),
            flight_number="AI101",
            airline="Air India",
            origin_airport="BOM",
            destination_airport="DEL",
            scheduled_departure=now,
            scheduled_arrival=now + timedelta(hours=2),
            actual_departure=now + timedelta(minutes=15),  # 15 minutes late
            created_at=now,
            updated_at=now
        )
        
        # The validator should calculate delay_minutes
        assert flight_data.delay_minutes == 15
    
    def test_flight_data_filter_model(self):
        """Test FlightDataFilter model."""
        now = datetime.utcnow()
        filter_params = FlightDataFilter(
            airline="Air India",
            origin_airport="BOM",
            destination_airport="DEL",
            date_from=now - timedelta(days=7),
            date_to=now,
            delay_category=DelayCategory.WEATHER,
            min_delay=10,
            max_delay=60
        )
        
        assert filter_params.airline == "Air India"
        assert filter_params.delay_category == DelayCategory.WEATHER
        assert filter_params.min_delay == 10
        assert filter_params.max_delay == 60


class TestAnalysisModels:
    """Test cases for analysis models."""
    
    def test_delay_analysis_request(self):
        """Test DelayAnalysisRequest model."""
        now = datetime.utcnow()
        request = DelayAnalysisRequest(
            airport_code="BOM",
            date_from=now - timedelta(days=7),
            date_to=now,
            include_weather=True,
            granularity="hourly"
        )
        
        assert request.airport_code == "BOM"
        assert request.include_weather is True
        assert request.granularity == "hourly"
    
    def test_delay_analysis_result(self):
        """Test DelayAnalysisResult model."""
        now = datetime.utcnow()
        result = DelayAnalysisResult(
            airport_code="BOM",
            analysis_period={"from": now - timedelta(days=7), "to": now},
            average_delay=15.5,
            median_delay=12.0,
            delay_by_hour={9: 20.0, 10: 15.0, 11: 10.0},
            delay_by_category={
                DelayCategory.WEATHER: 25.0,
                DelayCategory.OPERATIONAL: 10.0
            },
            optimal_time_slots=[
                {"hour": 6, "average_delay": 5.0},
                {"hour": 22, "average_delay": 7.0}
            ],
            recommendations=["Avoid scheduling during 9-11 AM"]
        )
        
        assert result.airport_code == "BOM"
        assert result.average_delay == 15.5
        assert len(result.delay_by_hour) == 3
        assert len(result.recommendations) == 1
    
    def test_congestion_analysis_request(self):
        """Test CongestionAnalysisRequest model."""
        now = datetime.utcnow()
        request = CongestionAnalysisRequest(
            airport_code="DEL",
            date_from=now - timedelta(days=30),
            date_to=now,
            runway_capacity=60
        )
        
        assert request.airport_code == "DEL"
        assert request.runway_capacity == 60
    
    def test_schedule_impact_request(self):
        """Test ScheduleImpactRequest model."""
        now = datetime.utcnow()
        flight_id = uuid4()
        request = ScheduleImpactRequest(
            flight_id=flight_id,
            proposed_departure=now + timedelta(hours=1),
            proposed_arrival=now + timedelta(hours=3),
            impact_radius=5
        )
        
        assert request.flight_id == flight_id
        assert request.impact_radius == 5
    
    def test_cascading_impact_request(self):
        """Test CascadingImpactRequest model."""
        now = datetime.utcnow()
        request = CascadingImpactRequest(
            airport_code="BOM",
            date_from=now - timedelta(days=7),
            date_to=now,
            network_depth=3
        )
        
        assert request.airport_code == "BOM"
        assert request.network_depth == 3


class TestNLPModels:
    """Test cases for NLP models."""
    
    def test_nlp_query_request(self):
        """Test NLPQueryRequest model."""
        request = NLPQueryRequest(
            query="What's the best time to fly from Mumbai to Delhi?",
            context={"user_id": "123", "preferences": "morning"},
            session_id="session-123"
        )
        
        assert request.query == "What's the best time to fly from Mumbai to Delhi?"
        assert request.context["user_id"] == "123"
        assert request.session_id == "session-123"
        
        # Test query length validation
        with pytest.raises(ValidationError):
            NLPQueryRequest(query="")  # Too short
        
        with pytest.raises(ValidationError):
            NLPQueryRequest(query="x" * 1001)  # Too long
    
    def test_nlp_query_response(self):
        """Test NLPQueryResponse model."""
        response = NLPQueryResponse(
            query="What's the best time to fly?",
            response="The best time to fly is early morning (6-8 AM).",
            confidence=0.85,
            data={"optimal_hours": [6, 7, 8]},
            visualizations=[{"type": "bar_chart", "data": "delay_by_hour"}],
            follow_up_questions=["What about weather impact?"],
            session_id="session-123"
        )
        
        assert response.confidence == 0.85
        assert len(response.follow_up_questions) == 1
        assert response.data["optimal_hours"] == [6, 7, 8]
        
        # Test confidence validation
        with pytest.raises(ValidationError):
            NLPQueryResponse(
                query="test",
                response="test",
                confidence=1.5  # Must be <= 1.0
            )
    
    def test_query_suggestion(self):
        """Test QuerySuggestion model."""
        suggestion = QuerySuggestion(
            text="Show me delay patterns for Mumbai airport",
            category="delay_analysis",
            confidence=0.9
        )
        
        assert suggestion.text == "Show me delay patterns for Mumbai airport"
        assert suggestion.category == "delay_analysis"
        assert suggestion.confidence == 0.9
    
    def test_query_feedback(self):
        """Test QueryFeedback model."""
        feedback = QueryFeedback(
            query_id="query-123",
            rating=4,
            feedback="Very helpful response"
        )
        
        assert feedback.query_id == "query-123"
        assert feedback.rating == 4
        assert feedback.feedback == "Very helpful response"
        
        # Test rating validation
        with pytest.raises(ValidationError):
            QueryFeedback(query_id="test", rating=0)  # Must be >= 1
        
        with pytest.raises(ValidationError):
            QueryFeedback(query_id="test", rating=6)  # Must be <= 5


class TestExportModels:
    """Test cases for export models."""
    
    def test_export_request(self):
        """Test ExportRequest model."""
        request = ExportRequest(
            format=ResponseFormat.CSV,
            filters=FlightDataFilter(airline="Air India"),
            include_analysis=True
        )
        
        assert request.format == ResponseFormat.CSV
        assert request.filters.airline == "Air India"
        assert request.include_analysis is True
    
    def test_export_response(self):
        """Test ExportResponse model."""
        now = datetime.utcnow()
        response = ExportResponse(
            download_url="https://api.example.com/downloads/file.csv",
            filename="flight_data_export.csv",
            format=ResponseFormat.CSV,
            expires_at=now + timedelta(hours=24)
        )
        
        assert response.download_url.startswith("https://")
        assert response.filename.endswith(".csv")
        assert response.format == ResponseFormat.CSV


class TestHealthModels:
    """Test cases for health check models."""
    
    def test_component_health(self):
        """Test ComponentHealth model."""
        component = ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            response_time=0.05
        )
        
        assert component.name == "database"
        assert component.status == HealthStatus.HEALTHY
        assert component.response_time == 0.05
        assert component.error is None
        
        # Test with error
        error_component = ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            error="Connection timeout"
        )
        
        assert error_component.status == HealthStatus.UNHEALTHY
        assert error_component.error == "Connection timeout"
    
    def test_health_check_response(self):
        """Test HealthCheckResponse model."""
        components = [
            ComponentHealth(name="db", status=HealthStatus.HEALTHY),
            ComponentHealth(name="redis", status=HealthStatus.HEALTHY)
        ]
        
        response = HealthCheckResponse(
            status=HealthStatus.HEALTHY,
            version="1.0.0",
            components=components,
            uptime=3600.0
        )
        
        assert response.status == HealthStatus.HEALTHY
        assert response.version == "1.0.0"
        assert len(response.components) == 2
        assert response.uptime == 3600.0
        assert isinstance(response.timestamp, datetime)


class TestEnums:
    """Test cases for enum models."""
    
    def test_delay_category_enum(self):
        """Test DelayCategory enum."""
        assert DelayCategory.WEATHER == "weather"
        assert DelayCategory.OPERATIONAL == "operational"
        assert DelayCategory.TRAFFIC == "traffic"
        assert DelayCategory.MECHANICAL == "mechanical"
        assert DelayCategory.SECURITY == "security"
        assert DelayCategory.OTHER == "other"
    
    def test_analysis_type_enum(self):
        """Test AnalysisType enum."""
        assert AnalysisType.DELAY == "delay"
        assert AnalysisType.CONGESTION == "congestion"
        assert AnalysisType.SCHEDULE_IMPACT == "schedule_impact"
        assert AnalysisType.CASCADING_IMPACT == "cascading_impact"
    
    def test_response_format_enum(self):
        """Test ResponseFormat enum."""
        assert ResponseFormat.JSON == "json"
        assert ResponseFormat.CSV == "csv"
        assert ResponseFormat.EXCEL == "excel"
        assert ResponseFormat.PDF == "pdf"
    
    def test_health_status_enum(self):
        """Test HealthStatus enum."""
        assert HealthStatus.HEALTHY == "healthy"
        assert HealthStatus.DEGRADED == "degraded"
        assert HealthStatus.UNHEALTHY == "unhealthy"