"""
Pydantic models for request/response validation.

This module contains all the data models used for API request and response validation.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from uuid import UUID


class DelayCategory(str, Enum):
    """Enumeration for delay categories."""
    WEATHER = "weather"
    OPERATIONAL = "operational"
    TRAFFIC = "traffic"
    MECHANICAL = "mechanical"
    SECURITY = "security"
    OTHER = "other"


class AnalysisType(str, Enum):
    """Enumeration for analysis types."""
    DELAY = "delay"
    CONGESTION = "congestion"
    SCHEDULE_IMPACT = "schedule_impact"
    CASCADING_IMPACT = "cascading_impact"


class ResponseFormat(str, Enum):
    """Enumeration for response formats."""
    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    message: str = "Operation completed successfully"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    status_code: int
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=50, ge=1, le=1000, description="Page size")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database queries."""
        return (self.page - 1) * self.size


class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    page: int
    size: int
    total: int
    pages: int
    has_next: bool
    has_prev: bool


# Flight Data Models
class FlightDataBase(BaseModel):
    """Base flight data model."""
    flight_number: str = Field(..., description="Flight number (e.g., AI101)")
    airline: str = Field(..., description="Airline code or name", alias="airline_code")
    aircraft_type: Optional[str] = Field(None, description="Aircraft type")
    origin_airport: str = Field(..., description="Origin airport code")
    destination_airport: str = Field(..., description="Destination airport code")
    scheduled_departure: datetime = Field(..., description="Scheduled departure time")
    scheduled_arrival: datetime = Field(..., description="Scheduled arrival time")
    actual_departure: Optional[datetime] = Field(None, description="Actual departure time")
    actual_arrival: Optional[datetime] = Field(None, description="Actual arrival time")
    passenger_count: Optional[int] = Field(None, ge=0, description="Number of passengers")
    runway_used: Optional[str] = Field(None, description="Runway identifier")


class FlightDataCreate(FlightDataBase):
    """Flight data creation model."""
    pass


class FlightDataUpdate(BaseModel):
    """Flight data update model."""
    actual_departure: Optional[datetime] = None
    actual_arrival: Optional[datetime] = None
    passenger_count: Optional[int] = Field(None, ge=0)
    runway_used: Optional[str] = None


class FlightData(FlightDataBase):
    """Complete flight data model with computed fields."""
    id: UUID
    delay_minutes: Optional[int] = Field(None, description="Delay in minutes")
    delay_category: Optional[DelayCategory] = Field(None, description="Delay category")
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)
    
    @model_validator(mode='after')
    def calculate_delay(self):
        """Calculate delay minutes from scheduled vs actual times."""
        if self.delay_minutes is None and self.actual_departure and self.scheduled_departure:
            self.delay_minutes = int((self.actual_departure - self.scheduled_departure).total_seconds() / 60)
        return self


class FlightDataResponse(BaseResponse):
    """Flight data response model."""
    data: FlightData


class FlightDataListResponse(PaginatedResponse):
    """Flight data list response model."""
    data: List[FlightData]


# File Upload Models
class FileUploadResponse(BaseResponse):
    """File upload response model."""
    filename: str
    size: int
    records_processed: int
    records_created: int
    records_updated: int
    errors: List[str] = []


# Filter Models
class FlightDataFilter(BaseModel):
    """Flight data filtering parameters."""
    airline: Optional[str] = None
    origin_airport: Optional[str] = None
    destination_airport: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    delay_category: Optional[DelayCategory] = None
    min_delay: Optional[int] = None
    max_delay: Optional[int] = None


# Analysis Models
class DelayAnalysisRequest(BaseModel):
    """Delay analysis request model."""
    airport_code: str = Field(..., description="Airport code (e.g., BOM, DEL)")
    date_from: datetime = Field(..., description="Analysis start date")
    date_to: datetime = Field(..., description="Analysis end date")
    include_weather: bool = Field(default=True, description="Include weather impact")
    granularity: str = Field(default="hourly", description="Analysis granularity")


class DelayAnalysisResult(BaseModel):
    """Delay analysis result model."""
    airport_code: str
    analysis_period: Dict[str, datetime]
    average_delay: float
    median_delay: float
    delay_by_hour: Dict[int, float]
    delay_by_category: Dict[DelayCategory, float]
    optimal_time_slots: List[Dict[str, Any]]
    recommendations: List[str]


class CongestionAnalysisRequest(BaseModel):
    """Congestion analysis request model."""
    airport_code: str
    date_from: datetime
    date_to: datetime
    runway_capacity: Optional[int] = Field(None, description="Runway capacity per hour")


class CongestionAnalysisResult(BaseModel):
    """Congestion analysis result model."""
    airport_code: str
    analysis_period: Dict[str, datetime]
    peak_hours: List[int]
    congestion_by_hour: Dict[int, float]
    capacity_utilization: Dict[int, float]
    busiest_time_slots: List[Dict[str, Any]]
    alternative_slots: List[Dict[str, Any]]


class ScheduleImpactRequest(BaseModel):
    """Schedule impact modeling request model."""
    flight_id: UUID
    proposed_departure: datetime
    proposed_arrival: datetime
    impact_radius: int = Field(default=3, description="Hours to analyze impact")


class ScheduleImpactResult(BaseModel):
    """Schedule impact modeling result model."""
    flight_id: UUID
    original_schedule: Dict[str, datetime]
    proposed_schedule: Dict[str, datetime]
    delay_impact: float
    affected_flights: List[Dict[str, Any]]
    confidence_score: float
    recommendations: List[str]


class CascadingImpactRequest(BaseModel):
    """Cascading impact analysis request model."""
    airport_code: str
    date_from: datetime
    date_to: datetime
    network_depth: int = Field(default=3, description="Network analysis depth")


class CascadingImpactResult(BaseModel):
    """Cascading impact analysis result model."""
    airport_code: str
    analysis_period: Dict[str, datetime]
    critical_flights: List[Dict[str, Any]]
    network_impact_scores: Dict[str, float]
    delay_propagation_paths: List[Dict[str, Any]]
    priority_recommendations: List[str]


# Natural Language Processing Models
class NLPQueryRequest(BaseModel):
    """Natural language query request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="Natural language query")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    session_id: Optional[str] = Field(None, description="Session identifier for conversation")


class NLPQueryResponse(BaseResponse):
    """Natural language query response model."""
    query: str
    response: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence score")
    data: Optional[Dict[str, Any]] = Field(None, description="Supporting data")
    visualizations: Optional[List[Dict[str, Any]]] = Field(None, description="Suggested visualizations")
    follow_up_questions: Optional[List[str]] = Field(None, description="Suggested follow-up questions")
    session_id: Optional[str] = None


class QuerySuggestion(BaseModel):
    """Query suggestion model."""
    text: str
    category: str
    confidence: float


class QuerySuggestionsResponse(BaseResponse):
    """Query suggestions response model."""
    suggestions: List[QuerySuggestion]


class QueryFeedback(BaseModel):
    """Query feedback model."""
    query_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    feedback: Optional[str] = Field(None, description="Optional feedback text")


# Export Models
class ExportRequest(BaseModel):
    """Data export request model."""
    format: ResponseFormat
    filters: Optional[FlightDataFilter] = None
    include_analysis: bool = Field(default=False, description="Include analysis results")


class ExportResponse(BaseResponse):
    """Data export response model."""
    download_url: str
    filename: str
    format: ResponseFormat
    expires_at: datetime


# Health Check Models
class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Individual component health model."""
    name: str
    status: HealthStatus
    response_time: Optional[float] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: HealthStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str
    components: List[ComponentHealth]
    uptime: float