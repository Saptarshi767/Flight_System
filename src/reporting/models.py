"""
Data models for reporting functionality
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ReportType(str, Enum):
    """Types of reports that can be generated"""
    DELAY_ANALYSIS = "delay_analysis"
    CONGESTION_ANALYSIS = "congestion_analysis"
    SCHEDULE_IMPACT = "schedule_impact"
    CASCADING_IMPACT = "cascading_impact"
    COMPREHENSIVE = "comprehensive"
    EXECUTIVE_SUMMARY = "executive_summary"


class ExportFormat(str, Enum):
    """Supported export formats"""
    PDF = "pdf"
    CSV = "csv"
    EXCEL = "excel"
    JSON = "json"


class StakeholderType(str, Enum):
    """Different stakeholder types for report templates"""
    OPERATIONS_MANAGER = "operations_manager"
    AIR_TRAFFIC_CONTROLLER = "air_traffic_controller"
    FLIGHT_SCHEDULER = "flight_scheduler"
    NETWORK_OPERATIONS = "network_operations"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"


class ReportRequest(BaseModel):
    """Request model for report generation"""
    report_type: ReportType
    stakeholder_type: StakeholderType
    export_format: ExportFormat
    airport_codes: List[str] = Field(default_factory=list)
    date_range: Dict[str, datetime] = Field(default_factory=dict)
    include_charts: bool = True
    include_recommendations: bool = True
    custom_filters: Dict[str, Any] = Field(default_factory=dict)


class ReportMetadata(BaseModel):
    """Metadata for generated reports"""
    report_id: str
    report_type: ReportType
    stakeholder_type: StakeholderType
    generated_at: datetime
    generated_by: Optional[str] = None
    data_period: Dict[str, datetime]
    airports_included: List[str]
    total_flights_analyzed: int
    confidence_score: Optional[float] = None


class ChartConfig(BaseModel):
    """Configuration for charts in reports"""
    chart_type: str
    title: str
    data_source: str
    x_axis: str
    y_axis: str
    color_scheme: Optional[str] = None
    size: Dict[str, int] = Field(default_factory=lambda: {"width": 800, "height": 600})


class ReportSection(BaseModel):
    """Individual section of a report"""
    section_id: str
    title: str
    content: str
    charts: List[ChartConfig] = Field(default_factory=list)
    tables: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    order: int = 0


class ReportTemplate(BaseModel):
    """Template structure for different report types"""
    template_id: str
    name: str
    description: str
    stakeholder_type: StakeholderType
    sections: List[ReportSection]
    default_charts: List[ChartConfig] = Field(default_factory=list)
    styling: Dict[str, Any] = Field(default_factory=dict)


class ScheduledReport(BaseModel):
    """Configuration for scheduled reports"""
    schedule_id: str
    name: str
    description: Optional[str] = None
    report_request: ReportRequest
    cron_expression: str
    recipients: List[str] = Field(default_factory=list)
    is_active: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)


class ExportResult(BaseModel):
    """Result of an export operation"""
    success: bool
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    export_format: ExportFormat
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None