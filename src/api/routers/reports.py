"""
API endpoints for report generation and export functionality
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ...reporting import ReportGenerator, ExportManager, ReportScheduler
from ...reporting.models import (
    ReportRequest, ReportType, StakeholderType, ExportFormat,
    ScheduledReport, ExportResult
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])

# Initialize reporting components
report_generator = ReportGenerator()
export_manager = ExportManager()
report_scheduler = ReportScheduler(report_generator, export_manager)


class ReportGenerationRequest(BaseModel):
    """Request model for report generation"""
    report_type: ReportType
    stakeholder_type: StakeholderType
    export_format: ExportFormat = ExportFormat.PDF
    airport_codes: List[str] = []
    date_range: Dict[str, str] = {}
    include_charts: bool = True
    include_recommendations: bool = True
    custom_filters: Dict[str, Any] = {}


class ExportDataRequest(BaseModel):
    """Request model for data export"""
    export_format: ExportFormat
    filename_prefix: str = "flight_data"
    airport_codes: List[str] = []
    date_range: Dict[str, str] = {}
    filters: Dict[str, Any] = {}


class ScheduleReportRequest(BaseModel):
    """Request model for scheduling reports"""
    name: str
    description: Optional[str] = None
    report_request: ReportGenerationRequest
    cron_expression: str
    recipients: List[str] = []
    is_active: bool = True


@router.post("/generate", response_model=Dict[str, str])
async def generate_report(
    request: ReportGenerationRequest,
    background_tasks: BackgroundTasks
):
    """Generate a report based on the request parameters"""
    try:
        # Convert date strings to datetime objects
        date_range = {}
        if request.date_range:
            for key, value in request.date_range.items():
                try:
                    date_range[key] = datetime.fromisoformat(value)
                except ValueError:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid date format for {key}: {value}"
                    )
        
        # Create report request
        report_request = ReportRequest(
            report_type=request.report_type,
            stakeholder_type=request.stakeholder_type,
            export_format=request.export_format,
            airport_codes=request.airport_codes,
            date_range=date_range,
            include_charts=request.include_charts,
            include_recommendations=request.include_recommendations,
            custom_filters=request.custom_filters
        )
        
        # Get analysis data (this would integrate with your analysis engines)
        analysis_data = await _get_analysis_data(report_request)
        
        # Generate report
        report_path = report_generator.generate_report(
            report_request,
            analysis_data
        )
        
        return {
            "status": "success",
            "report_path": report_path,
            "message": "Report generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{report_filename}")
async def download_report(report_filename: str):
    """Download a generated report"""
    try:
        report_path = Path("reports") / report_filename
        
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        
        return FileResponse(
            path=str(report_path),
            filename=report_filename,
            media_type='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"Report download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/export", response_model=ExportResult)
async def export_data(request: ExportDataRequest):
    """Export flight data in specified format"""
    try:
        # Get flight data based on filters (this would integrate with your data layer)
        flight_data = await _get_flight_data(
            request.airport_codes,
            request.date_range,
            request.filters
        )
        
        # Export data
        result = export_manager.export_flight_data(
            flight_data,
            request.export_format,
            request.filename_prefix
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Data export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/download/{filename}")
async def download_export(filename: str):
    """Download an exported data file"""
    try:
        export_path = Path("exports") / filename
        
        if not export_path.exists():
            raise HTTPException(status_code=404, detail="Export file not found")
        
        # Determine media type based on file extension
        media_type = "application/octet-stream"
        if filename.endswith('.csv'):
            media_type = "text/csv"
        elif filename.endswith('.xlsx'):
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif filename.endswith('.json'):
            media_type = "application/json"
        
        return FileResponse(
            path=str(export_path),
            filename=filename,
            media_type=media_type
        )
        
    except Exception as e:
        logger.error(f"Export download failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schedule", response_model=Dict[str, str])
async def schedule_report(request: ScheduleReportRequest):
    """Schedule a report for automatic generation"""
    try:
        # Convert report generation request
        report_request = ReportRequest(
            report_type=request.report_request.report_type,
            stakeholder_type=request.report_request.stakeholder_type,
            export_format=request.report_request.export_format,
            airport_codes=request.report_request.airport_codes,
            date_range={},  # Will be calculated at runtime
            include_charts=request.report_request.include_charts,
            include_recommendations=request.report_request.include_recommendations,
            custom_filters=request.report_request.custom_filters
        )
        
        # Create scheduled report
        scheduled_report = ScheduledReport(
            schedule_id=f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name=request.name,
            description=request.description,
            report_request=report_request,
            cron_expression=request.cron_expression,
            recipients=request.recipients,
            is_active=request.is_active
        )
        
        # Add to scheduler
        success = report_scheduler.add_scheduled_report(scheduled_report)
        
        if success:
            return {
                "status": "success",
                "schedule_id": scheduled_report.schedule_id,
                "message": "Report scheduled successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to schedule report")
            
    except Exception as e:
        logger.error(f"Report scheduling failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule", response_model=List[ScheduledReport])
async def get_scheduled_reports():
    """Get all scheduled reports"""
    try:
        return report_scheduler.get_scheduled_reports()
    except Exception as e:
        logger.error(f"Failed to get scheduled reports: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schedule/{schedule_id}", response_model=ScheduledReport)
async def get_scheduled_report(schedule_id: str):
    """Get a specific scheduled report"""
    try:
        scheduled_report = report_scheduler.get_scheduled_report(schedule_id)
        if not scheduled_report:
            raise HTTPException(status_code=404, detail="Scheduled report not found")
        return scheduled_report
    except Exception as e:
        logger.error(f"Failed to get scheduled report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/schedule/{schedule_id}", response_model=Dict[str, str])
async def update_scheduled_report(schedule_id: str, request: ScheduleReportRequest):
    """Update a scheduled report"""
    try:
        # Get existing scheduled report
        existing_report = report_scheduler.get_scheduled_report(schedule_id)
        if not existing_report:
            raise HTTPException(status_code=404, detail="Scheduled report not found")
        
        # Convert report generation request
        report_request = ReportRequest(
            report_type=request.report_request.report_type,
            stakeholder_type=request.report_request.stakeholder_type,
            export_format=request.report_request.export_format,
            airport_codes=request.report_request.airport_codes,
            date_range={},
            include_charts=request.report_request.include_charts,
            include_recommendations=request.report_request.include_recommendations,
            custom_filters=request.report_request.custom_filters
        )
        
        # Update scheduled report
        updated_report = ScheduledReport(
            schedule_id=schedule_id,
            name=request.name,
            description=request.description,
            report_request=report_request,
            cron_expression=request.cron_expression,
            recipients=request.recipients,
            is_active=request.is_active,
            created_at=existing_report.created_at
        )
        
        success = report_scheduler.update_scheduled_report(updated_report)
        
        if success:
            return {
                "status": "success",
                "message": "Scheduled report updated successfully"
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to update scheduled report")
            
    except Exception as e:
        logger.error(f"Failed to update scheduled report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/schedule/{schedule_id}", response_model=Dict[str, str])
async def delete_scheduled_report(schedule_id: str):
    """Delete a scheduled report"""
    try:
        success = report_scheduler.remove_scheduled_report(schedule_id)
        
        if success:
            return {
                "status": "success",
                "message": "Scheduled report deleted successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Scheduled report not found")
            
    except Exception as e:
        logger.error(f"Failed to delete scheduled report: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/templates", response_model=Dict[str, List[str]])
async def get_available_templates():
    """Get available report templates"""
    try:
        from ...reporting.templates import ReportTemplates
        return ReportTemplates.get_available_templates()
    except Exception as e:
        logger.error(f"Failed to get templates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup")
async def cleanup_old_files(
    days_to_keep: int = Query(30, description="Number of days to keep files")
):
    """Clean up old report and export files"""
    try:
        # Cleanup exports
        export_manager.cleanup_old_exports(days_to_keep)
        
        # Cleanup reports
        reports_dir = Path("reports")
        if reports_dir.exists():
            cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
            
            for file_path in reports_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted old report file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {str(e)}")
        
        return {
            "status": "success",
            "message": f"Cleaned up files older than {days_to_keep} days"
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions (these would integrate with your actual data and analysis layers)

async def _get_analysis_data(request: ReportRequest) -> Dict[str, Any]:
    """Get analysis data for report generation (placeholder)"""
    # This would integrate with your actual analysis engines
    return {
        "total_flights": 1000,
        "average_delay": 15.5,
        "on_time_performance": 0.85,
        "hourly_delays": [10, 8, 5, 3, 2, 5, 12, 18, 22, 25, 20, 15, 18, 22, 25, 28, 30, 25, 20, 15, 12, 10, 8, 6],
        "delay_causes": {
            "Weather": 35,
            "Air Traffic": 25,
            "Mechanical": 20,
            "Crew": 10,
            "Other": 10
        },
        "traffic_density": [[5, 8, 12, 15, 18, 22, 25, 20, 15, 12, 8, 5] for _ in range(7)],
        "optimal_times": {
            "scheduled_times": list(range(6, 23)),
            "delays": [5, 3, 2, 4, 8, 12, 15, 18, 22, 25, 20, 15, 12, 8, 5, 3, 2]
        },
        "network_impact": {
            "flights": ["AI101", "6E202", "SG303", "UK404", "9W505"],
            "impact_scores": [85, 72, 68, 55, 42]
        },
        "confidence_score": 0.92
    }


async def _get_flight_data(
    airport_codes: List[str],
    date_range: Dict[str, str],
    filters: Dict[str, Any]
) -> Any:
    """Get flight data for export (placeholder)"""
    import pandas as pd
    
    # This would integrate with your actual data layer
    # For now, return mock data
    data = {
        "flight_number": ["AI101", "6E202", "SG303", "UK404", "9W505"],
        "origin_airport": ["BOM", "DEL", "BOM", "DEL", "BOM"],
        "destination_airport": ["DEL", "BOM", "DEL", "BOM", "DEL"],
        "scheduled_departure": ["2024-01-01 08:00:00", "2024-01-01 10:00:00", "2024-01-01 12:00:00", "2024-01-01 14:00:00", "2024-01-01 16:00:00"],
        "actual_departure": ["2024-01-01 08:15:00", "2024-01-01 10:05:00", "2024-01-01 12:20:00", "2024-01-01 14:10:00", "2024-01-01 16:00:00"],
        "delay_minutes": [15, 5, 20, 10, 0]
    }
    
    return pd.DataFrame(data)