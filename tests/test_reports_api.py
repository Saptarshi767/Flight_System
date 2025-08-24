"""
Tests for reports API endpoints
"""

import pytest
import json
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.reporting.models import ReportType, StakeholderType, ExportFormat


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_report_request():
    """Sample report generation request"""
    return {
        "report_type": "delay_analysis",
        "stakeholder_type": "operations_manager",
        "export_format": "pdf",
        "airport_codes": ["BOM", "DEL"],
        "date_range": {
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-31T23:59:59"
        },
        "include_charts": True,
        "include_recommendations": True,
        "custom_filters": {}
    }


@pytest.fixture
def sample_export_request():
    """Sample data export request"""
    return {
        "export_format": "csv",
        "filename_prefix": "test_export",
        "airport_codes": ["BOM"],
        "date_range": {
            "start": "2024-01-01T00:00:00",
            "end": "2024-01-31T23:59:59"
        },
        "filters": {}
    }


@pytest.fixture
def sample_schedule_request():
    """Sample schedule report request"""
    return {
        "name": "Daily Operations Report",
        "description": "Daily report for operations team",
        "report_request": {
            "report_type": "delay_analysis",
            "stakeholder_type": "operations_manager",
            "export_format": "pdf",
            "airport_codes": ["BOM", "DEL"],
            "include_charts": True,
            "include_recommendations": True,
            "custom_filters": {}
        },
        "cron_expression": "0 8 * * *",
        "recipients": ["ops@airport.com"],
        "is_active": True
    }


class TestReportsAPI:
    """Test cases for reports API endpoints"""
    
    @patch('src.api.routers.reports.report_generator')
    @patch('src.api.routers.reports._get_analysis_data')
    def test_generate_report_success(self, mock_get_data, mock_generator, client, sample_report_request):
        """Test successful report generation"""
        # Mock analysis data
        mock_get_data.return_value = {
            "total_flights": 1000,
            "average_delay": 15.5
        }
        
        # Mock report generator
        mock_generator.generate_report.return_value = "/path/to/report.pdf"
        
        response = client.post("/api/v1/reports/generate", json=sample_report_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "report_path" in data
        assert data["message"] == "Report generated successfully"
    
    def test_generate_report_invalid_date(self, client):
        """Test report generation with invalid date format"""
        invalid_request = {
            "report_type": "delay_analysis",
            "stakeholder_type": "operations_manager",
            "export_format": "pdf",
            "date_range": {
                "start": "invalid-date"
            }
        }
        
        response = client.post("/api/v1/reports/generate", json=invalid_request)
        
        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]
    
    def test_generate_report_missing_fields(self, client):
        """Test report generation with missing required fields"""
        invalid_request = {
            "export_format": "pdf"
            # Missing required fields
        }
        
        response = client.post("/api/v1/reports/generate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.routers.reports.Path')
    def test_download_report_success(self, mock_path, client):
        """Test successful report download"""
        # Mock file existence
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file
        
        with patch('src.api.routers.reports.FileResponse') as mock_response:
            mock_response.return_value = Mock()
            
            response = client.get("/api/v1/reports/download/test_report.pdf")
            
            # The actual response would be a file, but we're mocking it
            mock_response.assert_called_once()
    
    @patch('src.api.routers.reports.Path')
    def test_download_report_not_found(self, mock_path, client):
        """Test report download when file doesn't exist"""
        # Mock file not existing
        mock_file = Mock()
        mock_file.exists.return_value = False
        mock_path.return_value = mock_file
        
        response = client.get("/api/v1/reports/download/nonexistent.pdf")
        
        assert response.status_code == 404
        assert "Report not found" in response.json()["detail"]
    
    @patch('src.api.routers.reports.export_manager')
    @patch('src.api.routers.reports._get_flight_data')
    def test_export_data_success(self, mock_get_data, mock_export_manager, client, sample_export_request):
        """Test successful data export"""
        # Mock flight data
        mock_get_data.return_value = Mock()
        
        # Mock export result
        from src.reporting.models import ExportResult, ExportFormat
        mock_result = ExportResult(
            success=True,
            file_path="/path/to/export.csv",
            file_size=1024,
            export_format=ExportFormat.CSV
        )
        mock_export_manager.export_flight_data.return_value = mock_result
        
        response = client.post("/api/v1/reports/export", json=sample_export_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["file_path"] == "/path/to/export.csv"
        assert data["export_format"] == "csv"
    
    @patch('src.api.routers.reports.Path')
    def test_download_export_success(self, mock_path, client):
        """Test successful export download"""
        # Mock file existence
        mock_file = Mock()
        mock_file.exists.return_value = True
        mock_path.return_value = mock_file
        
        with patch('src.api.routers.reports.FileResponse') as mock_response:
            mock_response.return_value = Mock()
            
            response = client.get("/api/v1/reports/export/download/test_export.csv")
            
            mock_response.assert_called_once()
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_schedule_report_success(self, mock_scheduler, client, sample_schedule_request):
        """Test successful report scheduling"""
        # Mock scheduler
        mock_scheduler.add_scheduled_report.return_value = True
        
        response = client.post("/api/v1/reports/schedule", json=sample_schedule_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "schedule_id" in data
        assert data["message"] == "Report scheduled successfully"
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_schedule_report_failure(self, mock_scheduler, client, sample_schedule_request):
        """Test report scheduling failure"""
        # Mock scheduler failure
        mock_scheduler.add_scheduled_report.return_value = False
        
        response = client.post("/api/v1/reports/schedule", json=sample_schedule_request)
        
        assert response.status_code == 400
        assert "Failed to schedule report" in response.json()["detail"]
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_get_scheduled_reports(self, mock_scheduler, client):
        """Test getting scheduled reports"""
        from src.reporting.models import ScheduledReport, ReportRequest
        
        # Mock scheduled reports
        mock_reports = [
            ScheduledReport(
                schedule_id="test_001",
                name="Test Report",
                report_request=ReportRequest(
                    report_type=ReportType.DELAY_ANALYSIS,
                    stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                    export_format=ExportFormat.PDF
                ),
                cron_expression="0 8 * * *",
                recipients=[],
                is_active=True
            )
        ]
        mock_scheduler.get_scheduled_reports.return_value = mock_reports
        
        response = client.get("/api/v1/reports/schedule")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["schedule_id"] == "test_001"
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_get_scheduled_report_by_id(self, mock_scheduler, client):
        """Test getting specific scheduled report"""
        from src.reporting.models import ScheduledReport, ReportRequest
        
        mock_report = ScheduledReport(
            schedule_id="test_001",
            name="Test Report",
            report_request=ReportRequest(
                report_type=ReportType.DELAY_ANALYSIS,
                stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                export_format=ExportFormat.PDF
            ),
            cron_expression="0 8 * * *",
            recipients=[],
            is_active=True
        )
        mock_scheduler.get_scheduled_report.return_value = mock_report
        
        response = client.get("/api/v1/reports/schedule/test_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["schedule_id"] == "test_001"
        assert data["name"] == "Test Report"
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_get_scheduled_report_not_found(self, mock_scheduler, client):
        """Test getting non-existent scheduled report"""
        mock_scheduler.get_scheduled_report.return_value = None
        
        response = client.get("/api/v1/reports/schedule/nonexistent")
        
        assert response.status_code == 404
        assert "Scheduled report not found" in response.json()["detail"]
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_update_scheduled_report(self, mock_scheduler, client, sample_schedule_request):
        """Test updating scheduled report"""
        from src.reporting.models import ScheduledReport, ReportRequest
        
        # Mock existing report
        existing_report = ScheduledReport(
            schedule_id="test_001",
            name="Old Name",
            report_request=ReportRequest(
                report_type=ReportType.DELAY_ANALYSIS,
                stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                export_format=ExportFormat.PDF
            ),
            cron_expression="0 8 * * *",
            recipients=[],
            is_active=True
        )
        mock_scheduler.get_scheduled_report.return_value = existing_report
        mock_scheduler.update_scheduled_report.return_value = True
        
        response = client.put("/api/v1/reports/schedule/test_001", json=sample_schedule_request)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Scheduled report updated successfully"
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_delete_scheduled_report(self, mock_scheduler, client):
        """Test deleting scheduled report"""
        mock_scheduler.remove_scheduled_report.return_value = True
        
        response = client.delete("/api/v1/reports/schedule/test_001")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["message"] == "Scheduled report deleted successfully"
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_delete_scheduled_report_not_found(self, mock_scheduler, client):
        """Test deleting non-existent scheduled report"""
        mock_scheduler.remove_scheduled_report.return_value = False
        
        response = client.delete("/api/v1/reports/schedule/nonexistent")
        
        assert response.status_code == 404
        assert "Scheduled report not found" in response.json()["detail"]
    
    def test_get_available_templates(self, client):
        """Test getting available report templates"""
        response = client.get("/api/v1/reports/templates")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert "operations_manager" in data
        assert "air_traffic_controller" in data
        assert isinstance(data["operations_manager"], list)
    
    @patch('src.api.routers.reports.export_manager')
    @patch('src.api.routers.reports.Path')
    def test_cleanup_old_files(self, mock_path, mock_export_manager, client):
        """Test cleanup of old files"""
        # Mock directory structure
        mock_reports_dir = Mock()
        mock_reports_dir.exists.return_value = True
        mock_reports_dir.iterdir.return_value = []
        mock_path.return_value = mock_reports_dir
        
        response = client.post("/api/v1/reports/cleanup?days_to_keep=30")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "Cleaned up files" in data["message"]
        
        # Verify cleanup was called
        mock_export_manager.cleanup_old_exports.assert_called_once_with(30)


class TestReportsAPIErrorHandling:
    """Test error handling in reports API"""
    
    @patch('src.api.routers.reports.report_generator')
    def test_generate_report_internal_error(self, mock_generator, client, sample_report_request):
        """Test report generation with internal error"""
        # Mock generator to raise exception
        mock_generator.generate_report.side_effect = Exception("Internal error")
        
        response = client.post("/api/v1/reports/generate", json=sample_report_request)
        
        assert response.status_code == 500
        assert "Internal error" in response.json()["detail"]
    
    @patch('src.api.routers.reports.export_manager')
    def test_export_data_internal_error(self, mock_export_manager, client, sample_export_request):
        """Test data export with internal error"""
        # Mock export manager to raise exception
        mock_export_manager.export_flight_data.side_effect = Exception("Export failed")
        
        response = client.post("/api/v1/reports/export", json=sample_export_request)
        
        assert response.status_code == 500
        assert "Export failed" in response.json()["detail"]
    
    @patch('src.api.routers.reports.report_scheduler')
    def test_schedule_report_internal_error(self, mock_scheduler, client, sample_schedule_request):
        """Test report scheduling with internal error"""
        # Mock scheduler to raise exception
        mock_scheduler.add_scheduled_report.side_effect = Exception("Scheduling failed")
        
        response = client.post("/api/v1/reports/schedule", json=sample_schedule_request)
        
        assert response.status_code == 500
        assert "Scheduling failed" in response.json()["detail"]


class TestReportsAPIValidation:
    """Test input validation for reports API"""
    
    def test_invalid_report_type(self, client):
        """Test with invalid report type"""
        invalid_request = {
            "report_type": "invalid_type",
            "stakeholder_type": "operations_manager",
            "export_format": "pdf"
        }
        
        response = client.post("/api/v1/reports/generate", json=invalid_request)
        
        assert response.status_code == 422
    
    def test_invalid_stakeholder_type(self, client):
        """Test with invalid stakeholder type"""
        invalid_request = {
            "report_type": "delay_analysis",
            "stakeholder_type": "invalid_stakeholder",
            "export_format": "pdf"
        }
        
        response = client.post("/api/v1/reports/generate", json=invalid_request)
        
        assert response.status_code == 422
    
    def test_invalid_export_format(self, client):
        """Test with invalid export format"""
        invalid_request = {
            "report_type": "delay_analysis",
            "stakeholder_type": "operations_manager",
            "export_format": "invalid_format"
        }
        
        response = client.post("/api/v1/reports/generate", json=invalid_request)
        
        assert response.status_code == 422
    
    def test_invalid_cron_expression(self, client):
        """Test scheduling with invalid cron expression"""
        invalid_request = {
            "name": "Test Report",
            "report_request": {
                "report_type": "delay_analysis",
                "stakeholder_type": "operations_manager",
                "export_format": "pdf"
            },
            "cron_expression": "invalid_cron",
            "recipients": [],
            "is_active": True
        }
        
        with patch('src.api.routers.reports.report_scheduler') as mock_scheduler:
            mock_scheduler.add_scheduled_report.return_value = False
            
            response = client.post("/api/v1/reports/schedule", json=invalid_request)
            
            assert response.status_code == 400