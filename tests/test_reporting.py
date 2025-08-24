"""
Tests for reporting and export functionality
"""

import pytest
import pandas as pd
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.reporting import ReportGenerator, ExportManager, ReportScheduler
from src.reporting.models import (
    ReportRequest, ReportType, StakeholderType, ExportFormat,
    ScheduledReport, ExportResult
)
from src.reporting.templates import ReportTemplates


class TestExportManager:
    """Test cases for ExportManager"""
    
    @pytest.fixture
    def export_manager(self):
        """Create ExportManager instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ExportManager(export_directory=temp_dir)
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing"""
        return pd.DataFrame({
            'flight_number': ['AI101', '6E202', 'SG303'],
            'origin_airport': ['BOM', 'DEL', 'BOM'],
            'destination_airport': ['DEL', 'BOM', 'DEL'],
            'delay_minutes': [15, 5, 20]
        })
    
    def test_export_csv(self, export_manager, sample_dataframe):
        """Test CSV export functionality"""
        result = export_manager.export_data(
            sample_dataframe,
            ExportFormat.CSV,
            "test_flights"
        )
        
        assert result.success is True
        assert result.export_format == ExportFormat.CSV
        assert result.file_path is not None
        assert Path(result.file_path).exists()
        assert result.file_size > 0
    
    def test_export_excel(self, export_manager, sample_dataframe):
        """Test Excel export functionality"""
        result = export_manager.export_data(
            sample_dataframe,
            ExportFormat.EXCEL,
            "test_flights"
        )
        
        assert result.success is True
        assert result.export_format == ExportFormat.EXCEL
        assert result.file_path is not None
        assert Path(result.file_path).exists()
        assert result.file_size > 0
    
    def test_export_json(self, export_manager, sample_dataframe):
        """Test JSON export functionality"""
        result = export_manager.export_data(
            sample_dataframe,
            ExportFormat.JSON,
            "test_flights"
        )
        
        assert result.success is True
        assert result.export_format == ExportFormat.JSON
        assert result.file_path is not None
        assert Path(result.file_path).exists()
        assert result.file_size > 0
        
        # Verify JSON content
        with open(result.file_path, 'r') as f:
            json_data = json.load(f)
            assert "data" in json_data
            assert "exported_at" in json_data
            assert "metadata" in json_data
    
    def test_export_flight_data_with_metadata(self, export_manager, sample_dataframe):
        """Test flight data export with metadata"""
        # Add datetime columns for metadata calculation
        sample_dataframe['scheduled_departure'] = pd.to_datetime([
            '2024-01-01 08:00:00',
            '2024-01-01 10:00:00',
            '2024-01-01 12:00:00'
        ])
        
        result = export_manager.export_flight_data(
            sample_dataframe,
            ExportFormat.JSON,
            "flight_data_test"
        )
        
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["total_flights"] == 3
        assert "date_range" in result.metadata
        assert "airports" in result.metadata
    
    def test_export_analysis_results(self, export_manager):
        """Test analysis results export"""
        analysis_data = {
            "total_flights": 1000,
            "average_delay": 15.5,
            "delay_causes": {"Weather": 35, "Traffic": 25}
        }
        
        result = export_manager.export_analysis_results(
            analysis_data,
            ExportFormat.JSON,
            "delay_analysis"
        )
        
        assert result.success is True
        assert result.metadata is not None
        assert result.metadata["analysis_type"] == "delay_analysis"
    
    def test_export_unsupported_format(self, export_manager, sample_dataframe):
        """Test handling of unsupported export format"""
        # This would require modifying the enum, so we'll test error handling
        with patch.object(export_manager, '_export_csv', side_effect=Exception("Test error")):
            result = export_manager.export_data(
                sample_dataframe,
                ExportFormat.CSV,
                "test_error"
            )
            
            assert result.success is False
            assert result.error_message is not None
    
    def test_cleanup_old_exports(self, export_manager, sample_dataframe):
        """Test cleanup of old export files"""
        import os
        
        # Create a test export
        result = export_manager.export_data(
            sample_dataframe,
            ExportFormat.CSV,
            "old_export"
        )
        
        # Modify file timestamp to make it appear old
        old_time = datetime.now().timestamp() - (31 * 24 * 3600)  # 31 days ago
        os.utime(result.file_path, (old_time, old_time))
        
        # Run cleanup
        export_manager.cleanup_old_exports(days_to_keep=30)
        
        # File should be deleted
        assert not Path(result.file_path).exists()


class TestReportTemplates:
    """Test cases for ReportTemplates"""
    
    def test_get_operations_manager_delay_template(self):
        """Test operations manager delay analysis template"""
        template = ReportTemplates.get_template(
            StakeholderType.OPERATIONS_MANAGER,
            ReportType.DELAY_ANALYSIS
        )
        
        assert template.stakeholder_type == StakeholderType.OPERATIONS_MANAGER
        assert template.template_id == "ops_mgr_delay"
        assert len(template.sections) > 0
        assert any(section.title == "Executive Summary" for section in template.sections)
    
    def test_get_air_traffic_controller_congestion_template(self):
        """Test air traffic controller congestion analysis template"""
        template = ReportTemplates.get_template(
            StakeholderType.AIR_TRAFFIC_CONTROLLER,
            ReportType.CONGESTION_ANALYSIS
        )
        
        assert template.stakeholder_type == StakeholderType.AIR_TRAFFIC_CONTROLLER
        assert template.template_id == "atc_congestion"
        assert len(template.sections) > 0
        assert any(section.title == "Peak Hours Analysis" for section in template.sections)
    
    def test_get_executive_comprehensive_template(self):
        """Test executive comprehensive report template"""
        template = ReportTemplates.get_template(
            StakeholderType.EXECUTIVE,
            ReportType.COMPREHENSIVE
        )
        
        assert template.stakeholder_type == StakeholderType.EXECUTIVE
        assert template.template_id == "exec_comprehensive"
        assert len(template.sections) > 0
        assert any(section.title == "Key Performance Indicators" for section in template.sections)
    
    def test_get_generic_template_fallback(self):
        """Test generic template fallback for unsupported combinations"""
        template = ReportTemplates.get_template(
            StakeholderType.TECHNICAL,
            ReportType.DELAY_ANALYSIS
        )
        
        assert template.stakeholder_type == StakeholderType.TECHNICAL
        assert "generic" in template.template_id
        assert len(template.sections) > 0
    
    def test_get_available_templates(self):
        """Test getting available templates list"""
        templates = ReportTemplates.get_available_templates()
        
        assert isinstance(templates, dict)
        assert "operations_manager" in templates
        assert "air_traffic_controller" in templates
        assert "executive" in templates
        assert isinstance(templates["operations_manager"], list)


class TestReportGenerator:
    """Test cases for ReportGenerator"""
    
    @pytest.fixture
    def report_generator(self):
        """Create ReportGenerator instance for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield ReportGenerator(
                template_directory=f"{temp_dir}/templates",
                output_directory=f"{temp_dir}/reports"
            )
    
    @pytest.fixture
    def sample_report_request(self):
        """Create sample report request"""
        return ReportRequest(
            report_type=ReportType.DELAY_ANALYSIS,
            stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
            export_format=ExportFormat.PDF,
            airport_codes=["BOM", "DEL"],
            include_charts=True,
            include_recommendations=True
        )
    
    @pytest.fixture
    def sample_analysis_data(self):
        """Create sample analysis data"""
        return {
            "total_flights": 1000,
            "average_delay": 15.5,
            "hourly_delays": [10, 8, 5, 3, 2, 5, 12, 18, 22, 25, 20, 15, 18, 22, 25, 28, 30, 25, 20, 15, 12, 10, 8, 6],
            "delay_causes": {
                "Weather": 35,
                "Air Traffic": 25,
                "Mechanical": 20,
                "Crew": 10,
                "Other": 10
            },
            "confidence_score": 0.92
        }
    
    @patch('src.reporting.report_generator.ReportGenerator._generate_charts')
    def test_generate_pdf_report(self, mock_charts, report_generator, sample_report_request, sample_analysis_data):
        """Test PDF report generation"""
        # Mock chart generation to avoid plotly dependencies
        mock_charts.return_value = {}
        
        report_path = report_generator.generate_report(
            sample_report_request,
            sample_analysis_data
        )
        
        assert report_path is not None
        assert "report_" in report_path
        assert report_path.endswith(".pdf")
        # The actual file should be created since we're using real ReportLab
        assert Path(report_path).exists()
    
    def test_create_metadata(self, report_generator, sample_report_request, sample_analysis_data):
        """Test report metadata creation"""
        metadata = report_generator._create_metadata(
            sample_report_request,
            sample_analysis_data,
            None
        )
        
        assert metadata.report_type == ReportType.DELAY_ANALYSIS
        assert metadata.stakeholder_type == StakeholderType.OPERATIONS_MANAGER
        assert metadata.total_flights_analyzed == 1000
        assert metadata.confidence_score == 0.92
        assert metadata.airports_included == ["BOM", "DEL"]
    
    @patch('plotly.express.bar')
    @patch('plotly.express.pie')
    def test_generate_delay_charts(self, mock_pie, mock_bar, report_generator, sample_analysis_data):
        """Test delay chart generation"""
        # Mock plotly figures
        mock_fig = Mock()
        mock_fig.to_image.return_value = b"fake_image_data"
        mock_bar.return_value = mock_fig
        mock_pie.return_value = mock_fig
        
        charts = report_generator._generate_delay_charts(sample_analysis_data)
        
        assert "hourly_delays" in charts
        assert "delay_causes" in charts
        assert charts["hourly_delays"].startswith("data:image/png;base64,")
        mock_bar.assert_called_once()
        mock_pie.assert_called_once()
    
    def test_fig_to_base64(self, report_generator):
        """Test figure to base64 conversion"""
        mock_fig = Mock()
        mock_fig.to_image.return_value = b"test_image_data"
        
        result = report_generator._fig_to_base64(mock_fig)
        
        assert result.startswith("data:image/png;base64,")
        mock_fig.to_image.assert_called_once_with(format="png", width=800, height=600)


class TestReportScheduler:
    """Test cases for ReportScheduler"""
    
    @pytest.fixture
    def report_scheduler(self):
        """Create ReportScheduler instance for testing"""
        mock_generator = Mock(spec=ReportGenerator)
        mock_export_manager = Mock(spec=ExportManager)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = f"{temp_dir}/test_scheduled_reports.json"
            yield ReportScheduler(
                mock_generator,
                mock_export_manager,
                config_file=config_file
            )
    
    @pytest.fixture
    def sample_scheduled_report(self):
        """Create sample scheduled report"""
        return ScheduledReport(
            schedule_id="test_schedule_001",
            name="Daily Delay Report",
            description="Daily delay analysis report",
            report_request=ReportRequest(
                report_type=ReportType.DELAY_ANALYSIS,
                stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                export_format=ExportFormat.PDF
            ),
            cron_expression="0 8 * * *",  # Daily at 8 AM
            recipients=["manager@airport.com"],
            is_active=True
        )
    
    def test_add_scheduled_report(self, report_scheduler, sample_scheduled_report):
        """Test adding a scheduled report"""
        success = report_scheduler.add_scheduled_report(sample_scheduled_report)
        
        assert success is True
        assert sample_scheduled_report.schedule_id in report_scheduler.scheduled_reports
        assert sample_scheduled_report.next_run is not None
    
    def test_add_scheduled_report_invalid_cron(self, report_scheduler):
        """Test adding scheduled report with invalid cron expression"""
        invalid_report = ScheduledReport(
            schedule_id="invalid_schedule",
            name="Invalid Report",
            report_request=ReportRequest(
                report_type=ReportType.DELAY_ANALYSIS,
                stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                export_format=ExportFormat.PDF
            ),
            cron_expression="invalid_cron",
            recipients=[],
            is_active=True
        )
        
        success = report_scheduler.add_scheduled_report(invalid_report)
        assert success is False
    
    def test_remove_scheduled_report(self, report_scheduler, sample_scheduled_report):
        """Test removing a scheduled report"""
        # Add report first
        report_scheduler.add_scheduled_report(sample_scheduled_report)
        
        # Remove report
        success = report_scheduler.remove_scheduled_report(sample_scheduled_report.schedule_id)
        
        assert success is True
        assert sample_scheduled_report.schedule_id not in report_scheduler.scheduled_reports
    
    def test_remove_nonexistent_report(self, report_scheduler):
        """Test removing a non-existent scheduled report"""
        success = report_scheduler.remove_scheduled_report("nonexistent_id")
        assert success is False
    
    def test_update_scheduled_report(self, report_scheduler, sample_scheduled_report):
        """Test updating a scheduled report"""
        # Add report first
        report_scheduler.add_scheduled_report(sample_scheduled_report)
        
        # Update report
        sample_scheduled_report.name = "Updated Daily Report"
        sample_scheduled_report.cron_expression = "0 9 * * *"  # Change to 9 AM
        
        success = report_scheduler.update_scheduled_report(sample_scheduled_report)
        
        assert success is True
        updated_report = report_scheduler.get_scheduled_report(sample_scheduled_report.schedule_id)
        assert updated_report.name == "Updated Daily Report"
    
    def test_get_scheduled_reports(self, report_scheduler, sample_scheduled_report):
        """Test getting all scheduled reports"""
        report_scheduler.add_scheduled_report(sample_scheduled_report)
        
        reports = report_scheduler.get_scheduled_reports()
        
        assert len(reports) == 1
        assert reports[0].schedule_id == sample_scheduled_report.schedule_id
    
    def test_validate_cron_expression(self, report_scheduler):
        """Test cron expression validation"""
        assert report_scheduler._validate_cron_expression("0 8 * * *") is True
        assert report_scheduler._validate_cron_expression("0 */2 * * *") is True
        assert report_scheduler._validate_cron_expression("invalid") is False
        assert report_scheduler._validate_cron_expression("") is False
    
    def test_calculate_next_run(self, report_scheduler):
        """Test next run time calculation"""
        base_time = datetime(2024, 1, 1, 6, 0, 0)  # 6 AM
        next_run = report_scheduler._calculate_next_run("0 8 * * *", base_time)
        
        # Should be 8 AM on the same day
        expected = datetime(2024, 1, 1, 8, 0, 0)
        assert next_run == expected
    
    def test_create_daily_report_schedule(self, report_scheduler):
        """Test creating daily report schedule"""
        report_request = ReportRequest(
            report_type=ReportType.DELAY_ANALYSIS,
            stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
            export_format=ExportFormat.PDF
        )
        
        scheduled_report = report_scheduler.create_daily_report_schedule(
            "test_daily",
            report_request,
            hour=9,
            recipients=["test@example.com"]
        )
        
        assert scheduled_report.name == "test_daily"
        assert scheduled_report.cron_expression == "0 9 * * *"
        assert scheduled_report.recipients == ["test@example.com"]
    
    def test_create_weekly_report_schedule(self, report_scheduler):
        """Test creating weekly report schedule"""
        report_request = ReportRequest(
            report_type=ReportType.COMPREHENSIVE,
            stakeholder_type=StakeholderType.EXECUTIVE,
            export_format=ExportFormat.PDF
        )
        
        scheduled_report = report_scheduler.create_weekly_report_schedule(
            "test_weekly",
            report_request,
            day_of_week=1,  # Monday
            hour=10,
            recipients=["exec@example.com"]
        )
        
        assert scheduled_report.name == "test_weekly"
        assert scheduled_report.cron_expression == "0 10 * * 1"
        assert scheduled_report.recipients == ["exec@example.com"]
    
    @pytest.mark.asyncio
    async def test_get_analysis_data(self, report_scheduler):
        """Test getting analysis data for scheduled reports"""
        report_request = ReportRequest(
            report_type=ReportType.DELAY_ANALYSIS,
            stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
            export_format=ExportFormat.PDF
        )
        
        analysis_data = await report_scheduler._get_analysis_data(report_request)
        
        assert isinstance(analysis_data, dict)
        assert "total_flights" in analysis_data
        assert "confidence_score" in analysis_data


@pytest.mark.asyncio
class TestReportingIntegration:
    """Integration tests for reporting functionality"""
    
    async def test_end_to_end_report_generation(self):
        """Test complete report generation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup components
            report_generator = ReportGenerator(
                template_directory=f"{temp_dir}/templates",
                output_directory=f"{temp_dir}/reports"
            )
            
            # Create report request
            request = ReportRequest(
                report_type=ReportType.DELAY_ANALYSIS,
                stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
                export_format=ExportFormat.PDF,
                airport_codes=["BOM", "DEL"],
                include_charts=True,
                include_recommendations=True
            )
            
            # Sample analysis data
            analysis_data = {
                "total_flights": 500,
                "average_delay": 12.3,
                "hourly_delays": [5, 3, 2, 4, 8, 12, 15, 18, 22, 25, 20, 15, 12, 8, 5, 3, 2, 1, 1, 2, 3, 4, 5, 4],
                "delay_causes": {"Weather": 40, "Traffic": 30, "Mechanical": 20, "Other": 10},
                "confidence_score": 0.88
            }
            
            # Mock the HTML to PDF conversion to avoid external dependencies
            with patch('src.reporting.report_generator.HTML') as mock_html:
                mock_html_instance = Mock()
                mock_html.return_value = mock_html_instance
                mock_html_instance.write_pdf = Mock()
                
                # Generate report
                report_path = report_generator.generate_report(request, analysis_data)
                
                # Verify report was "generated"
                assert report_path is not None
                assert "operations_manager_delay_analysis" in report_path
                mock_html.assert_called_once()
    
    async def test_export_and_report_workflow(self):
        """Test combined export and report generation workflow"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup components
            export_manager = ExportManager(export_directory=f"{temp_dir}/exports")
            
            # Create sample data
            flight_data = pd.DataFrame({
                'flight_number': ['AI101', '6E202', 'SG303', 'UK404'],
                'origin_airport': ['BOM', 'DEL', 'BOM', 'DEL'],
                'destination_airport': ['DEL', 'BOM', 'DEL', 'BOM'],
                'scheduled_departure': pd.to_datetime([
                    '2024-01-01 08:00:00',
                    '2024-01-01 10:00:00',
                    '2024-01-01 12:00:00',
                    '2024-01-01 14:00:00'
                ]),
                'delay_minutes': [15, 5, 20, 0]
            })
            
            # Export data in multiple formats
            csv_result = export_manager.export_flight_data(
                flight_data, ExportFormat.CSV, "test_flights"
            )
            excel_result = export_manager.export_flight_data(
                flight_data, ExportFormat.EXCEL, "test_flights"
            )
            json_result = export_manager.export_flight_data(
                flight_data, ExportFormat.JSON, "test_flights"
            )
            
            # Verify all exports succeeded
            assert csv_result.success is True
            assert excel_result.success is True
            assert json_result.success is True
            
            # Verify files exist
            assert Path(csv_result.file_path).exists()
            assert Path(excel_result.file_path).exists()
            assert Path(json_result.file_path).exists()
            
            # Verify metadata
            assert csv_result.metadata["total_flights"] == 4
            assert excel_result.metadata["total_flights"] == 4
            assert json_result.metadata["total_flights"] == 4