"""
Scheduled Report Generation System
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from croniter import croniter
import json
from pathlib import Path

from .models import ScheduledReport, ReportRequest
from .report_generator import ReportGenerator
from .export_manager import ExportManager

logger = logging.getLogger(__name__)


class ReportScheduler:
    """Manages scheduled report generation"""
    
    def __init__(
        self, 
        report_generator: ReportGenerator,
        export_manager: ExportManager,
        config_file: str = "scheduled_reports.json"
    ):
        self.report_generator = report_generator
        self.export_manager = export_manager
        self.config_file = Path(config_file)
        self.scheduled_reports: Dict[str, ScheduledReport] = {}
        self.running = False
        
        # Load existing scheduled reports
        self._load_scheduled_reports()
    
    def add_scheduled_report(self, scheduled_report: ScheduledReport) -> bool:
        """Add a new scheduled report"""
        try:
            # Validate cron expression
            if not self._validate_cron_expression(scheduled_report.cron_expression):
                raise ValueError(f"Invalid cron expression: {scheduled_report.cron_expression}")
            
            # Calculate next run time
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.cron_expression)
            
            # Add to scheduled reports
            self.scheduled_reports[scheduled_report.schedule_id] = scheduled_report
            
            # Save to file
            self._save_scheduled_reports()
            
            logger.info(f"Added scheduled report: {scheduled_report.schedule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add scheduled report: {str(e)}")
            return False
    
    def remove_scheduled_report(self, schedule_id: str) -> bool:
        """Remove a scheduled report"""
        try:
            if schedule_id in self.scheduled_reports:
                del self.scheduled_reports[schedule_id]
                self._save_scheduled_reports()
                logger.info(f"Removed scheduled report: {schedule_id}")
                return True
            else:
                logger.warning(f"Scheduled report not found: {schedule_id}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to remove scheduled report: {str(e)}")
            return False
    
    def update_scheduled_report(self, scheduled_report: ScheduledReport) -> bool:
        """Update an existing scheduled report"""
        try:
            if scheduled_report.schedule_id not in self.scheduled_reports:
                logger.warning(f"Scheduled report not found: {scheduled_report.schedule_id}")
                return False
            
            # Validate cron expression
            if not self._validate_cron_expression(scheduled_report.cron_expression):
                raise ValueError(f"Invalid cron expression: {scheduled_report.cron_expression}")
            
            # Update next run time
            scheduled_report.next_run = self._calculate_next_run(scheduled_report.cron_expression)
            
            # Update scheduled report
            self.scheduled_reports[scheduled_report.schedule_id] = scheduled_report
            
            # Save to file
            self._save_scheduled_reports()
            
            logger.info(f"Updated scheduled report: {scheduled_report.schedule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update scheduled report: {str(e)}")
            return False
    
    def get_scheduled_reports(self) -> List[ScheduledReport]:
        """Get all scheduled reports"""
        return list(self.scheduled_reports.values())
    
    def get_scheduled_report(self, schedule_id: str) -> Optional[ScheduledReport]:
        """Get a specific scheduled report"""
        return self.scheduled_reports.get(schedule_id)
    
    async def start_scheduler(self):
        """Start the report scheduler"""
        self.running = True
        logger.info("Report scheduler started")
        
        while self.running:
            try:
                await self._check_and_run_reports()
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
                await asyncio.sleep(60)
    
    def stop_scheduler(self):
        """Stop the report scheduler"""
        self.running = False
        logger.info("Report scheduler stopped")
    
    async def _check_and_run_reports(self):
        """Check for reports that need to be run"""
        current_time = datetime.now()
        
        for schedule_id, scheduled_report in self.scheduled_reports.items():
            if (scheduled_report.is_active and 
                scheduled_report.next_run and 
                current_time >= scheduled_report.next_run):
                
                try:
                    await self._run_scheduled_report(scheduled_report)
                    
                    # Update last run and calculate next run
                    scheduled_report.last_run = current_time
                    scheduled_report.next_run = self._calculate_next_run(
                        scheduled_report.cron_expression, 
                        current_time
                    )
                    
                    # Save updated schedule
                    self._save_scheduled_reports()
                    
                except Exception as e:
                    logger.error(f"Failed to run scheduled report {schedule_id}: {str(e)}")
    
    async def _run_scheduled_report(self, scheduled_report: ScheduledReport):
        """Run a scheduled report"""
        logger.info(f"Running scheduled report: {scheduled_report.schedule_id}")
        
        try:
            # This would need to be connected to your analysis system
            # For now, we'll create mock analysis data
            analysis_data = await self._get_analysis_data(scheduled_report.report_request)
            
            # Generate report
            report_path = self.report_generator.generate_report(
                scheduled_report.report_request,
                analysis_data
            )
            
            # Send to recipients if specified
            if scheduled_report.recipients:
                await self._send_report_to_recipients(report_path, scheduled_report.recipients)
            
            logger.info(f"Successfully generated scheduled report: {report_path}")
            
        except Exception as e:
            logger.error(f"Failed to run scheduled report {scheduled_report.schedule_id}: {str(e)}")
            raise
    
    async def _get_analysis_data(self, request: ReportRequest) -> Dict:
        """Get analysis data for the report (placeholder)"""
        # This would integrate with your actual analysis engines
        # For now, return mock data
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
            "confidence_score": 0.92
        }
    
    async def _send_report_to_recipients(self, report_path: str, recipients: List[str]):
        """Send report to recipients (placeholder)"""
        # This would integrate with email service or notification system
        logger.info(f"Would send report {report_path} to recipients: {recipients}")
        # Implementation would depend on your notification system
    
    def _validate_cron_expression(self, cron_expression: str) -> bool:
        """Validate cron expression"""
        try:
            croniter(cron_expression)
            return True
        except Exception:
            return False
    
    def _calculate_next_run(self, cron_expression: str, base_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time based on cron expression"""
        if base_time is None:
            base_time = datetime.now()
        
        cron = croniter(cron_expression, base_time)
        return cron.get_next(datetime)
    
    def _load_scheduled_reports(self):
        """Load scheduled reports from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                for schedule_data in data:
                    scheduled_report = ScheduledReport(**schedule_data)
                    self.scheduled_reports[scheduled_report.schedule_id] = scheduled_report
                    
                logger.info(f"Loaded {len(self.scheduled_reports)} scheduled reports")
        except Exception as e:
            logger.error(f"Failed to load scheduled reports: {str(e)}")
    
    def _save_scheduled_reports(self):
        """Save scheduled reports to file"""
        try:
            data = []
            for scheduled_report in self.scheduled_reports.values():
                data.append(scheduled_report.model_dump())
            
            with open(self.config_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save scheduled reports: {str(e)}")
    
    def create_daily_report_schedule(
        self,
        name: str,
        report_request: ReportRequest,
        hour: int = 8,
        recipients: Optional[List[str]] = None
    ) -> ScheduledReport:
        """Create a daily report schedule"""
        schedule_id = f"daily_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cron_expression = f"0 {hour} * * *"  # Daily at specified hour
        
        return ScheduledReport(
            schedule_id=schedule_id,
            name=name,
            description=f"Daily {name} report generated at {hour}:00",
            report_request=report_request,
            cron_expression=cron_expression,
            recipients=recipients or [],
            is_active=True
        )
    
    def create_weekly_report_schedule(
        self,
        name: str,
        report_request: ReportRequest,
        day_of_week: int = 1,  # Monday
        hour: int = 8,
        recipients: Optional[List[str]] = None
    ) -> ScheduledReport:
        """Create a weekly report schedule"""
        schedule_id = f"weekly_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        cron_expression = f"0 {hour} * * {day_of_week}"  # Weekly on specified day
        
        return ScheduledReport(
            schedule_id=schedule_id,
            name=name,
            description=f"Weekly {name} report generated on day {day_of_week} at {hour}:00",
            report_request=report_request,
            cron_expression=cron_expression,
            recipients=recipients or [],
            is_active=True
        )