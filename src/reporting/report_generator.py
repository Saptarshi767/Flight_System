"""
PDF Report Generator with charts and insights
"""

import io
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from jinja2 import Environment, FileSystemLoader, Template
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

from .models import (
    ReportRequest, ReportMetadata, ReportTemplate, 
    ReportType, StakeholderType, ExportFormat
)
from .templates import ReportTemplates

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates PDF reports with charts and insights"""
    
    def __init__(self, template_directory: str = "src/reporting/templates", output_directory: str = "reports"):
        self.template_directory = Path(template_directory)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(exist_ok=True)
        
        # Create templates directory if it doesn't exist
        self.template_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_directory)),
            autoescape=True
        )
        
        # Create default HTML templates if they don't exist
        self._create_default_templates()
    
    def generate_report(
        self,
        request: ReportRequest,
        analysis_data: Dict[str, Any],
        flight_data: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a comprehensive report based on the request
        
        Args:
            request: Report generation request
            analysis_data: Analysis results data
            flight_data: Raw flight data (optional)
            
        Returns:
            Path to generated report file
        """
        try:
            # Get appropriate template
            template = ReportTemplates.get_template(
                request.stakeholder_type, 
                request.report_type
            )
            
            # Generate report metadata
            metadata = self._create_metadata(request, analysis_data, flight_data)
            
            # Generate charts
            charts = self._generate_charts(analysis_data, template, request)
            
            # Prepare report data
            report_data = {
                "metadata": metadata,
                "analysis_data": analysis_data,
                "charts": charts,
                "template": template,
                "request": request,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if request.export_format == ExportFormat.PDF:
                return self._generate_pdf_report(report_data)
            else:
                raise ValueError(f"Unsupported report format: {request.export_format}")
                
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def _create_metadata(
        self, 
        request: ReportRequest, 
        analysis_data: Dict[str, Any],
        flight_data: Optional[pd.DataFrame]
    ) -> ReportMetadata:
        """Create report metadata"""
        total_flights = 0
        if flight_data is not None:
            total_flights = len(flight_data)
        elif "total_flights" in analysis_data:
            total_flights = analysis_data["total_flights"]
        
        return ReportMetadata(
            report_id=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type=request.report_type,
            stakeholder_type=request.stakeholder_type,
            generated_at=datetime.now(),
            data_period=request.date_range,
            airports_included=request.airport_codes,
            total_flights_analyzed=total_flights,
            confidence_score=analysis_data.get("confidence_score")
        )
    
    def _generate_charts(
        self, 
        analysis_data: Dict[str, Any], 
        template: ReportTemplate,
        request: ReportRequest
    ) -> Dict[str, str]:
        """Generate charts as base64 encoded images"""
        charts = {}
        
        if not request.include_charts:
            return charts
        
        try:
            # Generate delay analysis charts
            if request.report_type in [ReportType.DELAY_ANALYSIS, ReportType.COMPREHENSIVE]:
                charts.update(self._generate_delay_charts(analysis_data))
            
            # Generate congestion analysis charts
            if request.report_type in [ReportType.CONGESTION_ANALYSIS, ReportType.COMPREHENSIVE]:
                charts.update(self._generate_congestion_charts(analysis_data))
            
            # Generate schedule impact charts
            if request.report_type in [ReportType.SCHEDULE_IMPACT, ReportType.COMPREHENSIVE]:
                charts.update(self._generate_schedule_impact_charts(analysis_data))
            
            # Generate cascading impact charts
            if request.report_type in [ReportType.CASCADING_IMPACT, ReportType.COMPREHENSIVE]:
                charts.update(self._generate_cascading_impact_charts(analysis_data))
                
        except Exception as e:
            logger.error(f"Chart generation failed: {str(e)}")
            # Continue without charts rather than failing completely
        
        return charts
    
    def _generate_delay_charts(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate delay analysis charts"""
        charts = {}
        
        # Hourly delay pattern chart
        if "hourly_delays" in analysis_data:
            fig = px.bar(
                x=list(range(24)),
                y=analysis_data["hourly_delays"],
                title="Average Delays by Hour of Day",
                labels={"x": "Hour", "y": "Average Delay (minutes)"}
            )
            charts["hourly_delays"] = self._fig_to_base64(fig)
        
        # Delay causes pie chart
        if "delay_causes" in analysis_data:
            causes = analysis_data["delay_causes"]
            fig = px.pie(
                values=list(causes.values()),
                names=list(causes.keys()),
                title="Distribution of Delay Causes"
            )
            charts["delay_causes"] = self._fig_to_base64(fig)
        
        return charts
    
    def _generate_congestion_charts(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate congestion analysis charts"""
        charts = {}
        
        # Traffic density heatmap
        if "traffic_density" in analysis_data:
            density_data = analysis_data["traffic_density"]
            fig = px.imshow(
                density_data,
                title="Traffic Density Heatmap",
                labels=dict(x="Hour", y="Day of Week", color="Flight Count")
            )
            charts["traffic_density"] = self._fig_to_base64(fig)
        
        return charts
    
    def _generate_schedule_impact_charts(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate schedule impact charts"""
        charts = {}
        
        # Schedule optimization chart
        if "optimal_times" in analysis_data:
            times = analysis_data["optimal_times"]
            fig = px.scatter(
                x=times.get("scheduled_times", []),
                y=times.get("delays", []),
                title="Schedule Time vs Delay Analysis",
                labels={"x": "Scheduled Time", "y": "Average Delay (minutes)"}
            )
            charts["schedule_optimization"] = self._fig_to_base64(fig)
        
        return charts
    
    def _generate_cascading_impact_charts(self, analysis_data: Dict[str, Any]) -> Dict[str, str]:
        """Generate cascading impact charts"""
        charts = {}
        
        # Network impact visualization
        if "network_impact" in analysis_data:
            impact_data = analysis_data["network_impact"]
            fig = px.bar(
                x=impact_data.get("flights", []),
                y=impact_data.get("impact_scores", []),
                title="Flight Network Impact Scores",
                labels={"x": "Flight", "y": "Impact Score"}
            )
            charts["network_impact"] = self._fig_to_base64(fig)
        
        return charts
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 encoded image"""
        img_bytes = fig.to_image(format="png", width=800, height=600)
        img_base64 = base64.b64encode(img_bytes).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        """Generate PDF report using ReportLab"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{report_data['request'].stakeholder_type.value}_{report_data['request'].report_type.value}_{timestamp}.pdf"
        output_path = self.output_directory / filename
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        story.append(Paragraph(report_data['template'].name, title_style))
        story.append(Spacer(1, 20))
        
        # Add metadata section
        metadata = report_data['metadata']
        story.append(Paragraph("Report Information", styles['Heading2']))
        
        metadata_data = [
            ['Report ID:', metadata.report_id],
            ['Generated:', metadata.generated_at.strftime("%Y-%m-%d %H:%M:%S")],
            ['Stakeholder:', metadata.stakeholder_type.value.replace('_', ' ').title()],
            ['Report Type:', metadata.report_type.value.replace('_', ' ').title()],
            ['Airports:', ', '.join(metadata.airports_included)],
            ['Total Flights:', str(metadata.total_flights_analyzed)],
        ]
        
        if metadata.confidence_score:
            metadata_data.append(['Confidence Score:', f"{metadata.confidence_score:.2f}"])
        
        metadata_table = Table(metadata_data, colWidths=[2*inch, 4*inch])
        metadata_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (1, 0), (1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metadata_table)
        story.append(Spacer(1, 20))
        
        # Add sections from template
        for section in report_data['template'].sections:
            story.append(Paragraph(section.title, styles['Heading2']))
            story.append(Paragraph(section.content, styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Add charts if available
            for chart_config in section.charts:
                if chart_config.data_source in report_data['charts']:
                    story.append(Paragraph(chart_config.title, styles['Heading3']))
                    
                    # For now, add a placeholder for charts
                    # In a full implementation, you would convert the base64 image to ReportLab Image
                    chart_placeholder = Paragraph(
                        f"[Chart: {chart_config.title}]<br/>Chart data available in base64 format",
                        styles['Italic']
                    )
                    story.append(chart_placeholder)
                    story.append(Spacer(1, 12))
            
            # Add recommendations if available
            if section.recommendations:
                story.append(Paragraph("Recommendations:", styles['Heading3']))
                for i, rec in enumerate(section.recommendations, 1):
                    story.append(Paragraph(f"{i}. {rec}", styles['Normal']))
                story.append(Spacer(1, 12))
        
        # Add analysis summary
        if report_data['analysis_data']:
            story.append(PageBreak())
            story.append(Paragraph("Analysis Summary", styles['Heading2']))
            
            analysis_data = report_data['analysis_data']
            summary_data = []
            
            for key, value in analysis_data.items():
                if key not in ['hourly_delays', 'delay_causes', 'traffic_density', 'optimal_times', 'network_impact']:
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value_str = f"{value:.2f}"
                        else:
                            value_str = str(value)
                    else:
                        value_str = str(value)
                    
                    summary_data.append([
                        key.replace('_', ' ').title() + ':',
                        value_str
                    ])
            
            if summary_data:
                summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
                summary_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(summary_table)
        
        # Build PDF
        doc.build(story)
        
        return str(output_path)
    
    def _create_default_templates(self):
        """Create default HTML templates"""
        # Generic report template
        generic_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ template.name }}</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { text-align: center; margin-bottom: 30px; }
        .section { margin-bottom: 30px; }
        .chart { text-align: center; margin: 20px 0; }
        .chart img { max-width: 100%; height: auto; }
        .metadata { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }
        .recommendations { background-color: #e8f4fd; padding: 15px; border-radius: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ template.name }}</h1>
        <p>Generated on: {{ generated_at }}</p>
        <p>Report ID: {{ metadata.report_id }}</p>
    </div>
    
    <div class="metadata">
        <h2>Report Metadata</h2>
        <p><strong>Stakeholder:</strong> {{ request.stakeholder_type.value.replace('_', ' ').title() }}</p>
        <p><strong>Report Type:</strong> {{ request.report_type.value.replace('_', ' ').title() }}</p>
        <p><strong>Airports:</strong> {{ metadata.airports_included | join(', ') }}</p>
        <p><strong>Total Flights Analyzed:</strong> {{ metadata.total_flights_analyzed }}</p>
        {% if metadata.confidence_score %}
        <p><strong>Confidence Score:</strong> {{ "%.2f"|format(metadata.confidence_score) }}</p>
        {% endif %}
    </div>
    
    {% for section in template.sections %}
    <div class="section">
        <h2>{{ section.title }}</h2>
        <p>{{ section.content }}</p>
        
        {% for chart_config in section.charts %}
        {% if charts[chart_config.data_source] %}
        <div class="chart">
            <h3>{{ chart_config.title }}</h3>
            <img src="{{ charts[chart_config.data_source] }}" alt="{{ chart_config.title }}">
        </div>
        {% endif %}
        {% endfor %}
        
        {% if section.recommendations %}
        <div class="recommendations">
            <h3>Recommendations</h3>
            <ul>
            {% for rec in section.recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </div>
        {% endif %}
    </div>
    {% endfor %}
    
    {% if analysis_data %}
    <div class="section">
        <h2>Analysis Summary</h2>
        {% for key, value in analysis_data.items() %}
        {% if key not in ['hourly_delays', 'delay_causes', 'traffic_density', 'optimal_times', 'network_impact'] %}
        <p><strong>{{ key.replace('_', ' ').title() }}:</strong> {{ value }}</p>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
        """
        
        generic_path = self.template_directory / "generic_report.html"
        with open(generic_path, 'w') as f:
            f.write(generic_template)
    
    def _add_chart_to_story(self, story: List, chart_data: str, title: str, styles):
        """Add chart to PDF story (placeholder implementation)"""
        # In a full implementation, you would:
        # 1. Decode the base64 image
        # 2. Create a ReportLab Image object
        # 3. Add it to the story
        
        # For now, add a text placeholder
        story.append(Paragraph(title, styles['Heading3']))
        story.append(Paragraph("[Chart would be displayed here]", styles['Italic']))
        story.append(Spacer(1, 12))