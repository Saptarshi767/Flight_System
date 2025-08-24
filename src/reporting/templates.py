"""
Report templates for different stakeholders
"""

from typing import Dict, List
from .models import (
    ReportTemplate, ReportSection, ChartConfig, 
    StakeholderType, ReportType
)


class ReportTemplates:
    """Manages report templates for different stakeholders"""
    
    @staticmethod
    def get_template(stakeholder_type: StakeholderType, report_type: ReportType) -> ReportTemplate:
        """Get appropriate template based on stakeholder and report type"""
        template_key = f"{stakeholder_type.value}_{report_type.value}"
        
        if hasattr(ReportTemplates, f"_get_{template_key}"):
            return getattr(ReportTemplates, f"_get_{template_key}")()
        else:
            # Fallback to generic template
            return ReportTemplates._get_generic_template(stakeholder_type, report_type)
    
    @staticmethod
    def _get_operations_manager_delay_analysis() -> ReportTemplate:
        """Template for operations manager delay analysis report"""
        return ReportTemplate(
            template_id="ops_mgr_delay",
            name="Operations Manager - Delay Analysis Report",
            description="Comprehensive delay analysis for operations management",
            stakeholder_type=StakeholderType.OPERATIONS_MANAGER,
            sections=[
                ReportSection(
                    section_id="executive_summary",
                    title="Executive Summary",
                    content="Key delay metrics and operational impact overview",
                    order=1
                ),
                ReportSection(
                    section_id="delay_patterns",
                    title="Delay Patterns Analysis",
                    content="Detailed analysis of delay patterns by time, route, and cause",
                    charts=[
                        ChartConfig(
                            chart_type="bar",
                            title="Average Delays by Hour",
                            data_source="hourly_delays",
                            x_axis="hour",
                            y_axis="avg_delay_minutes"
                        ),
                        ChartConfig(
                            chart_type="pie",
                            title="Delay Causes Distribution",
                            data_source="delay_causes",
                            x_axis="cause",
                            y_axis="count"
                        )
                    ],
                    order=2
                ),
                ReportSection(
                    section_id="recommendations",
                    title="Operational Recommendations",
                    content="Actionable recommendations to reduce delays",
                    order=3
                )
            ],
            styling={
                "color_scheme": "professional",
                "font_family": "Arial",
                "include_logo": True
            }
        )
    
    @staticmethod
    def _get_air_traffic_controller_congestion_analysis() -> ReportTemplate:
        """Template for air traffic controller congestion analysis"""
        return ReportTemplate(
            template_id="atc_congestion",
            name="Air Traffic Controller - Congestion Analysis Report",
            description="Detailed congestion patterns for traffic management",
            stakeholder_type=StakeholderType.AIR_TRAFFIC_CONTROLLER,
            sections=[
                ReportSection(
                    section_id="peak_hours",
                    title="Peak Hours Analysis",
                    content="Identification of busiest time slots and congestion patterns",
                    charts=[
                        ChartConfig(
                            chart_type="heatmap",
                            title="Hourly Traffic Density",
                            data_source="hourly_traffic",
                            x_axis="hour",
                            y_axis="day_of_week"
                        ),
                        ChartConfig(
                            chart_type="line",
                            title="Daily Traffic Patterns",
                            data_source="daily_patterns",
                            x_axis="time",
                            y_axis="flight_count"
                        )
                    ],
                    order=1
                ),
                ReportSection(
                    section_id="runway_utilization",
                    title="Runway Utilization",
                    content="Analysis of runway capacity and utilization rates",
                    order=2
                ),
                ReportSection(
                    section_id="traffic_recommendations",
                    title="Traffic Management Recommendations",
                    content="Specific recommendations for managing peak traffic",
                    order=3
                )
            ],
            styling={
                "color_scheme": "technical",
                "font_family": "Calibri",
                "include_technical_details": True
            }
        )
    
    @staticmethod
    def _get_executive_comprehensive() -> ReportTemplate:
        """Template for executive comprehensive report"""
        return ReportTemplate(
            template_id="exec_comprehensive",
            name="Executive Comprehensive Report",
            description="High-level overview of all flight scheduling metrics",
            stakeholder_type=StakeholderType.EXECUTIVE,
            sections=[
                ReportSection(
                    section_id="key_metrics",
                    title="Key Performance Indicators",
                    content="High-level KPIs and performance metrics",
                    charts=[
                        ChartConfig(
                            chart_type="gauge",
                            title="On-Time Performance",
                            data_source="otp_metrics",
                            x_axis="metric",
                            y_axis="percentage"
                        ),
                        ChartConfig(
                            chart_type="trend",
                            title="Monthly Performance Trends",
                            data_source="monthly_trends",
                            x_axis="month",
                            y_axis="performance_score"
                        )
                    ],
                    order=1
                ),
                ReportSection(
                    section_id="financial_impact",
                    title="Financial Impact Analysis",
                    content="Cost implications of delays and scheduling inefficiencies",
                    order=2
                ),
                ReportSection(
                    section_id="strategic_recommendations",
                    title="Strategic Recommendations",
                    content="High-level strategic recommendations for improvement",
                    order=3
                )
            ],
            styling={
                "color_scheme": "executive",
                "font_family": "Times New Roman",
                "include_executive_summary": True,
                "high_level_focus": True
            }
        )
    
    @staticmethod
    def _get_flight_scheduler_schedule_impact() -> ReportTemplate:
        """Template for flight scheduler schedule impact report"""
        return ReportTemplate(
            template_id="scheduler_impact",
            name="Flight Scheduler - Schedule Impact Report",
            description="Detailed analysis of schedule changes and their impacts",
            stakeholder_type=StakeholderType.FLIGHT_SCHEDULER,
            sections=[
                ReportSection(
                    section_id="optimal_times",
                    title="Optimal Scheduling Windows",
                    content="Best time slots for scheduling based on historical data",
                    charts=[
                        ChartConfig(
                            chart_type="scatter",
                            title="Delay vs Schedule Time",
                            data_source="schedule_delays",
                            x_axis="scheduled_time",
                            y_axis="delay_minutes"
                        )
                    ],
                    order=1
                ),
                ReportSection(
                    section_id="impact_modeling",
                    title="Schedule Change Impact Modeling",
                    content="Predicted impacts of proposed schedule changes",
                    order=2
                ),
                ReportSection(
                    section_id="scheduling_recommendations",
                    title="Scheduling Recommendations",
                    content="Specific recommendations for schedule optimization",
                    order=3
                )
            ],
            styling={
                "color_scheme": "analytical",
                "font_family": "Arial",
                "include_detailed_analysis": True
            }
        )
    
    @staticmethod
    def _get_network_operations_cascading_impact() -> ReportTemplate:
        """Template for network operations cascading impact report"""
        return ReportTemplate(
            template_id="network_cascading",
            name="Network Operations - Cascading Impact Report",
            description="Analysis of flight network effects and cascading impacts",
            stakeholder_type=StakeholderType.NETWORK_OPERATIONS,
            sections=[
                ReportSection(
                    section_id="network_analysis",
                    title="Flight Network Analysis",
                    content="Analysis of flight connections and network structure",
                    charts=[
                        ChartConfig(
                            chart_type="network",
                            title="Flight Connection Network",
                            data_source="flight_network",
                            x_axis="origin",
                            y_axis="destination"
                        )
                    ],
                    order=1
                ),
                ReportSection(
                    section_id="critical_flights",
                    title="Critical Flight Identification",
                    content="Flights with highest cascading impact potential",
                    order=2
                ),
                ReportSection(
                    section_id="mitigation_strategies",
                    title="Impact Mitigation Strategies",
                    content="Strategies to minimize cascading delays",
                    order=3
                )
            ],
            styling={
                "color_scheme": "network",
                "font_family": "Arial",
                "include_network_diagrams": True
            }
        )
    
    @staticmethod
    def _get_generic_template(stakeholder_type: StakeholderType, report_type: ReportType) -> ReportTemplate:
        """Generic template fallback"""
        return ReportTemplate(
            template_id=f"generic_{stakeholder_type.value}_{report_type.value}",
            name=f"Generic {stakeholder_type.value.replace('_', ' ').title()} - {report_type.value.replace('_', ' ').title()} Report",
            description=f"Generic report template for {stakeholder_type.value} - {report_type.value}",
            stakeholder_type=stakeholder_type,
            sections=[
                ReportSection(
                    section_id="overview",
                    title="Overview",
                    content="General overview of the analysis",
                    order=1
                ),
                ReportSection(
                    section_id="analysis",
                    title="Analysis Results",
                    content="Detailed analysis results and findings",
                    order=2
                ),
                ReportSection(
                    section_id="recommendations",
                    title="Recommendations",
                    content="Recommendations based on the analysis",
                    order=3
                )
            ],
            styling={
                "color_scheme": "default",
                "font_family": "Arial"
            }
        )
    
    @staticmethod
    def get_available_templates() -> Dict[str, List[str]]:
        """Get list of available templates by stakeholder type"""
        return {
            "operations_manager": ["delay_analysis", "comprehensive"],
            "air_traffic_controller": ["congestion_analysis", "delay_analysis"],
            "flight_scheduler": ["schedule_impact", "delay_analysis"],
            "network_operations": ["cascading_impact", "comprehensive"],
            "executive": ["comprehensive", "executive_summary"],
            "technical": ["delay_analysis", "congestion_analysis", "schedule_impact", "cascading_impact"]
        }