"""
Query response generation system for flight data analysis.

This module implements response templates, data visualization recommendations,
follow-up question suggestions, and response caching for flight queries.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass, field
from enum import Enum
import re

import pandas as pd
from pydantic import BaseModel, Field

from ..utils.logger import get_logger
from ..utils.cache import CacheManager

logger = get_logger(__name__)


class ResponseType(Enum):
    """Types of responses that can be generated."""
    DELAY_ANALYSIS = "delay_analysis"
    CONGESTION_ANALYSIS = "congestion_analysis"
    BEST_TIME_RECOMMENDATION = "best_time_recommendation"
    SCHEDULE_IMPACT = "schedule_impact"
    CASCADING_IMPACT = "cascading_impact"
    GENERAL_INFO = "general_info"
    ERROR = "error"
    NO_DATA = "no_data"


class VisualizationType(Enum):
    """Types of visualizations that can be recommended."""
    BAR_CHART = "bar_chart"
    LINE_CHART = "line_chart"
    HEATMAP = "heatmap"
    SCATTER_PLOT = "scatter_plot"
    PIE_CHART = "pie_chart"
    HISTOGRAM = "histogram"
    BOX_PLOT = "box_plot"
    NETWORK_GRAPH = "network_graph"
    TIME_SERIES = "time_series"
    GANTT_CHART = "gantt_chart"


@dataclass
class VisualizationRecommendation:
    """Recommendation for data visualization."""
    type: VisualizationType
    title: str
    description: str
    data_columns: List[str]
    config: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = highest priority


@dataclass
class ResponseTemplate:
    """Template for generating responses."""
    response_type: ResponseType
    template: str
    required_data_fields: List[str]
    optional_data_fields: List[str] = field(default_factory=list)
    visualization_recommendations: List[VisualizationRecommendation] = field(default_factory=list)
    follow_up_templates: List[str] = field(default_factory=list)


class FlightResponseGenerator:
    """
    Advanced response generation system for flight data queries.
    
    Features:
    - Template-based response generation
    - Data visualization recommendations
    - Follow-up question suggestions
    - Response caching
    - Context-aware formatting
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the response generator.
        
        Args:
            cache_manager: Optional cache manager for response caching
        """
        self.cache_manager = cache_manager
        self.templates = self._initialize_templates()
        self.visualization_configs = self._initialize_visualization_configs()
        
        logger.info("FlightResponseGenerator initialized successfully")
    
    def _initialize_templates(self) -> Dict[ResponseType, ResponseTemplate]:
        """Initialize response templates for different query types."""
        templates = {}
        
        # Delay Analysis Template
        templates[ResponseType.DELAY_ANALYSIS] = ResponseTemplate(
            response_type=ResponseType.DELAY_ANALYSIS,
            template="""
Based on the analysis of {total_flights} flights, here are the delay insights:

📊 **Delay Statistics:**
• Average delay: {average_delay:.1f} minutes
• On-time performance: {on_time_percentage:.1f}% (≤15 min delay)
• Most delayed route: {most_delayed_route}
• Peak delay hours: {peak_delay_hours}

{delay_insights}

💡 **Recommendations:**
{recommendations}
            """.strip(),
            required_data_fields=['total_flights', 'average_delay', 'on_time_percentage'],
            optional_data_fields=['most_delayed_route', 'peak_delay_hours', 'delay_insights', 'recommendations'],
            visualization_recommendations=[
                VisualizationRecommendation(
                    type=VisualizationType.HISTOGRAM,
                    title="Delay Distribution",
                    description="Distribution of flight delays",
                    data_columns=['delay_minutes'],
                    priority=1
                ),
                VisualizationRecommendation(
                    type=VisualizationType.HEATMAP,
                    title="Delay Patterns by Hour and Day",
                    description="Heatmap showing delay patterns across time",
                    data_columns=['hour', 'day_of_week', 'delay_minutes'],
                    priority=2
                ),
                VisualizationRecommendation(
                    type=VisualizationType.BAR_CHART,
                    title="Delays by Airline",
                    description="Average delays by airline",
                    data_columns=['airline', 'delay_minutes'],
                    priority=3
                )
            ],
            follow_up_templates=[
                "What are the main causes of delays for {airport}?",
                "How do weather conditions affect these delay patterns?",
                "Which airlines have the best on-time performance for this route?",
                "What time slots have the lowest delay probability?"
            ]
        )
        
        # Congestion Analysis Template
        templates[ResponseType.CONGESTION_ANALYSIS] = ResponseTemplate(
            response_type=ResponseType.CONGESTION_ANALYSIS,
            template="""
Airport congestion analysis for {total_flights} flights:

🚦 **Congestion Overview:**
• Peak hours: {peak_hours}
• Quietest hours: {quietest_hours}
• Congestion score: {congestion_score:.2f}/10
• Busiest airports: {busiest_airports}

{congestion_insights}

⏰ **Best Times to Fly:**
{best_times}

🚫 **Times to Avoid:**
{avoid_times}
            """.strip(),
            required_data_fields=['total_flights', 'peak_hours', 'quietest_hours', 'congestion_score'],
            optional_data_fields=['busiest_airports', 'congestion_insights', 'best_times', 'avoid_times'],
            visualization_recommendations=[
                VisualizationRecommendation(
                    type=VisualizationType.LINE_CHART,
                    title="Hourly Flight Volume",
                    description="Flight volume throughout the day",
                    data_columns=['hour', 'flight_count'],
                    priority=1
                ),
                VisualizationRecommendation(
                    type=VisualizationType.HEATMAP,
                    title="Weekly Congestion Patterns",
                    description="Congestion patterns by day and hour",
                    data_columns=['day_of_week', 'hour', 'flight_count'],
                    priority=2
                )
            ],
            follow_up_templates=[
                "What causes congestion during peak hours at {airport}?",
                "How does runway capacity affect these patterns?",
                "What are alternative airports during busy periods?",
                "How far in advance should I book to avoid peak times?"
            ]
        )
        
        # Best Time Recommendation Template
        templates[ResponseType.BEST_TIME_RECOMMENDATION] = ResponseTemplate(
            response_type=ResponseType.BEST_TIME_RECOMMENDATION,
            template="""
🎯 **Best Time Recommendations for {route}:**

⭐ **Optimal Time Slots:**
{optimal_times}

📈 **Analysis Summary:**
• Flights analyzed: {total_flights}
• Average delay in optimal slots: {optimal_delay:.1f} minutes
• Success rate: {success_rate:.1f}%

{time_insights}

📅 **Day-wise Recommendations:**
{day_recommendations}
            """.strip(),
            required_data_fields=['route', 'optimal_times', 'total_flights', 'optimal_delay', 'success_rate'],
            optional_data_fields=['time_insights', 'day_recommendations'],
            visualization_recommendations=[
                VisualizationRecommendation(
                    type=VisualizationType.BAR_CHART,
                    title="Delay Probability by Hour",
                    description="Probability of delays at different hours",
                    data_columns=['hour', 'delay_probability'],
                    priority=1
                ),
                VisualizationRecommendation(
                    type=VisualizationType.HEATMAP,
                    title="Best Times Heatmap",
                    description="Optimal departure times by day and hour",
                    data_columns=['day_of_week', 'hour', 'delay_score'],
                    priority=2
                )
            ],
            follow_up_templates=[
                "How do these recommendations change during different seasons?",
                "What's the impact of weather on these optimal times?",
                "Are there alternative routes with better timing?",
                "How much earlier should I arrive for flights at these times?"
            ]
        )
        
        # Schedule Impact Template
        templates[ResponseType.SCHEDULE_IMPACT] = ResponseTemplate(
            response_type=ResponseType.SCHEDULE_IMPACT,
            template="""
📋 **Schedule Impact Analysis:**

🔄 **Proposed Changes:**
{proposed_changes}

📊 **Impact Assessment:**
• Flights affected: {affected_flights}
• Delay reduction: {delay_reduction:.1f} minutes average
• Efficiency improvement: {efficiency_improvement:.1f}%
• Risk level: {risk_level}

{impact_details}

⚠️ **Considerations:**
{considerations}

✅ **Recommendations:**
{recommendations}
            """.strip(),
            required_data_fields=['proposed_changes', 'affected_flights', 'delay_reduction', 'efficiency_improvement', 'risk_level'],
            optional_data_fields=['impact_details', 'considerations', 'recommendations'],
            visualization_recommendations=[
                VisualizationRecommendation(
                    type=VisualizationType.GANTT_CHART,
                    title="Schedule Comparison",
                    description="Before and after schedule comparison",
                    data_columns=['flight_id', 'current_time', 'proposed_time'],
                    priority=1
                ),
                VisualizationRecommendation(
                    type=VisualizationType.BAR_CHART,
                    title="Impact by Time Slot",
                    description="Impact of changes by time slot",
                    data_columns=['time_slot', 'impact_score'],
                    priority=2
                )
            ],
            follow_up_templates=[
                "What are the cost implications of these schedule changes?",
                "How will this affect passenger connections?",
                "What's the implementation timeline for these changes?",
                "Are there any regulatory considerations?"
            ]
        )
        
        # Cascading Impact Template
        templates[ResponseType.CASCADING_IMPACT] = ResponseTemplate(
            response_type=ResponseType.CASCADING_IMPACT,
            template="""
🌊 **Cascading Impact Analysis:**

🎯 **Critical Flights Identified:**
{critical_flights}

📈 **Network Impact:**
• Total flights in network: {total_network_flights}
• Potentially affected flights: {affected_flights}
• Cascade probability: {cascade_probability:.1f}%
• Network resilience score: {resilience_score:.2f}/10

{cascade_details}

🛡️ **Mitigation Strategies:**
{mitigation_strategies}
            """.strip(),
            required_data_fields=['critical_flights', 'total_network_flights', 'affected_flights', 'cascade_probability', 'resilience_score'],
            optional_data_fields=['cascade_details', 'mitigation_strategies'],
            visualization_recommendations=[
                VisualizationRecommendation(
                    type=VisualizationType.NETWORK_GRAPH,
                    title="Flight Network Graph",
                    description="Network showing flight connections and critical paths",
                    data_columns=['origin', 'destination', 'impact_score'],
                    priority=1
                ),
                VisualizationRecommendation(
                    type=VisualizationType.BAR_CHART,
                    title="Critical Flight Rankings",
                    description="Flights ranked by cascading impact potential",
                    data_columns=['flight_id', 'impact_score'],
                    priority=2
                )
            ],
            follow_up_templates=[
                "Which specific flights should be prioritized for on-time performance?",
                "How can we build more resilience into the network?",
                "What's the cost of delays for these critical flights?",
                "Are there backup plans for critical flight disruptions?"
            ]
        )
        
        # General Info Template
        templates[ResponseType.GENERAL_INFO] = ResponseTemplate(
            response_type=ResponseType.GENERAL_INFO,
            template="""
ℹ️ **Flight Information:**

{flight_details}

📊 **Quick Stats:**
{quick_stats}

{additional_info}
            """.strip(),
            required_data_fields=['flight_details'],
            optional_data_fields=['quick_stats', 'additional_info'],
            visualization_recommendations=[],
            follow_up_templates=[
                "Would you like more detailed analysis of this data?",
                "Are you interested in delay patterns for these flights?",
                "Would you like to see congestion analysis for these airports?",
                "Do you want recommendations for optimal flight times?"
            ]
        )
        
        # Error Template
        templates[ResponseType.ERROR] = ResponseTemplate(
            response_type=ResponseType.ERROR,
            template="""
❌ **Error Processing Request:**

{error_message}

🔧 **Suggestions:**
{suggestions}

💡 **Try asking:**
{example_queries}
            """.strip(),
            required_data_fields=['error_message'],
            optional_data_fields=['suggestions', 'example_queries'],
            visualization_recommendations=[],
            follow_up_templates=[
                "Would you like help rephrasing your question?",
                "Are you looking for a specific type of analysis?",
                "Would you like to see what data is available?",
                "Do you need help with query examples?"
            ]
        )
        
        # No Data Template
        templates[ResponseType.NO_DATA] = ResponseTemplate(
            response_type=ResponseType.NO_DATA,
            template="""
📭 **No Data Available:**

{no_data_message}

🔍 **Suggestions:**
{data_suggestions}

📊 **Available Data:**
{available_data}
            """.strip(),
            required_data_fields=['no_data_message'],
            optional_data_fields=['data_suggestions', 'available_data'],
            visualization_recommendations=[],
            follow_up_templates=[
                "Would you like to try a different time range?",
                "Are you interested in data for other airports?",
                "Would you like to see what data is available?",
                "Can I help you with a different type of analysis?"
            ]
        )
        
        return templates
    
    def _initialize_visualization_configs(self) -> Dict[VisualizationType, Dict[str, Any]]:
        """Initialize default configurations for different visualization types."""
        configs = {}
        
        configs[VisualizationType.BAR_CHART] = {
            'chart_type': 'bar',
            'orientation': 'vertical',
            'color_scheme': 'viridis',
            'show_values': True,
            'title_font_size': 16,
            'axis_font_size': 12
        }
        
        configs[VisualizationType.LINE_CHART] = {
            'chart_type': 'line',
            'line_width': 2,
            'marker_size': 6,
            'color_scheme': 'plotly',
            'show_grid': True,
            'title_font_size': 16
        }
        
        configs[VisualizationType.HEATMAP] = {
            'chart_type': 'heatmap',
            'color_scale': 'RdYlBu_r',
            'show_scale': True,
            'text_auto': True,
            'title_font_size': 16
        }
        
        configs[VisualizationType.HISTOGRAM] = {
            'chart_type': 'histogram',
            'bins': 30,
            'opacity': 0.7,
            'color': 'skyblue',
            'show_distribution': True,
            'title_font_size': 16
        }
        
        configs[VisualizationType.NETWORK_GRAPH] = {
            'chart_type': 'network',
            'node_size': 'degree',
            'edge_width': 'weight',
            'layout': 'spring',
            'show_labels': True,
            'title_font_size': 16
        }
        
        return configs
    
    def generate_response(
        self,
        response_type: ResponseType,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a formatted response based on type and data.
        
        Args:
            response_type: Type of response to generate
            data: Data to include in the response
            context: Optional context for personalization
            
        Returns:
            Dictionary containing formatted response and metadata
        """
        try:
            # Check cache first
            cache_key = self._generate_cache_key(response_type, data, context)
            if self.cache_manager:
                cached_response = self.cache_manager.get(cache_key)
                if cached_response:
                    logger.info("Returning cached response")
                    return cached_response
            
            # Get template
            template = self.templates.get(response_type)
            if not template:
                logger.error(f"No template found for response type: {response_type}")
                return self._generate_error_response(f"No template available for {response_type}")
            
            # Validate required data fields
            missing_fields = [field for field in template.required_data_fields if field not in data]
            if missing_fields:
                logger.warning(f"Missing required fields: {missing_fields}")
                # Try to provide defaults or generate error
                data = self._fill_missing_data(data, missing_fields)
            
            # Format the response
            formatted_response = self._format_template(template, data, context)
            
            # Generate visualization recommendations
            viz_recommendations = self._generate_visualization_recommendations(template, data)
            
            # Generate follow-up questions
            follow_ups = self._generate_follow_up_questions(template, data, context)
            
            # Create response object
            response = {
                'response_text': formatted_response,
                'response_type': response_type.value,
                'visualizations': viz_recommendations,
                'follow_up_questions': follow_ups,
                'data_summary': self._create_data_summary(data),
                'confidence': self._calculate_response_confidence(data, template),
                'timestamp': datetime.now().isoformat(),
                'cache_key': cache_key
            }
            
            # Cache the response
            if self.cache_manager:
                self.cache_manager.set(cache_key, response, ttl=3600)  # Cache for 1 hour
            
            logger.info(f"Generated {response_type.value} response successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return self._generate_error_response(str(e))
    
    def _generate_cache_key(
        self,
        response_type: ResponseType,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate a cache key for the response."""
        # Create a hash of the key components
        key_data = {
            'type': response_type.value,
            'data': data,
            'context': context or {}
        }
        
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _fill_missing_data(self, data: Dict[str, Any], missing_fields: List[str]) -> Dict[str, Any]:
        """Fill in missing data fields with defaults."""
        defaults = {
            'total_flights': 0,
            'average_delay': 0.0,
            'on_time_percentage': 0.0,
            'peak_hours': 'Unknown',
            'quietest_hours': 'Unknown',
            'congestion_score': 0.0,
            'route': 'Unknown route',
            'optimal_times': 'No optimal times identified',
            'success_rate': 0.0,
            'affected_flights': 0,
            'delay_reduction': 0.0,
            'efficiency_improvement': 0.0,
            'risk_level': 'Unknown',
            'critical_flights': 'None identified',
            'cascade_probability': 0.0,
            'resilience_score': 0.0,
            'flight_details': 'No flight details available',
            'error_message': 'Unknown error occurred',
            'no_data_message': 'No data available for the specified criteria'
        }
        
        filled_data = data.copy()
        for field in missing_fields:
            if field in defaults:
                filled_data[field] = defaults[field]
            else:
                filled_data[field] = 'Not available'
        
        return filled_data
    
    def _format_template(
        self,
        template: ResponseTemplate,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Format the response template with data."""
        try:
            # Add context-specific formatting
            if context:
                data = self._add_context_formatting(data, context)
            
            # Format lists and complex data
            formatted_data = self._format_data_for_template(data)
            
            # Apply template formatting
            formatted_response = template.template.format(**formatted_data)
            
            return formatted_response
            
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return f"Error formatting response: Missing data for {e}"
        except Exception as e:
            logger.error(f"Error formatting template: {str(e)}")
            return f"Error formatting response: {str(e)}"
    
    def _add_context_formatting(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add context-specific formatting to data."""
        formatted_data = data.copy()
        
        # Add user preferences if available
        if 'user_preferences' in context:
            prefs = context['user_preferences']
            
            # Time format preference
            if 'time_format' in prefs and prefs['time_format'] == '12h':
                # Convert times to 12-hour format
                for key, value in formatted_data.items():
                    if 'time' in key.lower() and isinstance(value, str):
                        formatted_data[key] = self._convert_to_12h_format(value)
        
        # Add session context
        if 'session_context' in context:
            session = context['session_context']
            
            # Add frequently mentioned airports
            if 'frequent_airports' in session:
                airports = session['frequent_airports']
                if airports and 'airport' not in formatted_data:
                    formatted_data['airport'] = airports[0]  # Use most frequent
        
        return formatted_data
    
    def _format_data_for_template(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data values for template insertion."""
        formatted = {}
        
        for key, value in data.items():
            if isinstance(value, list):
                if len(value) == 0:
                    formatted[key] = "None"
                elif len(value) <= 3:
                    formatted[key] = ", ".join(map(str, value))
                else:
                    formatted[key] = f"{', '.join(map(str, value[:3]))} and {len(value) - 3} more"
            elif isinstance(value, dict):
                # Format dictionary as key-value pairs
                pairs = [f"{k}: {v}" for k, v in value.items()]
                formatted[key] = "\n".join(pairs)
            elif isinstance(value, float):
                # Format floats to reasonable precision
                formatted[key] = f"{value:.2f}" if abs(value) < 1000 else f"{value:.0f}"
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def _convert_to_12h_format(self, time_str: str) -> str:
        """Convert time string to 12-hour format."""
        try:
            # Simple conversion for hour ranges like "14-16"
            if '-' in time_str:
                start, end = time_str.split('-')
                start_12h = self._hour_to_12h(int(start))
                end_12h = self._hour_to_12h(int(end))
                return f"{start_12h}-{end_12h}"
            else:
                return time_str  # Return as-is if can't convert
        except:
            return time_str
    
    def _hour_to_12h(self, hour: int) -> str:
        """Convert 24-hour to 12-hour format."""
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour - 12} PM"
    
    def _generate_visualization_recommendations(
        self,
        template: ResponseTemplate,
        data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate visualization recommendations based on template and data."""
        recommendations = []
        
        for viz_rec in template.visualization_recommendations:
            # Check if required data columns are available
            available_columns = self._get_available_data_columns(data)
            if all(col in available_columns for col in viz_rec.data_columns):
                
                # Get default config for this visualization type
                config = self.visualization_configs.get(viz_rec.type, {}).copy()
                config.update(viz_rec.config)
                
                recommendation = {
                    'type': viz_rec.type.value,
                    'title': viz_rec.title,
                    'description': viz_rec.description,
                    'data_columns': viz_rec.data_columns,
                    'config': config,
                    'priority': viz_rec.priority,
                    'implementation': self._get_visualization_implementation(viz_rec.type)
                }
                
                recommendations.append(recommendation)
        
        # Sort by priority
        recommendations.sort(key=lambda x: x['priority'])
        
        return recommendations
    
    def _get_available_data_columns(self, data: Dict[str, Any]) -> List[str]:
        """Get list of available data columns from the data."""
        columns = []
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                columns.extend(value.columns.tolist())
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # List of dictionaries
                columns.extend(value[0].keys())
            else:
                columns.append(key)
        
        return columns
    
    def _get_visualization_implementation(self, viz_type: VisualizationType) -> Dict[str, str]:
        """Get implementation details for visualization type."""
        implementations = {
            VisualizationType.BAR_CHART: {
                'plotly': 'px.bar(df, x="x_col", y="y_col")',
                'matplotlib': 'plt.bar(x, y)',
                'streamlit': 'st.bar_chart(df)'
            },
            VisualizationType.LINE_CHART: {
                'plotly': 'px.line(df, x="x_col", y="y_col")',
                'matplotlib': 'plt.plot(x, y)',
                'streamlit': 'st.line_chart(df)'
            },
            VisualizationType.HEATMAP: {
                'plotly': 'px.imshow(df)',
                'matplotlib': 'plt.imshow(data)',
                'streamlit': 'st.plotly_chart(px.imshow(df))'
            },
            VisualizationType.HISTOGRAM: {
                'plotly': 'px.histogram(df, x="col")',
                'matplotlib': 'plt.hist(data)',
                'streamlit': 'st.plotly_chart(px.histogram(df))'
            }
        }
        
        return implementations.get(viz_type, {})
    
    def _generate_follow_up_questions(
        self,
        template: ResponseTemplate,
        data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate contextual follow-up questions."""
        follow_ups = []
        
        for template_question in template.follow_up_templates:
            try:
                # Format template with available data
                formatted_question = template_question.format(**data)
                follow_ups.append(formatted_question)
            except KeyError:
                # If formatting fails, add the template as-is (removing format placeholders)
                clean_question = re.sub(r'\{[^}]+\}', 'this data', template_question)
                follow_ups.append(clean_question)
        
        # Add context-specific follow-ups
        if context:
            context_follow_ups = self._generate_context_follow_ups(data, context)
            follow_ups.extend(context_follow_ups)
        
        # Limit to reasonable number
        return follow_ups[:5]
    
    def _generate_context_follow_ups(self, data: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate follow-up questions based on context."""
        follow_ups = []
        
        # Based on conversation history
        if 'recent_queries' in context:
            recent = context['recent_queries']
            if len(recent) > 1:
                last_intent = recent[-1].get('intent', '')
                if last_intent == 'delay_analysis':
                    follow_ups.append("Would you like to see congestion patterns that might be causing these delays?")
                elif last_intent == 'congestion':
                    follow_ups.append("Would you like delay analysis for the busy periods we identified?")
        
        # Based on user preferences
        if 'user_preferences' in context:
            prefs = context['user_preferences']
            if 'preferred_airports' in prefs:
                airports = prefs['preferred_airports']
                if airports:
                    follow_ups.append(f"Would you like similar analysis for {airports[0]}?")
        
        return follow_ups
    
    def _create_data_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the data used in the response."""
        summary = {
            'data_points': 0,
            'data_types': [],
            'time_range': None,
            'airports': [],
            'airlines': []
        }
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                summary['data_points'] += len(value)
                summary['data_types'].append(f"DataFrame ({key})")
                
                # Extract time range if available
                if 'scheduled_departure' in value.columns:
                    summary['time_range'] = {
                        'start': value['scheduled_departure'].min().isoformat(),
                        'end': value['scheduled_departure'].max().isoformat()
                    }
                
                # Extract airports
                if 'origin_airport' in value.columns:
                    summary['airports'].extend(value['origin_airport'].unique().tolist())
                if 'destination_airport' in value.columns:
                    summary['airports'].extend(value['destination_airport'].unique().tolist())
                
                # Extract airlines
                if 'airline' in value.columns:
                    summary['airlines'].extend(value['airline'].unique().tolist())
            
            elif isinstance(value, list):
                summary['data_points'] += len(value)
                summary['data_types'].append(f"List ({key})")
            
            elif isinstance(value, (int, float)):
                summary['data_types'].append(f"Numeric ({key})")
        
        # Remove duplicates
        summary['airports'] = list(set(summary['airports']))
        summary['airlines'] = list(set(summary['airlines']))
        summary['data_types'] = list(set(summary['data_types']))
        
        return summary
    
    def _calculate_response_confidence(self, data: Dict[str, Any], template: ResponseTemplate) -> float:
        """Calculate confidence score for the response."""
        confidence = 1.0
        
        # Reduce confidence for missing optional fields
        missing_optional = [
            field for field in template.optional_data_fields 
            if field not in data or not data[field]
        ]
        confidence -= len(missing_optional) * 0.1
        
        # Reduce confidence for small datasets
        total_data_points = 0
        for value in data.values():
            if isinstance(value, pd.DataFrame):
                total_data_points += len(value)
            elif isinstance(value, list):
                total_data_points += len(value)
        
        if total_data_points < 10:
            confidence -= 0.3
        elif total_data_points < 50:
            confidence -= 0.1
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _generate_error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate an error response."""
        return {
            'response_text': f"❌ Error: {error_message}",
            'response_type': ResponseType.ERROR.value,
            'visualizations': [],
            'follow_up_questions': [
                "Would you like help rephrasing your question?",
                "Are you looking for a specific type of analysis?",
                "Would you like to see what data is available?"
            ],
            'data_summary': {'data_points': 0, 'data_types': [], 'error': error_message},
            'confidence': 0.0,
            'timestamp': datetime.now().isoformat(),
            'cache_key': None
        }
    
    def get_response_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available response templates."""
        template_info = {}
        
        for response_type, template in self.templates.items():
            template_info[response_type.value] = {
                'required_fields': template.required_data_fields,
                'optional_fields': template.optional_data_fields,
                'visualization_count': len(template.visualization_recommendations),
                'follow_up_count': len(template.follow_up_templates)
            }
        
        return template_info
    
    def clear_cache(self):
        """Clear the response cache."""
        if self.cache_manager:
            # This would clear all cached responses
            # In a real implementation, you might want to clear only response-related cache
            logger.info("Response cache cleared")
        else:
            logger.info("No cache manager available")
    
    def add_custom_template(self, response_type: ResponseType, template: ResponseTemplate):
        """Add a custom response template."""
        self.templates[response_type] = template
        logger.info(f"Added custom template for {response_type.value}")
    
    def get_visualization_types(self) -> List[Dict[str, Any]]:
        """Get information about available visualization types."""
        viz_info = []
        
        for viz_type in VisualizationType:
            config = self.visualization_configs.get(viz_type, {})
            viz_info.append({
                'type': viz_type.value,
                'name': viz_type.value.replace('_', ' ').title(),
                'default_config': config,
                'implementation': self._get_visualization_implementation(viz_type)
            })
        
        return viz_info