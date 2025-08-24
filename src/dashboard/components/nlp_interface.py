"""
Natural Language Processing Interface Components
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import re
import json

class QueryValidator:
    """Validates and preprocesses user queries"""
    
    def __init__(self):
        self.valid_airports = ['BOM', 'DEL', 'BLR', 'MAA', 'CCU', 'HYD', 'AMD', 'COK']
        self.valid_airlines = ['Airline A', 'Airline B', 'Airline C', 'Airline D', 'Airline E', 'Airline F']
        self.query_patterns = {
            'delay_analysis': [
                r'delay', r'late', r'on.?time', r'punctual'
            ],
            'congestion_analysis': [
                r'busy', r'congestion', r'traffic', r'peak', r'crowded'
            ],
            'schedule_optimization': [
                r'best time', r'optimal', r'schedule', r'when to fly'
            ],
            'network_impact': [
                r'impact', r'affect', r'cascade', r'network', r'connection'
            ]
        }
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Validate and analyze the user query
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with validation results and extracted entities
        """
        
        if not query or len(query.strip()) < 3:
            return {
                'valid': False,
                'error': 'Query too short. Please provide more details.',
                'suggestions': [
                    'What are the best times to fly from Mumbai to Delhi?',
                    'Which flights have the most delays?',
                    'Show me congestion patterns for today.'
                ]
            }
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine query type
        query_type = self._classify_query(query)
        
        # Check for ambiguity
        ambiguity_check = self._check_ambiguity(query, entities)
        
        return {
            'valid': True,
            'query_type': query_type,
            'entities': entities,
            'ambiguity': ambiguity_check,
            'processed_query': query.strip().lower()
        }
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract airports, airlines, and other entities from query"""
        
        entities = {
            'airports': [],
            'airlines': [],
            'time_references': [],
            'metrics': []
        }
        
        query_upper = query.upper()
        
        # Extract airports
        for airport in self.valid_airports:
            if airport in query_upper:
                entities['airports'].append(airport)
        
        # Extract airlines
        for airline in self.valid_airlines:
            if airline.upper() in query_upper:
                entities['airlines'].append(airline)
        
        # Extract time references
        time_patterns = [
            r'today', r'yesterday', r'tomorrow', r'this week', r'last week',
            r'morning', r'afternoon', r'evening', r'night',
            r'\d{1,2}:\d{2}', r'\d{1,2}\s?(am|pm)', r'\d{1,2}\s?o\'?clock'
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query.lower())
            entities['time_references'].extend(matches)
        
        # Extract metrics
        metric_patterns = [
            r'delay', r'on.?time', r'congestion', r'traffic', r'cost', r'efficiency'
        ]
        
        for pattern in metric_patterns:
            if re.search(pattern, query.lower()):
                entities['metrics'].append(pattern)
        
        return entities
    
    def _classify_query(self, query: str) -> str:
        """Classify the query into predefined categories"""
        
        query_lower = query.lower()
        scores = {}
        
        for category, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[category] = score
        
        if not any(scores.values()):
            return 'general'
        
        return max(scores, key=scores.get)
    
    def _check_ambiguity(self, query: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Check for ambiguous queries that need clarification"""
        
        ambiguity = {
            'is_ambiguous': False,
            'clarifications_needed': [],
            'suggestions': []
        }
        
        # Check for missing time context
        if not entities['time_references'] and any(word in query.lower() for word in ['when', 'time', 'schedule']):
            ambiguity['is_ambiguous'] = True
            ambiguity['clarifications_needed'].append('time_period')
            ambiguity['suggestions'].append('Please specify a time period (e.g., today, this week, morning)')
        
        # Check for missing airport context
        if not entities['airports'] and any(word in query.lower() for word in ['airport', 'fly', 'flight']):
            ambiguity['is_ambiguous'] = True
            ambiguity['clarifications_needed'].append('airport')
            ambiguity['suggestions'].append('Please specify which airport(s) you\'re interested in')
        
        return ambiguity

class QuerySuggestionEngine:
    """Generates intelligent query suggestions"""
    
    def __init__(self):
        self.suggestion_templates = {
            'delay_analysis': [
                "What are the average delays for {airline} flights?",
                "Show me delay patterns for {airport} airport",
                "Which time slots have the most delays?",
                "Compare delay performance between airlines"
            ],
            'congestion_analysis': [
                "When is {airport} airport busiest?",
                "Show me traffic patterns for {time_period}",
                "Which runways have the highest utilization?",
                "What are the peak congestion hours?"
            ],
            'schedule_optimization': [
                "What's the best time to fly from {origin} to {destination}?",
                "How can I optimize my flight schedule?",
                "Show me the least congested time slots",
                "When should I avoid flying to minimize delays?"
            ],
            'network_impact': [
                "Which flights have the biggest impact on delays?",
                "Show me how delays cascade through the network",
                "What happens if flight {flight_id} is delayed?",
                "Which are the most critical flights?"
            ]
        }
    
    def get_contextual_suggestions(self, query_history: List[Dict], 
                                 current_context: Dict[str, Any]) -> List[str]:
        """Generate contextual suggestions based on query history and current context"""
        
        suggestions = []
        
        # Base suggestions
        base_suggestions = [
            "What are the best times to fly today?",
            "Show me current delay patterns",
            "Which airports are most congested right now?",
            "How do weather conditions affect delays?"
        ]
        
        # Context-aware suggestions
        if current_context.get('airports'):
            airport = current_context['airports'][0]
            suggestions.extend([
                f"Show me delay statistics for {airport} airport",
                f"What are the peak hours at {airport}?",
                f"Compare {airport} with other airports"
            ])
        
        if current_context.get('airlines'):
            airline = current_context['airlines'][0]
            suggestions.extend([
                f"Analyze {airline} on-time performance",
                f"Compare {airline} with other airlines",
                f"Show {airline} delay causes"
            ])
        
        # History-based suggestions
        if query_history:
            recent_types = [q.get('query_type', 'general') for q in query_history[-3:]]
            if 'delay_analysis' in recent_types:
                suggestions.append("Now show me congestion patterns")
            if 'congestion_analysis' in recent_types:
                suggestions.append("How would schedule changes impact this?")
        
        return suggestions[:6]  # Limit to 6 suggestions
    
    def get_autocomplete_suggestions(self, partial_query: str) -> List[str]:
        """Generate autocomplete suggestions for partial queries"""
        
        if len(partial_query) < 3:
            return []
        
        # Common query completions
        completions = [
            "What are the best times to fly from Mumbai to Delhi?",
            "Show me delay patterns for today",
            "Which flights have the most delays?",
            "When is Delhi airport busiest?",
            "Compare airline performance",
            "How do weather conditions affect flights?",
            "What are the peak congestion hours?",
            "Show me network impact analysis",
            "Which runways are most utilized?",
            "Analyze cascading delay effects"
        ]
        
        # Filter based on partial query
        partial_lower = partial_query.lower()
        matching = [comp for comp in completions if partial_lower in comp.lower()]
        
        return matching[:5]  # Limit to 5 suggestions

class ResponseStreamer:
    """Handles real-time response streaming"""
    
    def __init__(self):
        self.response_chunks = []
    
    def stream_response(self, response_text: str, chunk_size: int = 50) -> None:
        """
        Stream response text in chunks for real-time display
        
        Args:
            response_text: Full response text
            chunk_size: Number of characters per chunk
        """
        
        # Create placeholder for streaming
        response_placeholder = st.empty()
        
        # Stream response in chunks
        displayed_text = ""
        
        for i in range(0, len(response_text), chunk_size):
            chunk = response_text[i:i + chunk_size]
            displayed_text += chunk
            
            # Update placeholder with current text
            response_placeholder.markdown(displayed_text + "▌")  # Add cursor
            
            # Small delay for streaming effect
            import time
            time.sleep(0.05)
        
        # Final update without cursor
        response_placeholder.markdown(displayed_text)

class QueryAnalytics:
    """Tracks and analyzes query patterns"""
    
    def __init__(self):
        self.analytics_data = {
            'query_count': 0,
            'query_types': {},
            'popular_entities': {},
            'response_ratings': [],
            'session_duration': 0
        }
    
    def track_query(self, query: str, query_type: str, entities: Dict[str, List[str]], 
                   response_time: float) -> None:
        """Track query analytics"""
        
        self.analytics_data['query_count'] += 1
        
        # Track query types
        if query_type in self.analytics_data['query_types']:
            self.analytics_data['query_types'][query_type] += 1
        else:
            self.analytics_data['query_types'][query_type] = 1
        
        # Track popular entities
        for entity_type, entity_list in entities.items():
            if entity_type not in self.analytics_data['popular_entities']:
                self.analytics_data['popular_entities'][entity_type] = {}
            
            for entity in entity_list:
                if entity in self.analytics_data['popular_entities'][entity_type]:
                    self.analytics_data['popular_entities'][entity_type][entity] += 1
                else:
                    self.analytics_data['popular_entities'][entity_type][entity] = 1
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary for display"""
        
        summary = {
            'total_queries': self.analytics_data['query_count'],
            'most_common_type': max(self.analytics_data['query_types'], 
                                  key=self.analytics_data['query_types'].get) 
                                  if self.analytics_data['query_types'] else 'None',
            'avg_rating': np.mean(self.analytics_data['response_ratings']) 
                         if self.analytics_data['response_ratings'] else 0
        }
        
        return summary

def create_chat_interface():
    """Create the main chat interface component"""
    
    # Initialize components
    if 'validator' not in st.session_state:
        st.session_state.validator = QueryValidator()
    
    if 'suggestion_engine' not in st.session_state:
        st.session_state.suggestion_engine = QuerySuggestionEngine()
    
    if 'response_streamer' not in st.session_state:
        st.session_state.response_streamer = ResponseStreamer()
    
    if 'query_analytics' not in st.session_state:
        st.session_state.query_analytics = QueryAnalytics()
    
    # Chat interface
    st.markdown("### 💬 Chat with AI Assistant")
    
    # Query input with autocomplete
    query_input = st.text_input(
        "Ask your question:",
        placeholder="Type your question about flights, delays, or scheduling...",
        help="Try asking about delays, congestion, best times to fly, or network impacts"
    )
    
    # Autocomplete suggestions
    if query_input and len(query_input) >= 3:
        suggestions = st.session_state.suggestion_engine.get_autocomplete_suggestions(query_input)
        if suggestions:
            st.markdown("**💡 Did you mean:**")
            for i, suggestion in enumerate(suggestions):
                if st.button(f"📝 {suggestion}", key=f"autocomplete_{i}_{hash(suggestion)}"):
                    st.session_state.current_query = suggestion
                    st.rerun()
    
    # Process query button
    col1, col2 = st.columns([1, 3])
    
    with col1:
        process_button = st.button("🚀 Ask AI", key="ask_ai_btn", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Chat", key="clear_chat_btn", use_container_width=True):
            # Clear chat history
            for key in ['ai_response', 'ai_visualization', 'query_history']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    
    # Process the query
    if process_button and query_input:
        process_nlp_query(query_input)
    
    # Show contextual suggestions
    show_contextual_suggestions()

def process_nlp_query(query: str):
    """Process the natural language query"""
    
    validator = st.session_state.validator
    
    # Validate query
    validation_result = validator.validate_query(query)
    
    if not validation_result['valid']:
        st.error(f"❌ {validation_result['error']}")
        if validation_result.get('suggestions'):
            st.markdown("**Try these instead:**")
            for suggestion in validation_result['suggestions']:
                st.markdown(f"• {suggestion}")
        return
    
    # Check for ambiguity
    if validation_result['ambiguity']['is_ambiguous']:
        st.warning("🤔 Your query needs clarification:")
        for clarification in validation_result['ambiguity']['suggestions']:
            st.markdown(f"• {clarification}")
        return
    
    # Process query with streaming response
    with st.spinner("🤖 AI is analyzing your question..."):
        # Simulate processing time
        import time
        time.sleep(1)
        
        # Generate response based on query type
        response = generate_contextual_response(query, validation_result)
        
        # Stream the response
        st.session_state.response_streamer.stream_response(response)
        
        # Track analytics
        st.session_state.query_analytics.track_query(
            query, 
            validation_result['query_type'],
            validation_result['entities'],
            1.5  # Mock response time
        )

def generate_contextual_response(query: str, validation_result: Dict[str, Any]) -> str:
    """Generate contextual response based on query analysis"""
    
    query_type = validation_result['query_type']
    entities = validation_result['entities']
    
    # Base response templates
    responses = {
        'delay_analysis': f"""
        **🕐 Delay Analysis Results**
        
        Based on your query about delays, here's what I found:
        
        **Key Insights:**
        • Average delay across all flights: 12.3 minutes
        • Most delayed time period: 8:00-10:00 AM (avg 18.7 min)
        • Best on-time performance: 6:00-7:00 AM (avg 4.2 min delay)
        
        **Recommendations:**
        • Schedule flights between 6:00-7:00 AM for minimal delays
        • Avoid 8:00-10:00 AM slot during weekdays
        • Weather delays peak at 3:00-5:00 PM during monsoon season
        """,
        
        'congestion_analysis': f"""
        **🚦 Congestion Analysis Results**
        
        Here's the current congestion analysis:
        
        **Peak Congestion Times:**
        • Morning rush: 8:00-10:00 AM (congestion level: 8.5/10)
        • Evening peak: 6:00-8:00 PM (congestion level: 7.8/10)
        • Lowest congestion: 2:00-4:00 AM (congestion level: 1.2/10)
        
        **Runway Utilization:**
        • Runway 09L/27R: 89% utilized (near capacity)
        • Runway 09R/27L: 76% utilized
        • Runway 14/32: 45% utilized (available capacity)
        """,
        
        'schedule_optimization': f"""
        **📊 Schedule Optimization Recommendations**
        
        Based on historical data and current patterns:
        
        **Optimal Time Slots:**
        • 6:00-7:00 AM: Excellent (minimal delays, low congestion)
        • 10:00-11:00 AM: Good (moderate traffic, acceptable delays)
        • 2:00-3:00 PM: Fair (weather risk during monsoon)
        
        **Avoid These Times:**
        • 8:00-10:00 AM: High congestion and delays
        • 6:00-8:00 PM: Peak traffic, extended taxi times
        • 3:00-5:00 PM: Weather delay risk
        """,
        
        'network_impact': f"""
        **🌐 Network Impact Analysis**
        
        Critical flights with highest cascading impact:
        
        **Top Impact Flights:**
        1. AI101 (BOM-DEL): Affects 12 downstream flights
        2. 6E234 (DEL-BOM): Affects 8 downstream flights  
        3. UK955 (BOM-BLR): Affects 6 downstream flights
        
        **Network Resilience:**
        • Single flight delay (15 min): Affects 3-5 flights on average
        • Hub closure (30 min): Affects 25-40 flights
        • Weather disruption: Can cascade to 50+ flights
        """,
        
        'general': f"""
        **🤖 General Flight Analysis**
        
        I've analyzed your query about flight operations. Here are some general insights:
        
        **Current System Status:**
        • Total flights monitored: 1,247 today
        • Average system delay: 12.5 minutes
        • On-time performance: 78.2%
        
        **Quick Recommendations:**
        • Use early morning slots for better punctuality
        • Monitor weather forecasts for afternoon flights
        • Consider alternative airports during peak congestion
        
        **Need more specific information?** Try asking about:
        • Specific airports or airlines
        • Particular time periods
        • Delay causes or congestion patterns
        """
    }
    
    # Customize response based on entities
    base_response = responses.get(query_type, responses['general'])
    
    # Add entity-specific information
    if entities['airports']:
        airport_info = f"\n\n**Airport-Specific Data for {', '.join(entities['airports'])}:**\n"
        for airport in entities['airports']:
            airport_info += f"• {airport}: Current delay avg 15.2 min, congestion level 6.8/10\n"
        base_response += airport_info
    
    if entities['airlines']:
        airline_info = f"\n\n**Airline-Specific Data for {', '.join(entities['airlines'])}:**\n"
        for airline in entities['airlines']:
            airline_info += f"• {airline}: On-time rate 76.3%, avg delay 13.8 min\n"
        base_response += airline_info
    
    return base_response

def show_contextual_suggestions():
    """Show contextual suggestions based on current context"""
    
    if 'suggestion_engine' in st.session_state:
        # Get query history
        history = st.session_state.get('query_history', [])
        current_context = st.session_state.get('current_context', {})
        
        # Get suggestions
        suggestions = st.session_state.suggestion_engine.get_contextual_suggestions(
            history, current_context
        )
        
        if suggestions:
            st.markdown("### 🎯 Smart Suggestions")
            
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                col = cols[i % 2]
                with col:
                    if st.button(f"💡 {suggestion}", key=f"context_suggestion_{i}_{hash(suggestion)}"):
                        st.session_state.current_query = suggestion
                        st.rerun()

def show_query_analytics():
    """Show query analytics dashboard"""
    
    if 'query_analytics' in st.session_state:
        analytics = st.session_state.query_analytics.get_analytics_summary()
        
        st.markdown("### 📊 Query Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Queries", analytics['total_queries'])
        
        with col2:
            st.metric("Most Common Type", analytics['most_common_type'])
        
        with col3:
            st.metric("Avg Rating", f"{analytics['avg_rating']:.1f}/5")