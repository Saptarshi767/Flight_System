"""
Natural Language Processing Interface page for the flight scheduling analysis dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dashboard.utils.session import add_query_to_history, get_query_history
from dashboard.components.nlp_interface import (
    create_chat_interface, 
    show_query_analytics,
    QueryValidator,
    QuerySuggestionEngine,
    ResponseStreamer,
    QueryAnalytics
)

def show():
    """Display the NLP interface page"""
    
    st.title("💬 AI Assistant - Natural Language Interface")
    st.markdown("Ask questions about flight data using natural language")
    
    # Quick start examples
    with st.expander("💡 Example Questions", expanded=False):
        examples = [
            "What's the best time to schedule a flight from Mumbai to Delhi?",
            "Which flights cause the most delays in the network?",
            "Show me congestion patterns for Delhi airport last week",
            "What are the main causes of delays on Friday evenings?",
            "How would moving flight AI101 to 7 AM affect other flights?",
            "Which runway has the highest utilization at Mumbai airport?"
        ]
        
        for i, example in enumerate(examples, 1):
            if st.button(f"📝 {example}", key=f"nlp_example_{i}"):
                st.session_state['current_query'] = example
                st.rerun()
    
    st.markdown("---")
    
    # Main chat interface using the new component
    create_chat_interface()
    
    # Response area
    if 'ai_response' in st.session_state and st.session_state['ai_response']:
        st.markdown("---")
        st.markdown("### 🤖 AI Response")
        
        with st.container():
            st.markdown(st.session_state['ai_response'])
            
            # Show visualization if available
            if 'ai_visualization' in st.session_state and st.session_state['ai_visualization']:
                st.plotly_chart(st.session_state['ai_visualization'], use_container_width=True)
            
            # Response actions
            col_x, col_y, col_z = st.columns(3)
            
            with col_x:
                if st.button("👍 Helpful", key="nlp_helpful"):
                    st.success("Thank you for your feedback!")
                    # Track positive feedback
                    if 'query_analytics' in st.session_state:
                        st.session_state.query_analytics.analytics_data['response_ratings'].append(5)
            
            with col_y:
                if st.button("👎 Not Helpful", key="nlp_not_helpful"):
                    st.info("We'll improve our responses!")
                    # Track negative feedback
                    if 'query_analytics' in st.session_state:
                        st.session_state.query_analytics.analytics_data['response_ratings'].append(2)
            
            with col_z:
                if st.button("📊 Show More Details", key="nlp_more_details"):
                    show_detailed_analysis()
    
    # Query history section (moved to main content area)
    with st.expander("📚 Query History", expanded=False):
        history = get_query_history()
        
        if history:
            # Show recent queries
            st.markdown("**Recent Questions:**")
            for i, entry in enumerate(reversed(history[-5:]), 1):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Q{i}:** {entry['query']}")
                    st.markdown(f"*{entry['timestamp'].strftime('%Y-%m-%d %H:%M')}*")
                with col2:
                    if st.button(f"🔄 Ask Again", key=f"nlp_reask_{i}"):
                        st.session_state['current_query'] = entry['query']
                        st.rerun()
                st.markdown("---")
        else:
            st.info("No previous queries yet. Start by asking a question!")
        
        st.markdown("---")
        
        # Show query analytics in sidebar
        show_query_analytics()
    
    # Query analytics dashboard
    st.markdown("---")
    st.subheader("📊 Query Analytics Dashboard")
    
    if history:
        # Query frequency analysis
        col1, col2 = st.columns(2)
        
        with col1:
            # Most common query types
            query_types = categorize_queries(history)
            
            fig_types = px.pie(
                values=list(query_types.values()),
                names=list(query_types.keys()),
                title="Query Categories Distribution"
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        with col2:
            # Query frequency over time
            dates = [entry['timestamp'].date() for entry in history]
            date_counts = pd.Series(dates).value_counts().sort_index()
            
            fig_freq = px.line(
                x=date_counts.index,
                y=date_counts.values,
                title="Query Frequency Over Time",
                labels={'x': 'Date', 'y': 'Number of Queries'}
            )
            st.plotly_chart(fig_freq, use_container_width=True)
        
        # Query performance metrics
        st.markdown("### 📈 Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Queries", len(history))
        
        with col2:
            avg_response_time = 2.3  # Mock data
            st.metric("Avg Response Time", f"{avg_response_time:.1f}s")
        
        with col3:
            success_rate = 94.2  # Mock data
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        with col4:
            user_satisfaction = 4.2  # Mock data
            st.metric("User Satisfaction", f"{user_satisfaction:.1f}/5")
    
    # System status
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🤖 **AI Status**\nOpenAI GPT-4: ✅ Active\nResponse Time: ~2.3s\nAPI Calls: 1,247 today")
    
    with col2:
        st.info("📊 **Data Status**\nLast Update: 5 min ago\nRecords: 1.2M flights\nData Quality: 98.7%")
    
    with col3:
        st.info("🔧 **System Health**\nAPI: ✅ Operational\nDatabase: ✅ Connected\nCache: ✅ Active")

def process_query(query: str):
    """Process the user query and generate AI response"""
    
    # Simulate AI processing
    with st.spinner("🤖 AI is analyzing your question..."):
        # In a real implementation, this would call the OpenAI API
        response = generate_mock_response(query)
        visualization = generate_mock_visualization(query)
        
        # Store in session state
        st.session_state['ai_response'] = response
        st.session_state['ai_visualization'] = visualization
        
        # Add to history
        add_query_to_history(query, response)
    
    st.success("✅ Analysis complete!")

def generate_mock_response(query: str) -> str:
    """Generate a mock AI response based on the query"""
    
    query_lower = query.lower()
    
    if 'best time' in query_lower or 'optimal' in query_lower:
        return """
        **🕐 Best Time Analysis**
        
        Based on historical data analysis, here are the optimal scheduling recommendations:
        
        **For Mumbai to Delhi Route:**
        - **Best departure times:** 6:00-7:00 AM (avg delay: 5.2 min) and 10:00-11:00 AM (avg delay: 7.8 min)
        - **Avoid:** 8:00-9:00 AM (peak congestion, avg delay: 23.4 min)
        
        **Key Insights:**
        - Early morning flights have 67% better on-time performance
        - Weather delays are minimal before 10 AM
        - Air traffic congestion peaks between 8-10 AM and 6-8 PM
        
        **Recommendation:** Schedule your flight between 6:00-7:00 AM for optimal performance.
        """
    
    elif 'delay' in query_lower and 'cause' in query_lower:
        return """
        **📊 Delay Cause Analysis**
        
        Primary causes of flight delays based on recent data:
        
        1. **Air Traffic Control (35%)** - Peak hour congestion
        2. **Weather (28%)** - Monsoon season impact
        3. **Technical Issues (18%)** - Aircraft maintenance
        4. **Operational (15%)** - Crew scheduling, ground handling
        5. **Other (4%)** - Security, passenger issues
        
        **Friday Evening Pattern:**
        - 45% higher delays compared to weekday average
        - Weather delays increase by 60% during 3-6 PM
        - Recommend avoiding 5-7 PM slots on Fridays
        """
    
    elif 'congestion' in query_lower or 'busy' in query_lower:
        return """
        **🚦 Congestion Pattern Analysis**
        
        **Delhi Airport - Last Week:**
        - **Peak congestion:** Monday 8:00-10:00 AM (187 flights/hour)
        - **Lowest traffic:** Tuesday 2:00-4:00 AM (12 flights/hour)
        - **Average runway utilization:** 78.5%
        
        **Congestion Hotspots:**
        - Morning rush: 7:00-10:00 AM
        - Evening peak: 6:00-9:00 PM
        - Weekend patterns show 23% less congestion
        
        **Recommendations:**
        - Reschedule 15 flights from 8-9 AM to 6-7 AM
        - Use secondary runways during peak hours
        """
    
    elif 'network' in query_lower or 'cascading' in query_lower:
        return """
        **🌐 Network Impact Analysis**
        
        **Flights with Highest Cascading Impact:**
        1. **AI101 (BOM-DEL)** - Affects 12 downstream flights
        2. **6E234 (DEL-BOM)** - Affects 8 downstream flights
        3. **UK955 (BOM-BLR)** - Affects 6 downstream flights
        
        **Impact Factors:**
        - Aircraft rotation schedules
        - Crew connection requirements
        - Passenger connection times
        
        **Critical Time Windows:**
        - 8:00-10:00 AM: High network sensitivity
        - 6:00-8:00 PM: Maximum connection impact
        """
    
    else:
        return """
        **🤖 AI Analysis Complete**
        
        I've analyzed your question about flight scheduling. Here are the key insights:
        
        - Current flight data shows normal operational patterns
        - No significant anomalies detected in recent scheduling
        - System recommendations are available for optimization
        
        **Would you like me to:**
        - Provide more specific analysis?
        - Show detailed visualizations?
        - Analyze a particular route or time period?
        
        Feel free to ask more specific questions about delays, congestion, or scheduling optimization!
        """

def generate_mock_visualization(query: str):
    """Generate a mock visualization based on the query"""
    
    query_lower = query.lower()
    
    if 'time' in query_lower or 'hour' in query_lower:
        # Generate hourly delay pattern
        hours = list(range(24))
        delays = np.random.normal(15, 8, 24)
        delays = np.maximum(delays, 0)
        
        # Make certain hours have higher delays
        delays[8:10] *= 1.8  # Morning peak
        delays[18:20] *= 1.5  # Evening peak
        
        fig = px.bar(
            x=hours,
            y=delays,
            title="Average Delay by Hour of Day",
            labels={'x': 'Hour', 'y': 'Average Delay (minutes)'},
            color=delays,
            color_continuous_scale='RdYlGn_r'
        )
        return fig
    
    elif 'congestion' in query_lower:
        # Generate congestion heatmap
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        congestion_data = np.random.rand(len(days), len(hours)) * 10
        
        fig = px.imshow(
            congestion_data,
            x=hours,
            y=days,
            color_continuous_scale='RdYlGn_r',
            title="Weekly Congestion Pattern",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Congestion Level'}
        )
        return fig
    
    return None

def expand_suggestion(suggestion: str) -> str:
    """Expand a suggestion into a full query"""
    
    expansions = {
        "Analyze today's delays": "What are the main causes of delays today and which flights are most affected?",
        "Show peak hours": "What are the busiest hours at Mumbai and Delhi airports and when should I avoid scheduling?",
        "Compare airlines": "Compare the on-time performance and delay patterns across different airlines",
        "Weather impact": "How does weather affect flight delays and what are the seasonal patterns?",
        "Best routes": "Which routes have the best on-time performance and lowest delay risk?"
    }
    
    return expansions.get(suggestion, suggestion)

def categorize_queries(history: list) -> dict:
    """Categorize queries for analytics"""
    
    categories = {
        'Delay Analysis': 0,
        'Congestion': 0,
        'Schedule Optimization': 0,
        'Network Impact': 0,
        'General': 0
    }
    
    for entry in history:
        query = entry['query'].lower()
        
        if 'delay' in query:
            categories['Delay Analysis'] += 1
        elif 'congestion' in query or 'busy' in query:
            categories['Congestion'] += 1
        elif 'schedule' in query or 'time' in query:
            categories['Schedule Optimization'] += 1
        elif 'network' in query or 'impact' in query:
            categories['Network Impact'] += 1
        else:
            categories['General'] += 1
    
    return categories

def show_detailed_analysis():
    """Show detailed analysis in an expanded view"""
    
    with st.expander("📊 Detailed Analysis", expanded=True):
        st.markdown("### 🔍 Deep Dive Analysis")
        
        # Additional charts and data
        col1, col2 = st.columns(2)
        
        with col1:
            # Sample detailed chart
            data = np.random.randn(100)
            fig = px.histogram(data, title="Data Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sample metrics
            st.metric("Confidence Score", "87%")
            st.metric("Data Points", "12,450")
            st.metric("Analysis Time", "2.3s")
        
        st.markdown("**📋 Detailed Recommendations:**")
        st.markdown("- Implement suggested schedule changes within 2 weeks")
        st.markdown("- Monitor impact for 30 days post-implementation")
        st.markdown("- Set up automated alerts for threshold breaches")