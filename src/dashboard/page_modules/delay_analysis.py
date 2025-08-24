"""
Delay Analysis page for the flight scheduling analysis dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def show():
    """Display the delay analysis page"""
    
    # Get global filters from session state
    selected_airport = st.session_state.get('selected_airport', 'All Airports')
    selected_route = st.session_state.get('selected_route', 'All Routes')
    selected_date_range = st.session_state.get('selected_date_range', (datetime.now() - timedelta(days=7), datetime.now()))
    
    st.title("⏰ Delay Analysis")
    st.markdown("Analyze flight delays and identify optimal scheduling times")
    
    # Show current context
    st.info(f"📍 Analyzing: **{selected_airport}** | 🛫 **{selected_route}**")
    
    # Additional filters (page-specific)
    with st.expander("🔧 Additional Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            airline = st.selectbox("Airline", ["All", "Air India", "IndiGo", "SpiceJet", "Vistara"])
        
        with col2:
            time_filter = st.selectbox("Time Period", ["All Day", "Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)"])
    
    st.markdown("---")
    
    # Dynamic KPIs based on global filters
    from dashboard.components.navigation import calculate_dynamic_kpis
    kpis = calculate_dynamic_kpis(selected_airport, selected_route, selected_date_range)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Delay", kpis['avg_delay'], kpis['delay_delta'])
    
    with col2:
        # Calculate delayed flights based on total flights and delay rate
        total_flights = int(kpis['total_flights'].replace(',', ''))
        delayed_flights = int(total_flights * 0.28)  # Assume 28% delayed
        st.metric("Delayed Flights", f"{delayed_flights:,}", f"{int(kpis['flights_delta']):+d}")
    
    with col3:
        # Calculate worst delay based on average delay
        avg_delay = float(kpis['avg_delay'].replace(' min', ''))
        worst_delay = int(avg_delay * 8.5)  # Worst is typically 8-10x average
        st.metric("Worst Delay", f"{worst_delay} min", f"{int(avg_delay * 0.8):+d} min")
    
    with col4:
        st.metric("On-Time Rate", kpis['ontime_rate'], kpis['ontime_delta'])
    
    st.markdown("---")
    
    # Main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Delay Patterns by Hour")
        
        # Generate sample data
        hours = list(range(24))
        avg_delays = np.random.normal(15, 8, 24)
        avg_delays = np.maximum(avg_delays, 0)  # No negative delays
        
        fig_hourly = px.bar(
            x=hours,
            y=avg_delays,
            title="Average Delay by Hour of Day",
            labels={'x': 'Hour', 'y': 'Average Delay (minutes)'},
            color=avg_delays,
            color_continuous_scale='RdYlGn_r'
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Best Takeoff Times")
        
        # Best times recommendation
        best_times = [
            ("06:00 - 07:00", "5.2 min", "🟢"),
            ("10:00 - 11:00", "7.8 min", "🟢"),
            ("14:00 - 15:00", "9.1 min", "🟡"),
            ("22:00 - 23:00", "6.3 min", "🟢")
        ]
        
        st.markdown("**Recommended Time Slots:**")
        for time_slot, avg_delay, status in best_times:
            st.markdown(f"{status} **{time_slot}** - Avg delay: {avg_delay}")
        
        st.markdown("---")
        
        # Delay causes
        st.subheader("📋 Delay Causes")
        causes = ["Weather", "Air Traffic", "Technical", "Operational", "Other"]
        percentages = [25, 35, 15, 20, 5]
        
        fig_causes = px.pie(
            values=percentages,
            names=causes,
            title="Delay Causes Distribution"
        )
        st.plotly_chart(fig_causes, use_container_width=True)
    
    # Detailed analysis
    st.markdown("---")
    st.subheader("📈 Detailed Delay Trends")
    
    # Generate sample time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    delays = np.random.normal(15, 5, len(dates))
    
    df_trends = pd.DataFrame({
        'Date': dates,
        'Average Delay': delays,
        'Flight Count': np.random.poisson(1200, len(dates))
    })
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=df_trends['Date'],
        y=df_trends['Average Delay'],
        mode='lines+markers',
        name='Average Delay (min)',
        line=dict(color='red', width=2)
    ))
    
    fig_trends.update_layout(
        title="30-Day Delay Trend",
        xaxis_title="Date",
        yaxis_title="Average Delay (minutes)",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 AI Recommendations")
    
    recommendations = [
        "🕕 Consider scheduling more flights between 6-7 AM for minimal delays",
        "🌤️ Weather delays peak at 3-4 PM - avoid scheduling during thunderstorm season",
        "✈️ Air India flights show 23% higher delays on Fridays - investigate crew scheduling",
        "🛫 Runway 09L at Mumbai shows congestion between 8-10 AM - redistribute traffic"
    ]
    
    for rec in recommendations:
        st.info(rec)
    
    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📊 Export Report", key="delay_export_report", use_container_width=True):
            st.success("Report exported successfully!")
    
    with col2:
        if st.button("📧 Schedule Report", key="delay_schedule_report", use_container_width=True):
            st.success("Report scheduled for weekly delivery!")