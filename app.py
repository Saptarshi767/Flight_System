"""
Simplified Flight Dashboard Entry Point
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Flight Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

def generate_sample_data():
    """Generate sample flight data for demonstration"""
    airports = {
        'BOM': {'name': 'Mumbai', 'flights': 950, 'delay': 18.3, 'ontime': 72.1},
        'DEL': {'name': 'Delhi', 'flights': 1200, 'delay': 22.7, 'ontime': 68.5},
        'BLR': {'name': 'Bangalore', 'flights': 680, 'delay': 8.9, 'ontime': 85.3},
        'MAA': {'name': 'Chennai', 'flights': 520, 'delay': 15.2, 'ontime': 76.8},
        'CCU': {'name': 'Kolkata', 'flights': 380, 'delay': 11.4, 'ontime': 81.2},
        'HYD': {'name': 'Hyderabad', 'flights': 420, 'delay': 9.6, 'ontime': 83.7}
    }
    return airports

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✈️ Flight Scheduling Analysis Dashboard</h1>
        <p>AI-Powered Insights for Airport Operations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### 🎛️ Flight Control Center")
    
    # Generate sample data
    airports_data = generate_sample_data()
    
    # Filters
    selected_airport = st.sidebar.selectbox(
        "🏢 Select Airport",
        options=list(airports_data.keys()),
        format_func=lambda x: f"{x} - {airports_data[x]['name']}"
    )
    
    route_type = st.sidebar.selectbox(
        "🛫 Route Type",
        ["All Routes", "Domestic", "International"]
    )
    
    date_range = st.sidebar.date_input(
        "📅 Date Range",
        value=[datetime.now() - timedelta(days=7), datetime.now()],
        max_value=datetime.now()
    )
    
    # Main content
    airport_info = airports_data[selected_airport]
    
    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Flights",
            value=f"{airport_info['flights']:,}",
            delta="12 vs yesterday"
        )
    
    with col2:
        st.metric(
            label="Average Delay",
            value=f"{airport_info['delay']:.1f} min",
            delta="-2.3 min"
        )
    
    with col3:
        st.metric(
            label="On-Time Performance",
            value=f"{airport_info['ontime']:.1f}%",
            delta="1.2%"
        )
    
    with col4:
        critical_flights = int(airport_info['flights'] * 0.15)
        st.metric(
            label="Critical Flights",
            value=f"{critical_flights}",
            delta="-5"
        )
    
    # Charts
    st.markdown("### 📈 Performance Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Delay distribution chart
        hours = list(range(24))
        delays = [np.random.normal(airport_info['delay'], 5) for _ in hours]
        
        fig = px.line(
            x=hours, 
            y=delays,
            title=f"Hourly Delay Patterns - {airport_info['name']}",
            labels={'x': 'Hour of Day', 'y': 'Average Delay (minutes)'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Airport comparison
        airport_names = [airports_data[code]['name'] for code in airports_data.keys()]
        ontime_rates = [airports_data[code]['ontime'] for code in airports_data.keys()]
        
        fig = px.bar(
            x=airport_names,
            y=ontime_rates,
            title="On-Time Performance Comparison",
            labels={'x': 'Airport', 'y': 'On-Time Rate (%)'},
            color=ontime_rates,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Status indicators
    st.markdown("### 🚦 Airport Status")
    
    status_cols = st.columns(3)
    
    with status_cols[0]:
        if airport_info['ontime'] > 80:
            st.success(f"🟢 {airport_info['name']} - Normal Operations")
        elif airport_info['ontime'] > 70:
            st.warning(f"🟡 {airport_info['name']} - Moderate Delays")
        else:
            st.error(f"🔴 {airport_info['name']} - High Congestion")
    
    with status_cols[1]:
        st.info(f"📊 Daily Capacity: {airport_info['flights']} flights")
    
    with status_cols[2]:
        weather_status = np.random.choice(["Clear", "Cloudy", "Rainy"], p=[0.6, 0.3, 0.1])
        st.info(f"🌤️ Weather: {weather_status}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Flight Scheduling Analysis System | Built with ❤️ for Aviation Industry</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()