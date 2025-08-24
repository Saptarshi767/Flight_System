"""
Home page for the flight scheduling analysis dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def calculate_filtered_metrics(airport, route, date_range):
    """Calculate metrics based on current filters"""
    
    # Base values
    base_flights = 1247
    base_delay = 12.5
    base_ontime = 78.2
    base_critical = 34
    
    # Airport-specific adjustments
    airport_factors = {
        "All Airports": 1.0,
        "Mumbai (BOM)": 1.3,
        "Delhi (DEL)": 1.2,
        "Bangalore (BLR)": 0.9,
        "Chennai (MAA)": 0.8,
        "Kolkata (CCU)": 0.7,
        "Hyderabad (HYD)": 0.6
    }
    
    # Route-specific adjustments
    route_factors = {
        "All Routes": 1.0,
        "Domestic": 0.85,
        "International": 1.4,
        "BOM-DEL": 1.2,
        "BOM-BLR": 1.0,
        "DEL-BOM": 1.2,
        "BLR-MAA": 0.8,
        "DEL-BLR": 0.9
    }
    
    airport_factor = airport_factors.get(airport, 1.0)
    route_factor = route_factors.get(route, 1.0)
    
    # Calculate adjusted metrics
    flights = int(base_flights * airport_factor * route_factor + np.random.randint(-100, 100))
    delay = round(base_delay * (1 + (airport_factor - 1) * 0.3) + np.random.uniform(-3, 3), 1)
    ontime = round(base_ontime * (2 - airport_factor * 0.3) + np.random.uniform(-5, 5), 1)
    critical = int(base_critical * airport_factor * 0.9 + np.random.randint(-10, 10))
    
    # Generate realistic deltas
    flights_delta = np.random.randint(-50, 80)
    delay_delta = round(np.random.uniform(-4, 4), 1)
    ontime_delta = round(np.random.uniform(-6, 6), 1)
    critical_delta = np.random.randint(-12, 12)
    
    return {
        'total_flights': f"{flights:,}",
        'flights_delta': f"{flights_delta:+d} vs yesterday",
        'avg_delay': f"{delay} min",
        'delay_delta': f"{delay_delta:+.1f} min vs yesterday",
        'ontime_rate': f"{ontime:.1f}%",
        'ontime_delta': f"{ontime_delta:+.1f}% vs yesterday",
        'critical_flights': str(critical),
        'critical_delta': f"{critical_delta:+d} vs yesterday"
    }

def show():
    """Display the home page"""
    
    # Get global filters from session state
    selected_airport = st.session_state.get('selected_airport', 'All Airports')
    selected_route = st.session_state.get('selected_route', 'All Routes')
    selected_date_range = st.session_state.get('selected_date_range', (datetime.now() - timedelta(days=7), datetime.now()))
    
    # Page header with context
    st.title("✈️ Flight Scheduling Analysis Dashboard")
    
    # Show current filter context
    filter_context = f"📍 **{selected_airport}** | 🛫 **{selected_route}**"
    if isinstance(selected_date_range, (list, tuple)) and len(selected_date_range) == 2:
        date_str = f" | 📅 **{selected_date_range[0].strftime('%Y-%m-%d')} to {selected_date_range[1].strftime('%Y-%m-%d')}**"
        filter_context += date_str
    
    st.markdown(filter_context)
    st.markdown("Welcome to the AI-powered flight scheduling analysis system")
    
    # Import the same dynamic KPI function from navigation
    from dashboard.components.navigation import calculate_dynamic_kpis
    
    # Calculate dynamic metrics based on filters
    metrics = calculate_dynamic_kpis(selected_airport, selected_route, selected_date_range)
    
    # Key metrics row
    st.markdown("### 📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Flights",
            value=metrics['total_flights'],
            delta=metrics['flights_delta'],
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            label="Average Delay",
            value=metrics['avg_delay'],
            delta=metrics['delay_delta'],
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="On-Time Performance",
            value=metrics['ontime_rate'],
            delta=metrics['ontime_delta'],
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Critical Flights",
            value=metrics['critical_flights'],
            delta=metrics['critical_delta'],
            delta_color="inverse"
        )
    
    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Flight Volume Trends")
        
        # Generate sample data for flight volume
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        flight_counts = np.random.poisson(1200, len(dates)) + np.random.randint(-100, 100, len(dates))
        
        df_volume = pd.DataFrame({
            'Date': dates,
            'Flight Count': flight_counts
        })
        
        fig_volume = px.line(
            df_volume, 
            x='Date', 
            y='Flight Count',
            title="Daily Flight Volume (Last 30 Days)",
            line_shape='spline'
        )
        fig_volume.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Flights",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔍 Analyze Delays", key="home_analyze_delays", use_container_width=True):
            st.session_state['current_page'] = 'Delay Analysis'
            st.rerun()
        
        if st.button("🚦 Check Congestion", key="home_check_congestion", use_container_width=True):
            st.session_state['current_page'] = 'Congestion Analysis'
            st.rerun()
        
        if st.button("📊 Schedule Impact", key="home_schedule_impact", use_container_width=True):
            st.session_state['current_page'] = 'Schedule Impact'
            st.rerun()
        
        if st.button("💬 Ask AI Assistant", key="home_ask_ai", use_container_width=True):
            st.session_state['current_page'] = 'NLP Interface'
            st.rerun()
        
        st.markdown("---")
        
        st.markdown("### 🏢 Airport Status")
        
        # Dynamic airport status based on selection
        if selected_airport == "All Airports":
            airports = [
                ("Mumbai (BOM)", "🟢", "Normal Operations"),
                ("Delhi (DEL)", "🟡", "Moderate Delays"),
                ("Bangalore (BLR)", "🟢", "Normal Operations"),
                ("Chennai (MAA)", "🔴", "High Congestion"),
                ("Kolkata (CCU)", "🟢", "Normal Operations"),
                ("Hyderabad (HYD)", "🟢", "Normal Operations")
            ]
            
            for airport, status, description in airports:
                st.markdown(f"**{airport}**")
                st.markdown(f"{status} {description}")
                st.markdown("")
        else:
            # Show detailed status for selected airport
            airport_details = get_detailed_airport_status(selected_airport)
            st.markdown(f"**{selected_airport}**")
            st.markdown(f"{airport_details['status_icon']} {airport_details['status']}")
            st.markdown(f"**Current Conditions:** {airport_details['conditions']}")
            st.markdown(f"**Active Runways:** {airport_details['active_runways']}")
            st.markdown(f"**Weather:** {airport_details['weather']}")
            st.markdown(f"**Traffic Level:** {airport_details['traffic_level']}")
    
    # Bottom section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⏰ Delay Distribution")
        
        # Generate sample delay data
        delay_ranges = ['0-5 min', '5-15 min', '15-30 min', '30-60 min', '60+ min']
        delay_counts = [450, 320, 180, 120, 80]
        
        fig_delays = px.pie(
            values=delay_counts,
            names=delay_ranges,
            title="Flight Delay Distribution Today"
        )
        fig_delays.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_delays, use_container_width=True)
    
    with col2:
        st.markdown("### 🕐 Hourly Traffic Pattern")
        
        # Generate sample hourly data
        hours = list(range(24))
        traffic = [50, 30, 20, 15, 25, 45, 80, 120, 150, 140, 130, 125, 
                  135, 145, 155, 160, 170, 180, 165, 140, 120, 100, 80, 65]
        
        fig_hourly = px.bar(
            x=hours,
            y=traffic,
            title="Hourly Flight Traffic Pattern",
            labels={'x': 'Hour of Day', 'y': 'Number of Flights'}
        )
        fig_hourly.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            showlegend=False
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    # System information
    st.markdown("---")
    st.markdown("### ℹ️ System Information")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.info("**Data Sources**\n- FlightRadar24\n- FlightAware\n- Airport Systems")
    
    with info_col2:
        st.info("**Last Updated**\n" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    with info_col3:
        st.info("**AI Models**\n- Delay Prediction: Active\n- Congestion Forecast: Active\n- NLP Assistant: Active")

def get_detailed_airport_status(airport):
    """Get detailed status for a specific airport"""
    
    airport_status = {
        "Mumbai (BOM)": {
            "status_icon": "🟢",
            "status": "Normal Operations",
            "conditions": "Clear skies, good visibility",
            "active_runways": "09L/27R, 09R/27L (14/32 under maintenance)",
            "weather": "28°C, Wind 12 kt from SW",
            "traffic_level": "Moderate (78% capacity)"
        },
        "Delhi (DEL)": {
            "status_icon": "🟡",
            "status": "Moderate Delays",
            "conditions": "Light fog reducing visibility",
            "active_runways": "10/28, 11/29 (others closed for fog)",
            "weather": "18°C, Wind 8 kt from NW, Visibility 2km",
            "traffic_level": "High (92% capacity)"
        },
        "Bangalore (BLR)": {
            "status_icon": "🟢",
            "status": "Normal Operations",
            "conditions": "Excellent weather conditions",
            "active_runways": "09/27, 14/32",
            "weather": "24°C, Wind 6 kt from E",
            "traffic_level": "Normal (65% capacity)"
        },
        "Chennai (MAA)": {
            "status_icon": "🔴",
            "status": "High Congestion",
            "conditions": "Heavy traffic, slot restrictions",
            "active_runways": "07/25 (12/30 closed for repairs)",
            "weather": "32°C, Wind 15 kt from SE",
            "traffic_level": "Very High (98% capacity)"
        },
        "Kolkata (CCU)": {
            "status_icon": "🟢",
            "status": "Normal Operations",
            "conditions": "Good operational conditions",
            "active_runways": "01L/19R, 01R/19L",
            "weather": "26°C, Wind 10 kt from S",
            "traffic_level": "Low (45% capacity)"
        },
        "Hyderabad (HYD)": {
            "status_icon": "🟢",
            "status": "Normal Operations",
            "conditions": "Optimal flying conditions",
            "active_runways": "09L/27R, 09R/27L",
            "weather": "29°C, Wind 8 kt from W",
            "traffic_level": "Normal (58% capacity)"
        }
    }
    
    return airport_status.get(airport, {
        "status_icon": "❓",
        "status": "Status Unknown",
        "conditions": "No data available",
        "active_runways": "Unknown",
        "weather": "No weather data",
        "traffic_level": "Unknown"
    })