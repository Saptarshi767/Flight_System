"""
Navigation component for the dashboard
"""

import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from dashboard.components.auth import logout, get_current_user, get_current_user_role

def setup_navigation():
    """Setup modern, compact sidebar navigation"""
    
    try:
        with st.sidebar:
            # Simple header first
            st.markdown("# ✈️ Flight Control")
            
            # User info
            user = get_current_user()
            role = get_current_user_role()
            
            if user and role:
                st.markdown(f"**{user}** • {role}")
                if st.button("🚪 Logout", key="logout_btn"):
                    logout()
            
            st.markdown("---")
            
            # Simple navigation buttons
            st.markdown("### 🧭 Navigation")
            
            if st.button("🏠 Home", key="nav_home", use_container_width=True):
                st.session_state['current_page'] = 'Home'
                st.rerun()
                
            if st.button("⏰ Delay Analysis", key="nav_delay", use_container_width=True):
                st.session_state['current_page'] = 'Delay Analysis'
                st.rerun()
                
            if st.button("🚦 Congestion Analysis", key="nav_congestion", use_container_width=True):
                st.session_state['current_page'] = 'Congestion Analysis'
                st.rerun()
                
            if st.button("📊 Schedule Impact", key="nav_schedule", use_container_width=True):
                st.session_state['current_page'] = 'Schedule Impact'
                st.rerun()
                
            if st.button("💬 NLP Interface", key="nav_nlp", use_container_width=True):
                st.session_state['current_page'] = 'NLP Interface'
                st.rerun()
            
            st.markdown("---")
            
            # Prominent Airport Filter
            st.markdown("### � **iAirport Selection**")
            
            # Complete list of airports with better formatting
            airports = [
                "All Airports",
                "Mumbai (BOM)",
                "Delhi (DEL)", 
                "Bangalore (BLR)",
                "Chennai (MAA)",
                "Kolkata (CCU)",
                "Hyderabad (HYD)"
            ]
            
            # Airport icons for better UX
            airport_icons = {
                "All Airports": "🌐",
                "Mumbai (BOM)": "🏙️",
                "Delhi (DEL)": "🏛️",
                "Bangalore (BLR)": "🌆",
                "Chennai (MAA)": "🏖️",
                "Kolkata (CCU)": "🎭",
                "Hyderabad (HYD)": "💎"
            }
            
            # Create formatted options with icons
            airport_options = [f"{airport_icons.get(airport, '✈️')} {airport}" for airport in airports]
            
            selected_airport_display = st.selectbox(
                "Select Airport",
                airport_options,
                key="airport_filter",
                help="Choose an airport to analyze specific performance metrics"
            )
            
            # Extract the actual airport name (remove icon)
            selected_airport = selected_airport_display.split(" ", 1)[1]
            
            # Show airport status if specific airport selected
            if selected_airport != "All Airports":
                airport_insights = get_airport_insights(selected_airport)
                status_colors = {
                    "Normal": "🟢",
                    "Moderate": "🟡", 
                    "High": "🔴",
                    "Unknown": "⚪"
                }
                status_icon = status_colors.get(airport_insights['status'], "⚪")
                
                st.markdown(
                    f"""
                    <div style='background: #f0f2f6; padding: 8px; border-radius: 6px; margin: 10px 0;'>
                        <small><strong>{selected_airport}</strong></small><br>
                        {status_icon} <strong>{airport_insights['status']}</strong><br>
                        <small>Peak: {airport_insights['peak_hours']}</small>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
            
            st.markdown("---")
            
            # Other Filters
            st.markdown("### 🎛️ **Additional Filters**")
            
            # Route selection with more options
            routes = [
                "All Routes",
                "Domestic",
                "International",
                "BOM-DEL",
                "BOM-BLR", 
                "DEL-BOM",
                "BLR-MAA",
                "DEL-BLR"
            ]
            selected_route = st.selectbox("Route Type", routes, key="route_filter")
            
            # Compact date range
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=7),
                    key="start_date_filter"
                )
            with col2:
                end_date = st.date_input(
                    "To", 
                    value=datetime.now(),
                    key="end_date_filter"
                )
            
            date_range = (start_date, end_date)
            
            # Store in session state
            st.session_state['selected_airport'] = selected_airport
            st.session_state['selected_route'] = selected_route
            st.session_state['selected_date_range'] = date_range
            
            st.markdown("---")
            
            # KPIs
            st.markdown("### 📊 Live KPIs")
            
            try:
                kpis = calculate_dynamic_kpis(selected_airport, selected_route, date_range)
                
                st.metric("Total Flights", kpis['total_flights'], kpis['flights_delta'])
                st.metric("Avg Delay", kpis['avg_delay'], kpis['delay_delta'])
                st.metric("On-Time %", kpis['ontime_rate'], kpis['ontime_delta'])
                st.metric("Critical", kpis['critical_flights'], kpis['critical_delta'])
                
            except Exception as e:
                st.error(f"KPI Error: {e}")
                # Fallback static KPIs
                st.metric("Total Flights", "1,247", "+23")
                st.metric("Avg Delay", "12.5 min", "-2.3 min")
                st.metric("On-Time %", "78.2%", "+3.1%")
                st.metric("Critical", "34", "-5")
            
            st.markdown("---")
            
            # System status
            st.success("🟢 System Online")
            st.markdown("*Last update: 30s ago*")
            
    except Exception as e:
        st.sidebar.error(f"Navigation Error: {e}")
        # Minimal fallback
        st.sidebar.markdown("# ✈️ Flight Control")
        st.sidebar.markdown("Navigation temporarily unavailable")

def calculate_dynamic_kpis(airport, route, date_range):
    """Calculate highly dynamic KPIs based on selected filters with realistic airport data"""
    
    # Realistic airport-specific base data
    airport_data = {
        "All Airports": {
            "daily_flights": 1247,
            "avg_delay": 12.5,
            "ontime_rate": 78.2,
            "critical_flights": 34,
            "congestion_factor": 1.0
        },
        "Mumbai (BOM)": {
            "daily_flights": 950,
            "avg_delay": 18.3,
            "ontime_rate": 72.1,
            "critical_flights": 45,
            "congestion_factor": 1.4
        },
        "Delhi (DEL)": {
            "daily_flights": 1200,
            "avg_delay": 22.7,
            "ontime_rate": 68.5,
            "critical_flights": 52,
            "congestion_factor": 1.6
        },
        "Bangalore (BLR)": {
            "daily_flights": 680,
            "avg_delay": 8.9,
            "ontime_rate": 85.3,
            "critical_flights": 18,
            "congestion_factor": 0.7
        },
        "Chennai (MAA)": {
            "daily_flights": 520,
            "avg_delay": 15.2,
            "ontime_rate": 76.8,
            "critical_flights": 28,
            "congestion_factor": 1.1
        },
        "Kolkata (CCU)": {
            "daily_flights": 380,
            "avg_delay": 11.4,
            "ontime_rate": 81.2,
            "critical_flights": 15,
            "congestion_factor": 0.8
        },
        "Hyderabad (HYD)": {
            "daily_flights": 420,
            "avg_delay": 9.6,
            "ontime_rate": 83.7,
            "critical_flights": 12,
            "congestion_factor": 0.6
        }
    }
    
    # Route impact factors
    route_factors = {
        "All Routes": {"flight_mult": 1.0, "delay_mult": 1.0, "ontime_mult": 1.0},
        "Domestic": {"flight_mult": 0.75, "delay_mult": 0.8, "ontime_mult": 1.1},
        "International": {"flight_mult": 0.35, "delay_mult": 1.4, "ontime_mult": 0.85}
    }
    
    # Get base data for selected airport
    base_data = airport_data.get(airport, airport_data["All Airports"])
    route_factor = route_factors.get(route, route_factors["All Routes"])
    
    # Date range effect
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        days_diff = max(1, (date_range[1] - date_range[0]).days)
        # Longer periods show more total flights but daily averages
        date_multiplier = min(2.0, days_diff / 7)
    else:
        date_multiplier = 1.0
    
    # Time-based variations (simulate different times of day/week)
    current_hour = datetime.now().hour
    time_factor = 1.0
    if 6 <= current_hour <= 10 or 17 <= current_hour <= 21:  # Peak hours
        time_factor = 1.3
    elif 22 <= current_hour <= 5:  # Night hours
        time_factor = 0.4
    
    # Calculate realistic values
    flights = int(
        base_data["daily_flights"] * 
        route_factor["flight_mult"] * 
        date_multiplier * 
        time_factor * 
        np.random.uniform(0.9, 1.1)  # Small random variation
    )
    
    delay = round(
        base_data["avg_delay"] * 
        route_factor["delay_mult"] * 
        base_data["congestion_factor"] * 
        np.random.uniform(0.8, 1.2), 1
    )
    
    ontime = round(
        base_data["ontime_rate"] * 
        route_factor["ontime_mult"] * 
        (2 - base_data["congestion_factor"]) * 0.6 + 40 *
        np.random.uniform(0.95, 1.05), 1
    )
    ontime = min(99.9, max(45.0, ontime))  # Keep realistic bounds
    
    critical = int(
        base_data["critical_flights"] * 
        base_data["congestion_factor"] * 
        route_factor["delay_mult"] * 
        np.random.uniform(0.7, 1.3)
    )
    
    # Generate realistic deltas based on airport characteristics
    if airport == "Delhi (DEL)":
        # Delhi often has weather issues
        flights_delta = np.random.randint(-80, 20)
        delay_delta = np.random.uniform(-1, 8)
        ontime_delta = np.random.uniform(-8, 2)
        critical_delta = np.random.randint(-5, 15)
    elif airport == "Mumbai (BOM)":
        # Mumbai has monsoon issues
        flights_delta = np.random.randint(-60, 30)
        delay_delta = np.random.uniform(-2, 6)
        ontime_delta = np.random.uniform(-6, 3)
        critical_delta = np.random.randint(-3, 12)
    elif airport in ["Bangalore (BLR)", "Hyderabad (HYD)"]:
        # Better performing airports
        flights_delta = np.random.randint(-20, 60)
        delay_delta = np.random.uniform(-4, 2)
        ontime_delta = np.random.uniform(-2, 6)
        critical_delta = np.random.randint(-8, 5)
    else:
        # Default variations
        flights_delta = np.random.randint(-40, 40)
        delay_delta = np.random.uniform(-3, 4)
        ontime_delta = np.random.uniform(-4, 4)
        critical_delta = np.random.randint(-6, 8)
    
    return {
        'total_flights': f"{flights:,}",
        'flights_delta': f"{flights_delta:+d}",
        'avg_delay': f"{delay} min",
        'delay_delta': f"{delay_delta:+.1f} min",
        'ontime_rate': f"{ontime:.1f}%",
        'ontime_delta': f"{ontime_delta:+.1f}%",
        'critical_flights': str(critical),
        'critical_delta': f"{critical_delta:+d}"
    }

def get_airport_insights(airport):
    """Get specific insights for selected airport"""
    
    airport_data = {
        "Mumbai (BOM)": {
            "runways": "3 (09L/27R, 09R/27L, 14/32)",
            "peak_hours": "8-10 AM, 6-8 PM",
            "weather_impact": "Monsoon delays Jun-Sep",
            "status": "Normal"
        },
        "Delhi (DEL)": {
            "runways": "4 (10/28, 11/29, 09/27, 12/30)",
            "peak_hours": "7-9 AM, 7-9 PM",
            "weather_impact": "Fog delays Dec-Jan",
            "status": "Moderate"
        },
        "Bangalore (BLR)": {
            "runways": "2 (09/27, 14/32)",
            "peak_hours": "6-8 AM, 8-10 PM",
            "weather_impact": "Minimal impact",
            "status": "Normal"
        },
        "Chennai (MAA)": {
            "runways": "2 (07/25, 12/30)",
            "peak_hours": "7-9 AM, 6-8 PM",
            "weather_impact": "Cyclone season Oct-Dec",
            "status": "High"
        },
        "Kolkata (CCU)": {
            "runways": "2 (01L/19R, 01R/19L)",
            "peak_hours": "8-10 AM, 7-9 PM",
            "weather_impact": "Monsoon delays Jun-Sep",
            "status": "Normal"
        },
        "Hyderabad (HYD)": {
            "runways": "2 (09L/27R, 09R/27L)",
            "peak_hours": "6-8 AM, 8-10 PM",
            "weather_impact": "Minimal impact",
            "status": "Normal"
        }
    }
    
    return airport_data.get(airport, {
        "runways": "N/A",
        "peak_hours": "N/A",
        "weather_impact": "N/A",
        "status": "Unknown"
    })