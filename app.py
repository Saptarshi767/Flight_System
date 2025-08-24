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

def load_demo_data():
    """Load demo data from JSON file or generate if not exists"""
    try:
        import json
        with open('demo_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback to basic data if demo_data.json doesn't exist
        return {
            'airports': {
                'BOM': {'name': 'Mumbai', 'daily_flights': 950, 'avg_delay': 18.3, 'ontime_rate': 72.1, 'status': 'Moderate', 'status_color': '🟡'},
                'DEL': {'name': 'Delhi', 'daily_flights': 1200, 'avg_delay': 22.7, 'ontime_rate': 68.5, 'status': 'High Congestion', 'status_color': '🔴'},
                'BLR': {'name': 'Bangalore', 'daily_flights': 680, 'avg_delay': 8.9, 'ontime_rate': 85.3, 'status': 'Normal', 'status_color': '🟢'},
                'MAA': {'name': 'Chennai', 'daily_flights': 520, 'avg_delay': 15.2, 'ontime_rate': 76.8, 'status': 'Moderate', 'status_color': '🟡'},
                'CCU': {'name': 'Kolkata', 'daily_flights': 380, 'avg_delay': 11.4, 'ontime_rate': 81.2, 'status': 'Normal', 'status_color': '🟢'},
                'HYD': {'name': 'Hyderabad', 'daily_flights': 420, 'avg_delay': 9.6, 'ontime_rate': 83.7, 'status': 'Normal', 'status_color': '🟢'}
            },
            'hourly_delays': {}
        }

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
    
    # Load demo data
    demo_data = load_demo_data()
    airports_data = demo_data['airports']
    
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
            value=f"{airport_info['daily_flights']:,}",
            delta="12 vs yesterday"
        )
    
    with col2:
        st.metric(
            label="Average Delay",
            value=f"{airport_info['avg_delay']:.1f} min",
            delta="-2.3 min"
        )
    
    with col3:
        st.metric(
            label="On-Time Performance",
            value=f"{airport_info['ontime_rate']:.1f}%",
            delta="1.2%"
        )
    
    with col4:
        critical_flights = int(airport_info['daily_flights'] * 0.15)
        st.metric(
            label="Critical Flights",
            value=f"{critical_flights}",
            delta="-5"
        )
    
    # Delay Analysis Section
    st.markdown("### 📈 Delay Analysis by Airport")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Hourly delay patterns for selected airport
        hours = list(range(24))
        if 'hourly_delays' in demo_data and selected_airport in demo_data['hourly_delays']:
            delays = demo_data['hourly_delays'][selected_airport]
        else:
            delays = [np.random.normal(airport_info['avg_delay'], 5) for _ in hours]
        
        fig = px.line(
            x=hours, 
            y=delays,
            title=f"Hourly Delay Patterns - {airport_info['name']} Airport ({selected_airport})",
            labels={'x': 'Hour of Day', 'y': 'Average Delay (minutes)'}
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            showlegend=False
        )
        fig.update_traces(line=dict(color='#2a5298', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average delay comparison across airports
        airport_names = [airports_data[code]['name'] for code in airports_data.keys()]
        airport_codes = list(airports_data.keys())
        avg_delays = [airports_data[code]['avg_delay'] for code in airports_data.keys()]
        
        # Create labels with airport names only (no company names)
        airport_labels = [f"{airports_data[code]['name']}\n({code})" for code in airports_data.keys()]
        
        fig = px.bar(
            x=airport_labels,
            y=avg_delays,
            title="Average Delay Comparison by Airport",
            labels={'x': 'Airport', 'y': 'Average Delay (minutes)'},
            color=avg_delays,
            color_continuous_scale='RdYlBu_r'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional delay analysis
    st.markdown("### 📊 Detailed Delay Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # On-time performance by airport
        airport_names = [airports_data[code]['name'] for code in airports_data.keys()]
        ontime_rates = [airports_data[code]['ontime_rate'] for code in airports_data.keys()]
        
        fig = px.bar(
            x=airport_names,
            y=ontime_rates,
            title="On-Time Performance by Airport",
            labels={'x': 'Airport', 'y': 'On-Time Rate (%)'},
            color=ontime_rates,
            color_continuous_scale='RdYlGn'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Delay severity distribution for selected airport
        delay_categories = ['On-Time (0-5 min)', 'Minor Delay (5-15 min)', 'Moderate Delay (15-30 min)', 'Major Delay (30+ min)']
        
        # Calculate distribution based on airport performance
        ontime_rate = airport_info['ontime_rate']
        minor_delay = (100 - ontime_rate) * 0.5
        moderate_delay = (100 - ontime_rate) * 0.3
        major_delay = (100 - ontime_rate) * 0.2
        
        delay_percentages = [ontime_rate, minor_delay, moderate_delay, major_delay]
        
        fig = px.pie(
            values=delay_percentages,
            names=delay_categories,
            title=f"Delay Distribution - {airport_info['name']} Airport",
            color_discrete_sequence=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Airport delay summary table
    st.markdown("### 📋 Airport Delay Summary")
    
    # Create a comprehensive delay analysis table with airport names only
    delay_summary_data = []
    for code, info in airports_data.items():
        delay_summary_data.append({
            'Airport Code': code,
            'Airport Name': info['name'],
            'Daily Flights': f"{info['daily_flights']:,}",
            'Average Delay (min)': f"{info['avg_delay']:.1f}",
            'On-Time Rate (%)': f"{info['ontime_rate']:.1f}%",
            'Status': info.get('status', 'Normal'),
            'Performance Rating': '⭐⭐⭐⭐⭐' if info['ontime_rate'] > 85 else '⭐⭐⭐⭐' if info['ontime_rate'] > 75 else '⭐⭐⭐' if info['ontime_rate'] > 65 else '⭐⭐'
        })
    
    delay_df = pd.DataFrame(delay_summary_data)
    
    # Style the dataframe
    st.dataframe(
        delay_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Best and worst performing airports
    col_best, col_worst = st.columns(2)
    
    with col_best:
        best_airport = max(airports_data.items(), key=lambda x: x[1]['ontime_rate'])
        st.success(f"🏆 **Best Performance**: {best_airport[1]['name']} ({best_airport[0]}) - {best_airport[1]['ontime_rate']:.1f}% on-time")
    
    with col_worst:
        worst_airport = min(airports_data.items(), key=lambda x: x[1]['ontime_rate'])
        st.error(f"⚠️ **Needs Attention**: {worst_airport[1]['name']} ({worst_airport[0]}) - {worst_airport[1]['ontime_rate']:.1f}% on-time")
    
    # Status indicators
    st.markdown("### 🚦 Airport Status")
    
    status_cols = st.columns(3)
    
    with status_cols[0]:
        status_color = airport_info.get('status_color', '🟢')
        status_text = airport_info.get('status', 'Normal')
        if airport_info['ontime_rate'] > 80:
            st.success(f"{status_color} {airport_info['name']} - {status_text}")
        elif airport_info['ontime_rate'] > 70:
            st.warning(f"{status_color} {airport_info['name']} - {status_text}")
        else:
            st.error(f"{status_color} {airport_info['name']} - {status_text}")
    
    with status_cols[1]:
        st.info(f"📊 Daily Capacity: {airport_info['daily_flights']} flights")
    
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