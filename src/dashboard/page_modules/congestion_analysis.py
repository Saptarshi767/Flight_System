"""
Congestion Analysis page for the flight scheduling analysis dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show():
    """Display the congestion analysis page"""
    
    st.title("🚦 Congestion Analysis")
    st.markdown("Identify busy time slots and optimize traffic flow")
    
    # Airport selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        selected_airport = st.selectbox("Select Airport", ["Mumbai (BOM)", "Delhi (DEL)"])
    
    with col2:
        analysis_date = st.date_input("Analysis Date", value=datetime.now())
    
    st.markdown("---")
    
    # Key congestion metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Peak Hour Traffic", "187 flights", "+12")
    
    with col2:
        st.metric("Runway Utilization", "89.2%", "+5.3%")
    
    with col3:
        st.metric("Congestion Score", "7.4/10", "+0.8")
    
    with col4:
        st.metric("Queue Time", "23 min", "+7 min")
    
    st.markdown("---")
    
    # Main visualization - Congestion Heatmap
    st.subheader("🔥 Hourly Congestion Heatmap")
    
    # Generate sample heatmap data
    hours = list(range(24))
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # Create congestion matrix
    congestion_data = np.random.rand(len(days), len(hours)) * 10
    
    # Make certain hours busier (morning and evening peaks)
    for i in range(len(days)):
        congestion_data[i, 7:10] *= 1.5  # Morning peak
        congestion_data[i, 17:20] *= 1.8  # Evening peak
        if i < 5:  # Weekdays busier
            congestion_data[i] *= 1.2
    
    fig_heatmap = px.imshow(
        congestion_data,
        x=hours,
        y=days,
        color_continuous_scale='RdYlGn_r',
        title="Weekly Congestion Pattern",
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Congestion Level'}
    )
    
    fig_heatmap.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=2),
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Two column layout for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Peak Hours Analysis")
        
        # Peak hours data
        peak_hours = list(range(6, 23))
        traffic_volume = [45, 78, 156, 187, 165, 142, 138, 145, 152, 168, 
                         175, 182, 178, 165, 158, 145, 132]
        
        fig_peak = px.bar(
            x=peak_hours,
            y=traffic_volume,
            title="Traffic Volume by Hour",
            labels={'x': 'Hour of Day', 'y': 'Number of Flights'},
            color=traffic_volume,
            color_continuous_scale='RdYlGn_r'
        )
        
        # Add threshold line
        fig_peak.add_hline(y=150, line_dash="dash", line_color="red", 
                          annotation_text="Congestion Threshold")
        
        st.plotly_chart(fig_peak, use_container_width=True)
    
    with col2:
        st.subheader("🛫 Runway Utilization")
        
        # Runway data
        runways = ['09L/27R', '09R/27L', '14/32']
        utilization = [89, 76, 45]
        capacity = [100, 100, 100]
        
        fig_runway = go.Figure()
        
        fig_runway.add_trace(go.Bar(
            name='Current Utilization',
            x=runways,
            y=utilization,
            marker_color='lightblue'
        ))
        
        fig_runway.add_trace(go.Bar(
            name='Remaining Capacity',
            x=runways,
            y=[cap - util for cap, util in zip(capacity, utilization)],
            marker_color='lightgray'
        ))
        
        fig_runway.update_layout(
            barmode='stack',
            title='Runway Capacity Utilization',
            yaxis_title='Utilization %'
        )
        
        st.plotly_chart(fig_runway, use_container_width=True)
    
    # Congestion forecast
    st.markdown("---")
    st.subheader("🔮 Congestion Forecast")
    
    # Generate forecast data
    forecast_hours = list(range(24))
    current_congestion = np.random.rand(24) * 8 + 2
    predicted_congestion = current_congestion + np.random.normal(0, 0.5, 24)
    
    fig_forecast = go.Figure()
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_hours,
        y=current_congestion,
        mode='lines+markers',
        name='Current Pattern',
        line=dict(color='blue', width=2)
    ))
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_hours,
        y=predicted_congestion,
        mode='lines+markers',
        name='Tomorrow Forecast',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Add congestion zones
    fig_forecast.add_hrect(y0=0, y1=3, fillcolor="green", opacity=0.2, 
                          annotation_text="Low Congestion", annotation_position="top left")
    fig_forecast.add_hrect(y0=3, y1=7, fillcolor="yellow", opacity=0.2,
                          annotation_text="Moderate Congestion", annotation_position="top left")
    fig_forecast.add_hrect(y0=7, y1=10, fillcolor="red", opacity=0.2,
                          annotation_text="High Congestion", annotation_position="top left")
    
    fig_forecast.update_layout(
        title="24-Hour Congestion Forecast",
        xaxis_title="Hour of Day",
        yaxis_title="Congestion Level",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Recommendations and alerts
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚠️ Congestion Alerts")
        
        alerts = [
            ("🔴 High", "8:00-10:00 AM", "Runway 09L approaching capacity"),
            ("🟡 Medium", "6:00-7:00 PM", "Evening rush building up"),
            ("🟡 Medium", "2:00-3:00 PM", "Weather delays expected"),
            ("🟢 Low", "11:00 PM-5:00 AM", "Optimal scheduling window")
        ]
        
        for severity, time_slot, message in alerts:
            st.markdown(f"{severity} **{time_slot}**: {message}")
    
    with col2:
        st.subheader("💡 Optimization Suggestions")
        
        suggestions = [
            "🕘 Reschedule 15 flights from 8-9 AM to 6-7 AM slot",
            "🛫 Use runway 14/32 for smaller aircraft during peak hours",
            "⏰ Implement slot coordination with neighboring airports",
            "🌤️ Prepare contingency plans for weather delays at 2 PM"
        ]
        
        for suggestion in suggestions:
            st.info(suggestion)
    
    # Export and actions
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Analysis", key="congestion_export_analysis", use_container_width=True):
            st.success("Congestion analysis exported!")
    
    with col2:
        if st.button("📧 Send Alerts", key="congestion_send_alerts", use_container_width=True):
            st.success("Alerts sent to operations team!")
    
    with col3:
        if st.button("🔄 Refresh Data", key="congestion_refresh_data", use_container_width=True):
            st.success("Data refreshed!")
    
    with col4:
        if st.button("⚙️ Configure Thresholds", key="congestion_configure_thresholds", use_container_width=True):
            st.info("Threshold configuration opened!")