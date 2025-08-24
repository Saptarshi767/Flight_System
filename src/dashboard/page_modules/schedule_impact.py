"""
Schedule Impact Analysis page for the flight scheduling analysis dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def show():
    """Display the schedule impact analysis page"""
    
    st.title("📊 Schedule Impact Analysis")
    st.markdown("Model the impact of schedule changes on flight delays and network performance")
    
    # Scenario configuration
    st.subheader("🔧 Schedule Change Scenario")
    
    with st.expander("Configure Scenario", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            flight_number = st.text_input("Flight Number", value="AI101")
            current_time = st.time_input("Current Departure", value=datetime.strptime("08:30", "%H:%M").time())
        
        with col2:
            new_time = st.time_input("Proposed Departure", value=datetime.strptime("07:00", "%H:%M").time())
            aircraft_type = st.selectbox("Aircraft Type", ["A320", "B737", "A321", "B777"])
        
        with col3:
            route = st.selectbox("Route", ["BOM-DEL", "DEL-BOM", "BOM-BLR", "DEL-BLR"])
            passenger_count = st.number_input("Passengers", min_value=50, max_value=400, value=180)
        
        if st.button("🔍 Analyze Impact", key="schedule_analyze_impact", use_container_width=True):
            st.success("Analysis complete! Results shown below.")
    
    st.markdown("---")
    
    # Impact summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Delay Reduction", "12.3 min", "-8.7 min")
    
    with col2:
        st.metric("Network Impact", "Medium", "↓ Improved")
    
    with col3:
        st.metric("Affected Flights", "23", "-5")
    
    with col4:
        st.metric("Cost Impact", "$2,340", "-$890")
    
    st.markdown("---")
    
    # Main analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Delay Impact Comparison")
        
        # Before/After comparison
        scenarios = ['Current Schedule', 'Proposed Schedule']
        delays = [15.2, 6.8]
        colors = ['red', 'green']
        
        fig_comparison = px.bar(
            x=scenarios,
            y=delays,
            title="Average Delay Comparison",
            labels={'x': 'Scenario', 'y': 'Average Delay (minutes)'},
            color=scenarios,
            color_discrete_sequence=colors
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Confidence interval
        st.info("📊 **Confidence Level**: 87% (High confidence in prediction)")
    
    with col2:
        st.subheader("🌐 Network Ripple Effect")
        
        # Network impact visualization
        flights = ['AI101', 'AI102', 'AI103', 'AI104', 'AI105']
        current_delays = [15, 8, 12, 18, 6]
        predicted_delays = [7, 5, 8, 10, 4]
        
        fig_network = go.Figure()
        
        fig_network.add_trace(go.Scatter(
            x=flights,
            y=current_delays,
            mode='lines+markers',
            name='Current Delays',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        fig_network.add_trace(go.Scatter(
            x=flights,
            y=predicted_delays,
            mode='lines+markers',
            name='Predicted Delays',
            line=dict(color='green', width=3),
            marker=dict(size=8)
        ))
        
        fig_network.update_layout(
            title="Cascading Impact on Connected Flights",
            xaxis_title="Flight Number",
            yaxis_title="Delay (minutes)"
        )
        
        st.plotly_chart(fig_network, use_container_width=True)
    
    # Detailed impact analysis
    st.markdown("---")
    st.subheader("📋 Detailed Impact Analysis")
    
    # Create tabs for different aspects
    tab1, tab2, tab3, tab4 = st.tabs(["⏰ Time Analysis", "💰 Cost Analysis", "👥 Passenger Impact", "🛫 Operational Impact"])
    
    with tab1:
        st.markdown("#### Time-based Impact Assessment")
        
        # Hourly impact
        hours = list(range(6, 23))
        current_congestion = np.random.rand(len(hours)) * 10
        new_congestion = current_congestion - np.random.rand(len(hours)) * 3
        
        fig_hourly = go.Figure()
        
        fig_hourly.add_trace(go.Bar(
            name='Current Schedule',
            x=hours,
            y=current_congestion,
            marker_color='lightcoral'
        ))
        
        fig_hourly.add_trace(go.Bar(
            name='Proposed Schedule',
            x=hours,
            y=new_congestion,
            marker_color='lightgreen'
        ))
        
        fig_hourly.update_layout(
            title='Hourly Congestion Impact',
            xaxis_title='Hour of Day',
            yaxis_title='Congestion Level',
            barmode='group'
        )
        
        st.plotly_chart(fig_hourly, use_container_width=True)
    
    with tab2:
        st.markdown("#### Cost Impact Assessment")
        
        cost_categories = ['Fuel', 'Crew', 'Passenger Compensation', 'Airport Fees', 'Maintenance']
        current_costs = [1200, 800, 500, 300, 200]
        new_costs = [1000, 750, 200, 280, 180]
        
        cost_df = pd.DataFrame({
            'Category': cost_categories,
            'Current': current_costs,
            'Proposed': new_costs,
            'Savings': [c - n for c, n in zip(current_costs, new_costs)]
        })
        
        fig_costs = px.bar(
            cost_df,
            x='Category',
            y=['Current', 'Proposed'],
            title='Cost Comparison by Category',
            barmode='group'
        )
        
        st.plotly_chart(fig_costs, use_container_width=True)
        
        st.dataframe(cost_df, use_container_width=True)
    
    with tab3:
        st.markdown("#### Passenger Experience Impact")
        
        # Passenger metrics
        metrics = {
            'On-time Arrivals': {'current': '72%', 'proposed': '85%', 'change': '+13%'},
            'Average Wait Time': {'current': '23 min', 'proposed': '15 min', 'change': '-8 min'},
            'Missed Connections': {'current': '12', 'proposed': '6', 'change': '-6'},
            'Satisfaction Score': {'current': '3.2/5', 'proposed': '4.1/5', 'change': '+0.9'}
        }
        
        for metric, values in metrics.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{metric} (Current)", values['current'])
            with col2:
                st.metric(f"{metric} (Proposed)", values['proposed'])
            with col3:
                st.metric("Change", values['change'])
            st.markdown("---")
    
    with tab4:
        st.markdown("#### Operational Impact Assessment")
        
        # Operational factors
        st.markdown("**🛫 Runway Utilization**")
        runway_util = pd.DataFrame({
            'Runway': ['09L/27R', '09R/27L', '14/32'],
            'Current Utilization': [89, 76, 45],
            'Proposed Utilization': [82, 78, 48],
            'Change': [-7, +2, +3]
        })
        st.dataframe(runway_util, use_container_width=True)
        
        st.markdown("**👥 Crew Scheduling Impact**")
        crew_impact = [
            "✅ No additional crew changes required",
            "⚠️ 2 crew rotations need adjustment",
            "✅ Improved crew rest time compliance",
            "⚠️ Gate assignment needs coordination"
        ]
        
        for impact in crew_impact:
            st.markdown(impact)
    
    # Recommendations
    st.markdown("---")
    st.subheader("💡 AI Recommendations")
    
    recommendations = [
        {
            'priority': '🔴 High',
            'action': 'Implement proposed schedule change',
            'reason': 'Significant delay reduction with minimal operational impact',
            'timeline': 'Next scheduling cycle'
        },
        {
            'priority': '🟡 Medium',
            'action': 'Coordinate with ground operations',
            'reason': 'Ensure gate availability at new departure time',
            'timeline': '2 weeks before implementation'
        },
        {
            'priority': '🟢 Low',
            'action': 'Monitor passenger feedback',
            'reason': 'Track satisfaction improvements',
            'timeline': '1 month after implementation'
        }
    ]
    
    for rec in recommendations:
        with st.container():
            col1, col2, col3, col4 = st.columns([1, 3, 3, 2])
            with col1:
                st.markdown(rec['priority'])
            with col2:
                st.markdown(f"**{rec['action']}**")
            with col3:
                st.markdown(rec['reason'])
            with col4:
                st.markdown(rec['timeline'])
    
    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✅ Approve Changes", key="schedule_approve_changes", use_container_width=True):
            st.success("Schedule changes approved and queued for implementation!")
    
    with col2:
        if st.button("📊 Generate Report", key="schedule_generate_report", use_container_width=True):
            st.success("Detailed impact report generated!")
    
    with col3:
        if st.button("🔄 Run Another Scenario", key="schedule_run_scenario", use_container_width=True):
            st.info("Ready for new scenario configuration!")
    
    with col4:
        if st.button("📧 Share Analysis", key="schedule_share_analysis", use_container_width=True):
            st.success("Analysis shared with stakeholders!")