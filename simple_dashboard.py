#!/usr/bin/env python3
"""
Simple Flight Scheduling Analysis Dashboard
"""

import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Flight Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapse sidebar
)

def authenticate():
    """Simple authentication"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if not st.session_state['authenticated']:
        st.title("🔐 Flight Analysis Dashboard Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            
            if login_btn:
                if username in ['admin', 'operator', 'analyst'] and password in ['admin123', 'operator123', 'analyst123']:
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        st.info("Demo: admin/admin123, operator/operator123, analyst/analyst123")
        return False
    
    return True

def show_home():
    """Home page"""
    st.title("✈️ Flight Scheduling Analysis Dashboard")
    st.markdown("Welcome to the AI-powered flight scheduling analysis system")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Flights", "1,247", "23")
    with col2:
        st.metric("Average Delay", "12.5 min", "-2.3 min")
    with col3:
        st.metric("On-Time Rate", "78.2%", "3.1%")
    with col4:
        st.metric("Critical Flights", "34", "-5")
    
    # Sample chart
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
    flight_counts = np.random.poisson(1200, len(dates))
    df = pd.DataFrame({'Date': dates, 'Flights': flight_counts})
    
    fig = px.line(df, x='Date', y='Flights', title="Daily Flight Volume (Last 30 Days)")
    st.plotly_chart(fig, use_container_width=True)

def show_delay_analysis():
    """Delay analysis page"""
    st.title("⏰ Delay Analysis")
    st.markdown("Analyze flight delays and identify optimal scheduling times")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        airport = st.selectbox("Airport", ["Mumbai (BOM)", "Delhi (DEL)", "Bangalore (BLR)"])
    with col2:
        airline = st.selectbox("Airline", ["All", "Air India", "IndiGo", "SpiceJet"])
    with col3:
        days = st.slider("Days Back", 1, 90, 30)
    
    # Sample delay data
    hours = list(range(24))
    delays = np.random.normal(15, 8, 24)
    delays = np.maximum(delays, 0)
    delays[8:10] *= 1.8  # Morning peak
    delays[18:20] *= 1.5  # Evening peak
    
    fig = px.bar(x=hours, y=delays, title="Average Delay by Hour", 
                labels={'x': 'Hour', 'y': 'Delay (minutes)'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("💡 Best times: 6-7 AM (avg 5.2 min delay)")
    st.warning("⚠️ Avoid: 8-10 AM (avg 23.4 min delay)")

def show_congestion_analysis():
    """Congestion analysis page"""
    st.title("🚦 Congestion Analysis")
    st.markdown("Identify peak hours and congestion patterns")
    
    # Sample congestion heatmap
    hours = list(range(24))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    congestion = np.random.rand(len(days), len(hours)) * 10
    
    fig = px.imshow(congestion, x=hours, y=days, 
                   title="Weekly Congestion Pattern",
                   labels={'x': 'Hour', 'y': 'Day', 'color': 'Congestion Level'})
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("🔴 Peak: Mon 8-10 AM (Level 9.2/10)")
    with col2:
        st.info("🟢 Low: Tue 2-4 AM (Level 1.2/10)")

def show_nlp_interface():
    """NLP interface page"""
    st.title("💬 AI Assistant")
    st.markdown("Ask questions about flight data using natural language")
    
    # Query input
    query = st.text_input("Ask your question:", 
                         placeholder="What are the best times to fly from Mumbai to Delhi?")
    
    if st.button("🚀 Ask AI", key="ask_ai_main"):
        if query:
            with st.spinner("AI is analyzing..."):
                # Mock response
                st.markdown("### 🤖 AI Response")
                st.markdown("""
                **Best Time Analysis for Mumbai to Delhi:**
                
                - **Optimal departure:** 6:00-7:00 AM (avg delay: 5.2 min)
                - **Good alternative:** 10:00-11:00 AM (avg delay: 7.8 min)
                - **Avoid:** 8:00-9:00 AM (peak congestion, avg delay: 23.4 min)
                
                **Key insights:**
                - Early morning flights have 67% better on-time performance
                - Weather delays minimal before 10 AM
                - Air traffic peaks 8-10 AM and 6-8 PM
                """)
    
    # Example queries
    st.markdown("### 💡 Example Questions")
    examples = [
        "What are the busiest hours at Mumbai airport?",
        "Which flights cause the most delays?",
        "Best time to schedule a Delhi to Bangalore flight?",
        "How does weather affect delays?"
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"📝 {example}", key=f"example_{i}"):
            st.session_state['query'] = example
            st.rerun()

def main():
    """Main application"""
    
    # Authentication
    if not authenticate():
        return
    
    # Top navigation
    st.markdown(f"**Welcome, {st.session_state.get('username', 'User')}**")
    
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", key="nav_home_simple"):
            st.session_state['page'] = 'Home'
            st.rerun()
    
    with col2:
        if st.button("⏰ Delays", key="nav_delays_simple"):
            st.session_state['page'] = 'Delays'
            st.rerun()
    
    with col3:
        if st.button("🚦 Congestion", key="nav_congestion_simple"):
            st.session_state['page'] = 'Congestion'
            st.rerun()
    
    with col4:
        if st.button("💬 AI Chat", key="nav_chat_simple"):
            st.session_state['page'] = 'Chat'
            st.rerun()
    
    with col5:
        st.metric("System", "🟢 Online")
    
    with col6:
        if st.button("🚪 Logout", key="logout_simple"):
            st.session_state['authenticated'] = False
            st.rerun()
    
    st.markdown("---")
    
    # Page routing
    page = st.session_state.get('page', 'Home')
    
    if page == 'Home':
        show_home()
    elif page == 'Delays':
        show_delay_analysis()
    elif page == 'Congestion':
        show_congestion_analysis()
    elif page == 'Chat':
        show_nlp_interface()

if __name__ == "__main__":
    main()