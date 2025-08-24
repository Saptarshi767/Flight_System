"""
Main Streamlit Dashboard Application

This is the entry point for the flight scheduling analysis dashboard.
"""

import streamlit as st
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

# Import with error handling
try:
    from dashboard.page_modules import home, delay_analysis, congestion_analysis, schedule_impact, nlp_interface
    from dashboard.components.navigation import setup_navigation
    from dashboard.utils.session import initialize_session_state
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(f"Python path: {sys.path}")
    st.error(f"Current working directory: {os.getcwd()}")
    IMPORTS_SUCCESSFUL = False

# Page configuration
st.set_page_config(
    page_title="Flight Scheduling Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/flight-analysis',
        'Report a bug': 'https://github.com/your-repo/flight-analysis/issues',
        'About': "Flight Scheduling Analysis System - AI-powered insights for airport operations"
    }
)

# Hide only Streamlit's automatic multipage navigation
hide_streamlit_style = """
<style>
    /* Hide only the automatic page navigation, not our custom sidebar */
    section[data-testid="stSidebar"] nav[data-testid="stSidebarNav"] {
        display: none !important;
    }
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    """Main application entry point"""
    
    if not IMPORTS_SUCCESSFUL:
        st.title("🚨 Dashboard Startup Error")
        st.error("Failed to import required modules. Please check the installation.")
        st.info("Try running: `pip install streamlit plotly pandas`")
        return
    
    try:
        # Initialize session state
        initialize_session_state()
        
        # Setup sidebar navigation
        setup_navigation()
        
        # Page routing based on sidebar selection
        page = st.session_state.get('current_page', 'Home')
        
        # Main content area
        if page == 'Home':
            home.show()
        elif page == 'Delay Analysis':
            delay_analysis.show()
        elif page == 'Congestion Analysis':
            congestion_analysis.show()
        elif page == 'Schedule Impact':
            schedule_impact.show()
        elif page == 'NLP Interface':
            nlp_interface.show()
        else:
            st.error(f"Unknown page: {page}")
            
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please check the logs for more details.")

if __name__ == "__main__":
    main()