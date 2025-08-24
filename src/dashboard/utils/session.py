"""
Session state management utilities
"""

import streamlit as st
from datetime import datetime
from typing import Any, Dict

def initialize_session_state():
    """Initialize session state variables"""
    
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    
    if 'user_role' not in st.session_state:
        st.session_state['user_role'] = None
    
    # Navigation state
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Home'
    
    # Data cache
    if 'data_cache' not in st.session_state:
        st.session_state['data_cache'] = {}
    
    # Query history for NLP interface
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []
    
    # Dashboard preferences
    if 'dashboard_preferences' not in st.session_state:
        st.session_state['dashboard_preferences'] = {
            'theme': 'light',
            'auto_refresh': False,
            'refresh_interval': 30,
            'default_airport': 'BOM'
        }
    
    # Session metadata
    if 'session_start' not in st.session_state:
        st.session_state['session_start'] = datetime.now()

def get_session_data(key: str, default: Any = None) -> Any:
    """Get data from session state"""
    return st.session_state.get(key, default)

def set_session_data(key: str, value: Any):
    """Set data in session state"""
    st.session_state[key] = value

def clear_session_data(key: str):
    """Clear specific data from session state"""
    if key in st.session_state:
        del st.session_state[key]

def get_cache_data(cache_key: str) -> Any:
    """Get cached data"""
    return st.session_state['data_cache'].get(cache_key)

def set_cache_data(cache_key: str, data: Any, ttl_minutes: int = 30):
    """Set cached data with TTL"""
    st.session_state['data_cache'][cache_key] = {
        'data': data,
        'timestamp': datetime.now(),
        'ttl_minutes': ttl_minutes
    }

def is_cache_valid(cache_key: str) -> bool:
    """Check if cached data is still valid"""
    cache_entry = st.session_state['data_cache'].get(cache_key)
    if not cache_entry:
        return False
    
    elapsed = (datetime.now() - cache_entry['timestamp']).total_seconds() / 60
    return elapsed < cache_entry['ttl_minutes']

def clear_expired_cache():
    """Clear expired cache entries"""
    current_time = datetime.now()
    expired_keys = []
    
    for key, entry in st.session_state['data_cache'].items():
        elapsed = (current_time - entry['timestamp']).total_seconds() / 60
        if elapsed >= entry['ttl_minutes']:
            expired_keys.append(key)
    
    for key in expired_keys:
        del st.session_state['data_cache'][key]

def add_query_to_history(query: str, response: str):
    """Add query and response to history"""
    st.session_state['query_history'].append({
        'timestamp': datetime.now(),
        'query': query,
        'response': response
    })
    
    # Keep only last 50 queries
    if len(st.session_state['query_history']) > 50:
        st.session_state['query_history'] = st.session_state['query_history'][-50:]

def get_query_history() -> list:
    """Get query history"""
    return st.session_state.get('query_history', [])

def update_preferences(preferences: Dict[str, Any]):
    """Update dashboard preferences"""
    st.session_state['dashboard_preferences'].update(preferences)

def get_preferences() -> Dict[str, Any]:
    """Get dashboard preferences"""
    return st.session_state.get('dashboard_preferences', {})