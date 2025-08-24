"""
Authentication component for the dashboard
"""

import streamlit as st
import hashlib
import os
from typing import Dict, Optional

# Default credentials (in production, use proper authentication system)
DEFAULT_USERS = {
    "admin": "admin123",
    "operator": "operator123",
    "analyst": "analyst123"
}

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hashed

def get_users() -> Dict[str, str]:
    """Get user credentials (hashed passwords)"""
    users = {}
    for username, password in DEFAULT_USERS.items():
        users[username] = hash_password(password)
    return users

def authenticate_user() -> bool:
    """
    Authenticate user with simple login form
    Returns True if authenticated, False otherwise
    """
    
    # Check if user is already authenticated
    if st.session_state.get('authenticated', False):
        return True
    
    # Create login form
    st.title("🔐 Flight Analysis Dashboard Login")
    st.markdown("---")
    
    with st.form("login_form"):
        st.subheader("Please log in to continue")
        
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col2:
            login_button = st.form_submit_button("Login", use_container_width=True)
        
        if login_button:
            if authenticate_credentials(username, password):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.session_state['user_role'] = get_user_role(username)
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
    
    # Show demo credentials
    with st.expander("Demo Credentials", expanded=False):
        st.info("""
        **Demo Users:**
        - Username: `admin` | Password: `admin123` (Full access)
        - Username: `operator` | Password: `operator123` (Operations focus)
        - Username: `analyst` | Password: `analyst123` (Analysis focus)
        """)
    
    return False

def authenticate_credentials(username: str, password: str) -> bool:
    """Verify username and password"""
    if not username or not password:
        return False
    
    users = get_users()
    if username not in users:
        return False
    
    return verify_password(password, users[username])

def get_user_role(username: str) -> str:
    """Get user role based on username"""
    role_mapping = {
        "admin": "Administrator",
        "operator": "Operations Manager", 
        "analyst": "Data Analyst"
    }
    return role_mapping.get(username, "User")

def logout():
    """Logout current user"""
    for key in ['authenticated', 'username', 'user_role']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def get_current_user() -> Optional[str]:
    """Get current authenticated user"""
    return st.session_state.get('username')

def get_current_user_role() -> Optional[str]:
    """Get current user role"""
    return st.session_state.get('user_role')

def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return st.session_state.get('authenticated', False)