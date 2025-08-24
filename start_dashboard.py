#!/usr/bin/env python3
"""
Startup script for the Flight Scheduling Analysis Dashboard
"""

import sys
import os
import subprocess
from pathlib import Path

def main():
    """Start the Streamlit dashboard"""
    
    # Add src directory to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Set environment variables
    os.environ.setdefault('PYTHONPATH', str(src_path))
    
    # Dashboard file path
    dashboard_file = src_path / "dashboard" / "main.py"
    
    if not dashboard_file.exists():
        print(f"Error: Dashboard file not found at {dashboard_file}")
        return 1
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_file),
        "--server.port", "8506",  # Use different port
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("Starting Flight Scheduling Analysis Dashboard...")
    print(f"Command: {' '.join(cmd)}")
    print("Dashboard will be available at: http://localhost:8505")
    print("\nDemo Credentials:")
    print("- Username: admin | Password: admin123 (Full access)")
    print("- Username: operator | Password: operator123 (Operations focus)")
    print("- Username: analyst | Password: analyst123 (Analysis focus)")
    print("\nPress Ctrl+C to stop the dashboard")
    
    try:
        # Run streamlit
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
        return 0
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())