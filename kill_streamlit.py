#!/usr/bin/env python3
"""
Kill all Streamlit processes
"""

import subprocess
import sys

def kill_streamlit_processes():
    """Kill all running Streamlit processes"""
    try:
        # Find all Python processes running Streamlit
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines[1:]:  # Skip header
                if 'streamlit' in line.lower():
                    # Extract PID (second column in CSV)
                    parts = line.split(',')
                    if len(parts) >= 2:
                        pid = parts[1].strip('"')
                        print(f"Killing Streamlit process PID: {pid}")
                        subprocess.run(['taskkill', '/PID', pid, '/F'], 
                                     capture_output=True)
        
        # Also try to kill by port
        ports = ['8501', '8502', '8503', '8504']
        for port in ports:
            try:
                result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if f':{port}' in line and 'LISTENING' in line:
                        parts = line.split()
                        if parts:
                            pid = parts[-1]
                            print(f"Killing process on port {port}, PID: {pid}")
                            subprocess.run(['taskkill', '/PID', pid, '/F'], 
                                         capture_output=True)
            except:
                pass
        
        print("All Streamlit processes killed successfully!")
        return True
        
    except Exception as e:
        print(f"Error killing processes: {e}")
        return False

if __name__ == "__main__":
    kill_streamlit_processes()