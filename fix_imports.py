#!/usr/bin/env python3
"""
Quick fix script for missing import dependencies
"""

import subprocess
import sys

def install_missing_packages():
    """Install commonly missing packages"""
    packages = [
        "reportlab",
        "Pillow", 
        "jinja2",
        "streamlit",
        "plotly",
        "pandas"
    ]
    
    print("🔧 Installing missing packages...")
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print(f"✅ {package}")
        except subprocess.CalledProcessError:
            print(f"❌ {package} (failed)")
    
    print("\n✨ Done! Try running your dashboard again.")

if __name__ == "__main__":
    install_missing_packages()