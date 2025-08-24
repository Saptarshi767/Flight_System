#!/usr/bin/env python3
"""
Installation script for Flight Scheduling Analysis System dependencies
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    """Main installation function"""
    print("🚀 Installing Flight Scheduling Analysis System Dependencies...")
    print("=" * 60)
    
    # Essential packages for the dashboard
    essential_packages = [
        "streamlit>=1.28.0",
        "plotly>=5.17.0", 
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "reportlab>=4.0.0",
        "Pillow>=10.0.0",
        "jinja2>=3.1.0",
        "python-dotenv>=1.0.0"
    ]
    
    # Optional packages
    optional_packages = [
        "openpyxl>=3.1.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0"
    ]
    
    print("Installing essential packages...")
    failed_essential = []
    for package in essential_packages:
        if not install_package(package):
            failed_essential.append(package)
    
    print("\nInstalling optional packages...")
    failed_optional = []
    for package in optional_packages:
        if not install_package(package):
            failed_optional.append(package)
    
    print("\n" + "=" * 60)
    print("📋 Installation Summary:")
    
    if not failed_essential:
        print("✅ All essential packages installed successfully!")
    else:
        print(f"❌ Failed essential packages: {', '.join(failed_essential)}")
    
    if failed_optional:
        print(f"⚠️  Failed optional packages: {', '.join(failed_optional)}")
    else:
        print("✅ All optional packages installed successfully!")
    
    print("\n🎉 Installation complete!")
    print("You can now run the dashboard with: streamlit run src/dashboard/main.py")

if __name__ == "__main__":
    main()