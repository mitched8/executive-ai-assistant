#!/usr/bin/env python3
"""
Dashboard Launcher
Simple script to launch the Email AI Assistant Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install dashboard requirements if needed"""
    try:
        import streamlit
        import pandas
        import plotly
    except ImportError:
        print("📦 Installing dashboard requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "dashboard_requirements.txt"])
        print("✅ Requirements installed!")

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Email AI Assistant Dashboard...")
    print("📱 Dashboard will open in your default browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    # Set environment variables for better Streamlit experience
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "false"
    os.environ["STREAMLIT_SERVER_PORT"] = "8501"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    
    # Launch Streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.headless", "false",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--theme.base", "light"
    ])

if __name__ == "__main__":
    try:
        install_requirements()
        launch_dashboard()
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {str(e)}")
        print("💡 Try running: streamlit run dashboard.py") 