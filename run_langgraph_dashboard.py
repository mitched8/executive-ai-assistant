#!/usr/bin/env python3
"""
🚀 LangGraph Enhanced Dashboard Launcher
Simple script to launch the comprehensive LangGraph dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the LangGraph enhanced dashboard"""
    
    print("🚀 Starting Executive AI Assistant - LangGraph Enhanced Dashboard")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('dashboard_langgraph.py'):
        print("❌ Error: dashboard_langgraph.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if required files exist
    required_files = ['email_assistant.db', 'eaia/main/dashboard_integration.py']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"⚠️  Warning: Missing files: {', '.join(missing_files)}")
        print("Some features may not work correctly.")
    
    # Launch the dashboard
    try:
        print("🌐 Launching dashboard on http://localhost:8502")
        print("📊 Features available:")
        print("   • 🔄 LangGraph Workflow Management")
        print("   • ✅ Human Approval Queue")
        print("   • 🧠 AI Learning & Reflection")
        print("   • 📈 Comprehensive Analytics")
        print("   • ⚙️ System Management")
        print()
        print("Press Ctrl+C to stop the dashboard")
        print("=" * 60)
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "dashboard_langgraph.py", 
            "--server.port", "8502",
            "--server.headless", "true"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 