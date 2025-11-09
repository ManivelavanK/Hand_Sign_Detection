#!/usr/bin/env python3
"""
Quick launcher for the web interface
"""
import subprocess
import sys
import webbrowser
import time
from threading import Timer

def open_browser():
    webbrowser.open('http://localhost:5000')

if __name__ == "__main__":
    print("🚀 Starting Sign Language Detection Web Interface...")
    print("📱 The web interface will open automatically in your browser")
    print("🔗 Manual URL: http://localhost:5000")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Open browser after 2 seconds
    Timer(2.0, open_browser).start()
    
    # Run the Flask app
    try:
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Server stopped")