#!/usr/bin/env python3
"""
ARC AGI Captcha Launcher
Simple script to run the captcha server
"""

import os
import sys
import subprocess

def check_requirements():
    """Check if required packages are installed"""
    try:
        import flask
        import flask_cors
        print("✓ Flask dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependencies: {e}")
        print("Installing requirements...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("✓ Requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install requirements")
            return False

def check_dataset():
    """Check if ARC dataset is available"""
    challenges_path = "dataset/script-tests/grouped-tasks/challenges.json"
    solutions_path = "dataset/script-tests/grouped-tasks/solutions.json"
    
    if os.path.exists(challenges_path) and os.path.exists(solutions_path):
        print("✓ ARC dataset found")
        return True
    else:
        print("✗ ARC dataset not found")
        print(f"Please ensure these files exist:")
        print(f"  - {challenges_path}")
        print(f"  - {solutions_path}")
        return False

def main():
    print("🧠 ARC AGI Captcha Server")
    print("=" * 40)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check dataset
    if not check_dataset():
        return
    
    # Change to arc_captcha directory
    captcha_dir = "arc_captcha"
    if not os.path.exists(captcha_dir):
        print(f"✗ Captcha directory '{captcha_dir}' not found")
        return
    
    print("\n🚀 Starting ARC AGI Captcha server...")
    print("📍 Server will be available at: http://localhost:5001")
    print("📱 Demo page available at: http://localhost:5001/demo")
    print("\n⌨️  Press Ctrl+C to stop the server")
    print("=" * 40)
    
    try:
        # Run the Flask app
        os.chdir(captcha_dir)
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped")
    except Exception as e:
        print(f"\n✗ Error running server: {e}")

if __name__ == "__main__":
    main() 