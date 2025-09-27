#!/usr/bin/env python3
"""
Environment setup script for Lightpanda Playwright integration
"""

import os
import subprocess
import sys

def check_environment():
    """Check if the environment is properly set up"""
    print("Checking environment setup...")
    
    # Check if LIGHTPANDA_TOKEN is set
    token = os.getenv('LIGHTPANDA_TOKEN')
    if not token:
        print("❌ LIGHTPANDA_TOKEN environment variable is not set")
        print("   Please set it with: export LIGHTPANDA_TOKEN='your_token_here'")
        return False
    else:
        print("✅ LIGHTPANDA_TOKEN is set")
    
    # Check if required packages are installed
    try:
        import playwright
        print("✅ Playwright is installed")
    except ImportError:
        print("❌ Playwright is not installed")
        print("   Install with: pip install playwright")
        return False
    
    try:
        from playwright.sync_api import sync_playwright
        print("✅ Playwright sync API is available")
    except ImportError:
        print("❌ Playwright sync API is not available")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

def install_playwright_browsers():
    """Install Playwright browsers"""
    print("Installing Playwright browsers...")
    try:
        subprocess.check_call([sys.executable, '-m', 'playwright', 'install'])
        print("✅ Playwright browsers installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install Playwright browsers")
        return False

def main():
    """Main setup function"""
    print("Lightpanda Playwright Setup")
    print("=" * 30)
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Install Playwright browsers
    if not install_playwright_browsers():
        return
    
    # Check environment
    if check_environment():
        print("\n🎉 Setup completed successfully!")
        print("You can now run: python main.py")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please address the issues above before running the scripts")

if __name__ == "__main__":
    main()
