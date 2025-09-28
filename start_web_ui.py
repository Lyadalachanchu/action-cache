#!/usr/bin/env python3
"""
Startup script for the Weaviate Agent Web UI
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ['WEAVIATE_URL', 'WEAVIATE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n📝 Please create a .env file with your API keys:")
        print("   WEAVIATE_URL=your_weaviate_url")
        print("   WEAVIATE_API_KEY=your_weaviate_api_key")
        print("   OPENAI_API_KEY=your_openai_api_key")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🚀 Starting Weaviate Agent Web UI")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please set up your .env file.")
        return
    
    print("✅ Environment variables found")
    print("✅ Starting web server...")
    print("\n📱 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Import and run the web UI
    try:
        from web_ui import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error starting web UI: {e}")

if __name__ == "__main__":
    main()
