#!/usr/bin/env python3
"""
Setup script for Weaviate with OpenAI embeddings.
This script helps you get Weaviate running locally with the proper configuration.
"""

import subprocess
import sys
import os
import time
import requests
from pathlib import Path

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is installed")
            return True
        else:
            print("❌ Docker is not installed or not running")
            return False
    except FileNotFoundError:
        print("❌ Docker is not installed")
        return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker', 'compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker Compose is available")
            return True
        else:
            print("❌ Docker Compose is not available")
            return False
    except FileNotFoundError:
        print("❌ Docker Compose is not available")
        return False

def create_docker_compose():
    """Create docker-compose.yml for Weaviate with OpenAI"""
    docker_compose_content = """version: '3.4'
services:
  weaviate:
    command:
    - --host
    - 0.0.0.0
    - --port
    - '8080'
    - --scheme
    - http
    image: semitechnologies/weaviate:1.25.0
    ports:
    - 8080:8080
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-openai'
      ENABLE_MODULES: 'text2vec-openai'
      CLUSTER_HOSTNAME: 'node1'
      OPENAI_APIKEY: '${OPENAI_API_KEY}'
    volumes:
    - weaviate_data:/var/lib/weaviate
volumes:
  weaviate_data:
"""
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose_content)
    
    print("✅ Created docker-compose.yml")

def start_weaviate():
    """Start Weaviate using Docker Compose"""
    print("🚀 Starting Weaviate...")
    
    # Set environment variable for OpenAI API key
    env = os.environ.copy()
    if 'OPENAI_API_KEY' not in env:
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key: export OPENAI_API_KEY=your_key_here")
        return False
    
    try:
        # Start Weaviate
        result = subprocess.run(['docker', 'compose', 'up', '-d'], 
                              capture_output=True, text=True, env=env)
        
        if result.returncode == 0:
            print("✅ Weaviate started successfully")
            return True
        else:
            print(f"❌ Failed to start Weaviate: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error starting Weaviate: {e}")
        return False

def wait_for_weaviate():
    """Wait for Weaviate to be ready"""
    print("⏳ Waiting for Weaviate to be ready...")
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:8080/v1/meta', timeout=5)
            if response.status_code == 200:
                print("✅ Weaviate is ready!")
                return True
        except:
            pass
        
        time.sleep(2)
        print(f"   Attempt {attempt + 1}/{max_attempts}...")
    
    print("❌ Weaviate failed to start within timeout")
    return False

def test_weaviate_connection():
    """Test connection to Weaviate"""
    try:
        import weaviate
        client = weaviate.Client(url="http://localhost:8080")
        
        # Test basic connection
        meta = client.schema.get()
        print("✅ Successfully connected to Weaviate")
        print(f"   Available classes: {list(meta.get('classes', []))}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate: {e}")
        return False

def main():
    """Main setup function"""
    print("🔧 Setting up Weaviate for Browser Memory System")
    print("=" * 50)
    
    # Check prerequisites
    if not check_docker():
        print("\n❌ Docker not found!")
        print("\nYou have two options:")
        print("1. Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        print("2. Use Weaviate Cloud (no Docker required): python setup_weaviate_cloud.py")
        print("\nFor Docker setup instructions, see: DOCKER_SETUP.md")
        return False
    
    if not check_docker_compose():
        print("\n❌ Please install Docker Compose first")
        return False
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key using one of these methods:")
        print("\nMethod 1 - PowerShell (temporary):")
        print("  $env:OPENAI_API_KEY = 'sk-your-actual-openai-api-key-here'")
        print("\nMethod 2 - Create .env file (recommended):")
        print("  Create a .env file with: OPENAI_API_KEY=sk-your-actual-openai-api-key-here")
        print("\nMethod 3 - System Environment Variables (permanent):")
        print("  Add OPENAI_API_KEY to your system environment variables")
        print("\nGet your API key at: https://platform.openai.com/api-keys")
        return False
    
    # Create docker-compose file
    create_docker_compose()
    
    # Start Weaviate
    if not start_weaviate():
        return False
    
    # Wait for Weaviate to be ready
    if not wait_for_weaviate():
        return False
    
    # Test connection
    if not test_weaviate_connection():
        return False
    
    print("\n🎉 Setup complete! Weaviate is running and ready to use.")
    print("\nNext steps:")
    print("1. Run: python enhanced_main.py")
    print("2. The memory system will automatically store and retrieve experiences")
    print("3. Stop Weaviate with: docker compose down")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

