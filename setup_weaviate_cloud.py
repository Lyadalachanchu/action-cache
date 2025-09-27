#!/usr/bin/env python3
"""
Cloud-based Weaviate setup (no Docker required).
This script helps you set up Weaviate using Weaviate Cloud instead of local Docker.
"""

import os
import weaviate
import asyncio
from memory_system import BrowserMemorySystem

def setup_weaviate_cloud():
    """Setup instructions for Weaviate Cloud"""
    print("🌐 Weaviate Cloud Setup")
    print("=" * 30)
    
    print("\n1. Go to https://console.weaviate.cloud/")
    print("2. Create a free account")
    print("3. Create a new cluster")
    print("4. Get your cluster URL and API key")
    print("5. Set environment variables:")
    print("   WEAVIATE_URL=https://your-cluster-url.weaviate.network")
    print("   WEAVIATE_API_KEY=your-api-key-here")
    
    # Check if cloud credentials are set
    weaviate_url = os.getenv('WEAVIATE_URL')
    weaviate_api_key = os.getenv('WEAVIATE_API_KEY')
    
    if not weaviate_url or not weaviate_api_key:
        print("\n⚠️  Cloud credentials not found!")
        print("Please set WEAVIATE_URL and WEAVIATE_API_KEY environment variables")
        return False
    
    print(f"\n✅ Found cloud credentials:")
    print(f"   URL: {weaviate_url}")
    print(f"   API Key: {weaviate_api_key[:10]}...")
    
    return True

async def test_cloud_connection():
    """Test connection to Weaviate Cloud"""
    try:
        print("\n🔍 Testing cloud connection...")
        memory = BrowserMemorySystem(weaviate_url=os.getenv('WEAVIATE_URL'))
        
        # Test basic connection
        stats = memory.get_stats()
        print(f"✅ Successfully connected to Weaviate Cloud!")
        print(f"   Total experiences: {stats['total_experiences']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to Weaviate Cloud: {e}")
        return False

def main():
    """Main setup function for cloud Weaviate"""
    print("🚀 Weaviate Cloud Setup")
    print("=" * 40)
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key:")
        print("  $env:OPENAI_API_KEY = 'sk-your-actual-openai-api-key-here'")
        return False
    
    # Setup cloud credentials
    if not setup_weaviate_cloud():
        return False
    
    # Test connection
    success = asyncio.run(test_cloud_connection())
    
    if success:
        print("\n🎉 Weaviate Cloud setup complete!")
        print("\nNext steps:")
        print("1. Run: python test_memory.py")
        print("2. Run: python enhanced_main.py")
    else:
        print("\n❌ Setup failed. Please check your cloud credentials.")
    
    return success

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

