#!/usr/bin/env python3
"""
Cache monitoring script for the action-cache project.
Shows cache status, usage statistics, and recent activity.
"""

import sqlite3
import time
import json
from datetime import datetime
from pathlib import Path

def check_cache_status():
    """Check the current status of the cache database."""
    db_path = Path("bday/cachedb/cache.sqlite3")
    
    if not db_path.exists():
        print("❌ Cache database not found at:", db_path)
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        print(f"📊 Cache Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Check all table counts
        tables = ['llm_cache', 'answers', 'plans', 'dom_chunks']
        total_records = 0
        
        for table in tables:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            total_records += count
            status = "✅" if count > 0 else "⚪"
            print(f"{status} {table:15}: {count:3} records")
        
        print(f"\n📈 Total records: {total_records}")
        
        # Check recent activity (last hour)
        one_hour_ago = time.time() - 3600
        cursor.execute('SELECT COUNT(*) FROM llm_cache WHERE ts > ?', (one_hour_ago,))
        recent_hits = cursor.fetchone()[0]
        print(f"🕐 Recent activity (1h): {recent_hits} records")
        
        # Show recent LLM cache entries
        print(f"\n🔍 Recent LLM Cache Entries:")
        print("-" * 40)
        cursor.execute('''
            SELECT ts, prompt_text, output_text, usage_json 
            FROM llm_cache 
            ORDER BY ts DESC 
            LIMIT 3
        ''')
        recent = cursor.fetchall()
        
        for i, (timestamp, prompt, output, usage) in enumerate(recent, 1):
            time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
            prompt_preview = prompt[:50] + "..." if len(prompt) > 50 else prompt
            output_preview = output[:50] + "..." if len(output) > 50 else output
            
            print(f"{i}. [{time_str}]")
            print(f"   Prompt: {prompt_preview}")
            print(f"   Output: {output_preview}")
            
            if usage:
                try:
                    usage_data = json.loads(usage)
                    tokens = usage_data.get('total_tokens', 'N/A')
                    print(f"   Tokens: {tokens}")
                except:
                    print(f"   Usage: {usage}")
            print()
        
        # Check cache hit patterns
        print("📊 Cache Statistics:")
        print("-" * 40)
        
        # Most common prompts
        cursor.execute('''
            SELECT prompt_text, COUNT(*) as count 
            FROM llm_cache 
            GROUP BY prompt_text 
            ORDER BY count DESC 
            LIMIT 3
        ''')
        common_prompts = cursor.fetchall()
        
        if common_prompts:
            print("Most common prompts:")
            for prompt, count in common_prompts:
                prompt_preview = prompt[:40] + "..." if len(prompt) > 40 else prompt
                print(f"  {count}x: {prompt_preview}")
        
        # Check answers cache
        cursor.execute('SELECT COUNT(*) FROM answers WHERE ts > ?', (one_hour_ago,))
        recent_answers = cursor.fetchone()[0]
        print(f"\nRecent answers (1h): {recent_answers}")
        
        # Check plans cache
        cursor.execute('SELECT COUNT(*) FROM plans WHERE ts > ?', (one_hour_ago,))
        recent_plans = cursor.fetchone()[0]
        print(f"Recent plans (1h): {recent_plans}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking cache: {e}")
        return False

def test_cache_operations():
    """Test basic cache operations to verify functionality."""
    print("\n🧪 Testing Cache Operations:")
    print("=" * 40)
    
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.getcwd(), 'bday'))
        
        from cachedb_integrations.cache_adapters import (
            AnswerCacheAdapter, 
            PlanStoreAdapter, 
            LLMCacheAdapter
        )
        
        # Test answer cache
        answer_cache = AnswerCacheAdapter()
        print("✅ Answer cache adapter loaded")
        
        # Test plan store
        plan_store = PlanStoreAdapter()
        print("✅ Plan store adapter loaded")
        
        # Test LLM cache
        llm_cache = LLMCacheAdapter()
        print("✅ LLM cache adapter loaded")
        
        # Test a simple query
        test_question = "Test question for cache monitoring"
        result = answer_cache.get(test_question)
        print(f"✅ Answer cache query test: {'Hit' if result else 'Miss'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing cache operations: {e}")
        print("   (This is expected if running from outside the bday directory)")
        return False

def main():
    """Main monitoring function."""
    print("🔍 Action-Cache Monitoring Tool")
    print("=" * 60)
    
    # Check cache status
    cache_ok = check_cache_status()
    
    if cache_ok:
        # Test cache operations
        test_cache_operations()
        
        print(f"\n✅ Cache monitoring completed successfully!")
        print("💡 Tip: Run this script regularly to monitor cache health")
    else:
        print(f"\n❌ Cache monitoring failed - check database setup")

if __name__ == "__main__":
    main()
