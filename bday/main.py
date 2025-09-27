# main.py
import sys
import argparse
from cachedb.db import init_db
from cachedb.migrate_from_json import run as migrate_run

from agent import LLMBrowserAgent

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Run the LLMBrowserAgent against a goal")
    parser.add_argument("goal_words", nargs="*", help="Goal text if not provided via --goal")
    parser.add_argument("--goal", dest="goal", help="Goal text for the agent")
    parser.add_argument("--headful", action="store_true", help="Reserved flag for browser mode")
    parser.add_argument("--llm-policy", dest="llm_policy", help="Reserved flag to control LLM policy")
    return parser.parse_args(argv)

def main(argv=None):
    args = parse_args(argv or sys.argv[1:])

    # Initialize cache DB
    init_db()
    try:
        migrate_run()  # migrate old JSON caches (safe to run even if missing)
    except Exception as e:
        print("Cache migration skipped:", e, file=sys.stderr)

    goal_parts = args.goal if args.goal else " ".join(args.goal_words)
    goal = goal_parts.strip() or "When was Marie Curie born?"

    agent = LLMBrowserAgent()
    agent.run(goal)

if __name__ == "__main__":
    main()
