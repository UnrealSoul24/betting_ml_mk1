
import os
import requests
import json
from dotenv import load_dotenv

# Load env from parent dir
load_dotenv(os.path.join(os.path.dirname(__file__), '../../.env'))

API_KEY = os.getenv('SPORTGAMEODDSAPIKEY')
if not API_KEY:
    print("Error: SPORTGAMEODDSAPIKEY not found in .env")
    exit(1)


def analyze_dump():
    dump_path = 'src/dev/nba_odds_dump.json'
    if not os.path.exists(dump_path):
        print("Dump not found.")
        return

    with open(dump_path, 'r') as f:
        data = json.load(f)

    print("Analyzing Dump...")
    events = data.get('data', [])
    if isinstance(events, list) and len(events) > 0:
        event = events[0]
        odds = event.get('odds', {})
        print(f"Total Odds Keys: {len(odds)}")
        
        # Find player props
        player_props = [k for k in odds.keys() if '_NBA-' in k or 'points-' in k]
        print("\n--- Usage Examples ---")
        for i, k in enumerate(player_props[:20]):
            print(f"{k}: {json.dumps(odds[k], indent=2)[:200]}...") # Print first few chars of value

        # specific check for a star player if possible
        # We saw PHILADELPHIA_76ERS_NBA vs HOUSTON_ROCKETS_NBA
        # Look for Embiid or Sengun
        print("\n--- Searching for 'EMBIID' ---")
        embiid_props = [k for k in odds.keys() if 'EMBIID' in k]
        for k in embiid_props[:5]:
            print(k)

if __name__ == "__main__":
    # probe_nba_odds()
    analyze_dump()

