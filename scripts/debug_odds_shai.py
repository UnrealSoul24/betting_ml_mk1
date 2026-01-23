
import sys
import os
import re
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.odds_service import _player_name_to_api_id, _load_from_cache, fetch_nba_odds, get_player_prop_odds

def debug_shai():
    name = "Shai Gilgeous-Alexander"
    api_id = _player_name_to_api_id(name)
    print(f"Name: {name}")
    print(f"Generated API ID: {api_id}")

    # Load cache or fetch
    data = fetch_nba_odds()
    if not data:
        print("No odds data available.")
        return

    print(f"\nSearching for Shai in {len(data.get('data', []))} events...")
    
    found_any = False
    for event in data.get('data', []):
        odds = event.get('odds', {})
        # Just dump some keys to see format
        keys = list(odds.keys())
        # Search for "SHAI" in keys
        shai_keys = [k for k in keys if "SHAI" in k.upper()]
        if shai_keys:
            print(f"Found partial matches for SHAI: {shai_keys[:5]}...")
            found_any = True
            break
    
    if not found_any:
        print("No keys containing 'SHAI' found in odds data.")

    # Try specific prop
    res = get_player_prop_odds(name, "PTS")
    print(f"\nget_player_prop_odds result: {res}")

if __name__ == "__main__":
    debug_shai()
