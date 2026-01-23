import pandas as pd
import requests
import io
import json
import os
from datetime import datetime
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent / "../data"
CACHE_DIR = DATA_DIR / "cache"
INJURY_CACHE_PATH = CACHE_DIR / "injury_state.json"
INJURY_ARCHIVE_DIR = CACHE_DIR / "injury_archives"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True, parents=True)
INJURY_ARCHIVE_DIR.mkdir(exist_ok=True, parents=True)

def load_injury_cache():
    """Loads the last saved injury state from cache."""
    if not INJURY_CACHE_PATH.exists():
        return {"last_updated": None, "injuries": {}}
    
    try:
        with open(INJURY_CACHE_PATH, 'r') as f:
            return json.load(f)
    except:
        return {"last_updated": None, "injuries": {}}

def save_injury_cache(injury_report: dict):
    """Saves current injury report to cache with timestamp."""
    cache_data = {
        "last_updated": datetime.now().isoformat(),
        "injuries": injury_report
    }
    
    with open(INJURY_CACHE_PATH, 'w') as f:
        json.dump(cache_data, f, indent=2)
    
    # Also archive this snapshot
    archive_injury_report(injury_report)

def archive_injury_report(report: dict):
    """Archives daily injury snapshot for historical backfilling."""
    today = datetime.now().strftime('%Y-%m-%d')
    archive_path = INJURY_ARCHIVE_DIR / f"{today}.json"
    
    with open(archive_path, 'w') as f:
        json.dump({
            "date": today,
            "injuries": report
        }, f, indent=2)
    
    print(f"[InjuryService] Archived injury report to {archive_path}")

def compare_injury_states(old_state: dict, new_state: dict):
    """
    Compares two injury states and returns what changed.
    
    Returns:
        dict with:
        - new_injuries: Players newly listed as Out
        - returns: Players who were Out, now Active
        - status_changes: Players whose status changed
        - all_affected: Set of all player names affected
    """
    old_injuries = old_state.get("injuries", {})
    new_injuries = new_state
    
    changes = {
        "new_injuries": [],
        "returns": [],
        "status_changes": [],
        "all_affected": set()
    }
    
    # Find new injuries
    for player, status in new_injuries.items():
        if player not in old_injuries:
            if status == "Out":
                changes["new_injuries"].append(player)
                changes["all_affected"].add(player)
        else:
            old_status = old_injuries[player]
            if old_status != status:
                changes["status_changes"].append({
                    "player": player,
                    "old": old_status,
                    "new": status
                })
                changes["all_affected"].add(player)
                
                # Track significant changes
                if status == "Out" and old_status != "Out":
                    changes["new_injuries"].append(player)
    
    # Find returns (players no longer listed)
    for player, status in old_injuries.items():
        if player not in new_injuries and status == "Out":
            changes["returns"].append(player)
            changes["all_affected"].add(player)
    
    return changes

def get_injury_report():
    """
    Fetches the latest NBA injury report from CBS Sports.
    Returns a dictionary: { 'PLAYER_NAME': 'STATUS' }
    Status values: 'Out', 'Questionable', 'Doubtful', 'Game Time Decision'
    """
    url = "https://www.cbssports.com/nba/injuries/"
    
    print(f"Fetching injury report from {url}...")
    
    try:
        # Use simple requests to get html, then pandas to parse
        # Adding headers to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        
        if r.status_code != 200:
            print(f"Failed to fetch injuries. Status: {r.status_code}")
            return {}

        dfs = pd.read_html(io.StringIO(r.text))
        
        # CBS usually separates by team, so multiple tables.
        # We want to combine them all.
        injury_map = {}
        
        for df in dfs:
            if 'Player' in df.columns and 'Injury' in df.columns:
                # Iterate rows
                for _, row in df.iterrows():
                    player_name = str(row['Player']).strip()
                    status = str(row['Injury']).strip() # e.g. "Game Time Decision", "Out", "Knee"
                    
                    # Clean up player name (sometimes "Player Name POS")
                    # CBS format: "LeBron James SF"
                    # But usually just Name. Let's see.
                    # Actually CBS is usually "Player Name"
                    
                    # Normalize Status
                    status_lower = status.lower()
                    clean_status = "Active"
                    
                    if "out" in status_lower:
                        clean_status = "Out"
                    elif "doubtful" in status_lower:
                        clean_status = "Doubtful"
                    elif "questionable" in status_lower:
                        clean_status = "Questionable"
                    elif "decision" in status_lower or "gtd" in status_lower:
                        clean_status = "Questionable"
                    else:
                        # Check specific status columns
                        status_col = None
                        if 'Injury Status' in df.columns:
                            status_col = 'Injury Status'
                        elif 'Status' in df.columns:
                            status_col = 'Status'
                            
                        if status_col:
                            real_status = str(row[status_col]).lower()
                            if "out" in real_status: clean_status = "Out"
                            elif "quest" in real_status: clean_status = "Questionable"
                            elif "doubt" in real_status: clean_status = "Doubtful"
                            elif "day-to-day" in real_status: clean_status = "Questionable"
                            elif "decision" in real_status: clean_status = "Questionable"
                    
                    injury_map[player_name] = clean_status
                    
        print(f"Injury Report Loaded: {len(injury_map)} players found.")
        return injury_map

    except Exception as e:
        print(f"Error fetching injury report: {e}")
        return {}

if __name__ == "__main__":
    # Test cache functionality
    print("Testing injury cache system...")
    
    # Fetch current report
    report = get_injury_report()
    
    # Load old cache
    old_cache = load_injury_cache()
    
    # Compare
    changes = compare_injury_states(old_cache, report)
    
    if changes['all_affected']:
        print(f"\n🚨 INJURY CHANGES DETECTED: {len(changes['all_affected'])} players affected")
        if changes['new_injuries']:
            print(f"  New Injuries/Out: {changes['new_injuries']}")
        if changes['returns']:
            print(f"  Returns: {changes['returns']}")
        if changes['status_changes']:
            print(f"  Status Changes: {len(changes['status_changes'])}")
    else:
        print("\n✅ No injury changes since last check")
    
    # Save new cache
    save_injury_cache(report)
    
    # Show all Out players
    print("\nPlayers currently OUT:")
    for p, s in report.items():
        if s == "Out":
            print(f"  - {p}")
