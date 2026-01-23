"""
Retrain Trigger - Detects injury changes and identifies players needing retraining.

Key Functions:
- detect_injury_changes(): Main entry point for change detection
- get_affected_players(changes): Maps player names to IDs and finds teammates
- queue_retraining(player_ids): Triggers background retraining jobs
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Set
from nba_api.stats.static import players, teams


DATA_DIR = Path(__file__).parent / "../data"
RAW_DIR = DATA_DIR / "raw"


def get_player_id_by_name(player_name: str) -> int:
    """
    Maps player name to NBA API player ID.
    
    Args:
        player_name: Full player name (e.g., "Joel Embiid")
        
    Returns:
        Player ID (int) or None if not found
    """
    all_players = players.get_players()
    
    # Fuzzy matching (handle middle initials, name variations)
    for p in all_players:
        if p['full_name'].lower() == player_name.lower():
            return p['id']
    
    # Try first + last name only (ignore middle)
    for p in all_players:
        full = p['full_name'].lower()
        search = player_name.lower()
        if full.startswith(search.split()[0]) and full.endswith(search.split()[-1]):
            return p['id']
    
    return None


def get_team_roster(team_id: int, season='2025-26') -> List[int]:
    """
    Fetches all player IDs on a given team's roster.
    
    Args:
        team_id: NBA team ID
        season: NBA season string (e.g., '2025-26')
        
    Returns:
        List of player IDs
    """
    # Load from processed game logs (most reliable source)
    game_logs_files = list(RAW_DIR.glob(f'game_logs_{season}*.csv'))
    
    if not game_logs_files:
        print(f"Warning: No game logs found for {season}")
        return []
    
    import pandas as pd
    df = pd.read_csv(game_logs_files[0])
    
    # Filter by team
    team_players = df[df['TEAM_ID'] == team_id]['PLAYER_ID'].unique().tolist()
    return team_players


def get_player_team(player_id: int) -> int:
    """
    Gets the team ID for a given player.
    
    Args:
        player_id: NBA player ID
        
    Returns:
        Team ID or None
    """
    import pandas as pd
    
    # Check most recent game logs
    game_logs_files = sorted(RAW_DIR.glob('game_logs_*.csv'), reverse=True)
    
    for log_file in game_logs_files:
        df = pd.read_csv(log_file)
        player_rows = df[df['PLAYER_ID'] == player_id]
        
        if not player_rows.empty:
            # Return most recent team
            return player_rows.iloc[-1]['TEAM_ID']
    
    return None


def detect_injury_changes() -> Dict:
    """
    Main entry point: Detects injury changes and returns affected players.
    
    Returns:
        dict with:
        - changes: Raw change detection results
        - affected_player_ids: List of player IDs needing retraining
        - affected_player_names: Dict mapping ID -> name for logging
    """
    from src.injury_service import load_injury_cache, get_injury_report, compare_injury_states, save_injury_cache
    
    # Load cached state
    old_cache = load_injury_cache()
    
    # Fetch current state
    new_report = get_injury_report()
    
    # Compare
    changes = compare_injury_states(old_cache, new_report)
    
    if not changes['all_affected']:
        print("[RetrainTrigger] No injury changes detected.")
        return {
            'changes': changes,
            'affected_player_ids': [],
            'affected_player_names': {}
        }
    
    print(f"[RetrainTrigger] 🚨 {len(changes['all_affected'])} players with status changes")
    
    # Map names to IDs and find affected teammates
    affected_ids = set()
    id_to_name = {}
    
    for player_name in changes['all_affected']:
        player_id = get_player_id_by_name(player_name)
        
        if not player_id:
            print(f"  ⚠️ Could not find ID for: {player_name}")
            continue
        
        print(f"  ✓ {player_name} (ID: {player_id})")
        
        # Add the player themselves
        affected_ids.add(player_id)
        id_to_name[player_id] = player_name
        
        # Find their teammates (usage shifts when star is out)
        team_id = get_player_team(player_id)
        if team_id:
            teammates = get_team_roster(team_id)
            print(f"    → Affecting {len(teammates)} teammates on Team {team_id}")
            
            for teammate_id in teammates:
                affected_ids.add(teammate_id)
                if teammate_id not in id_to_name:
                    id_to_name[teammate_id] = f"Teammate_{teammate_id}"
    
    # Save updated cache
    save_injury_cache(new_report)
    
    return {
        'changes': changes,
        'affected_player_ids': list(affected_ids),
        'affected_player_names': id_to_name
    }


if __name__ == "__main__":
    print("=" * 60)
    print("INJURY CHANGE DETECTION TEST")
    print("=" * 60)
    
    result = detect_injury_changes()
    
    if result['affected_player_ids']:
        print(f"\n📋 {len(result['affected_player_ids'])} players need retraining:")
        for pid in result['affected_player_ids'][:10]:  # Show first 10
            name = result['affected_player_names'].get(pid, f"ID:{pid}")
            print(f"  - {name} ({pid})")
        
        if len(result['affected_player_ids']) > 10:
            print(f"  ... and {len(result['affected_player_ids']) - 10} more")
    else:
        print("\n✅ No retraining needed - injury report unchanged")
