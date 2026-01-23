import sys
import os
import argparse
from datetime import datetime

# Add project root to path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nba_api.stats.endpoints import commonteamroster
from src.data_fetch import fetch_daily_scoreboard
from src.train_models import train_model

def train_active_players(date_str=None, force=False):
    """
    Finds all players playing on the given date (default: today) and trains models for them.
    """
    target_date = date_str if date_str else datetime.now().strftime('%Y-%m-%d')
    print(f"Fetching games for {target_date}...")
    
    scoreboard = fetch_daily_scoreboard(target_date)
    if scoreboard.empty:
        print("No games found for this date.")
        return

    team_ids = set(scoreboard['HOME_TEAM_ID'].unique().tolist() + scoreboard['VISITOR_TEAM_ID'].unique().tolist())
    print(f"Found {len(team_ids)} teams playing: {team_ids}")
    
    current_season = "2025-26" 
    players_to_train = []
    
    print("Fetching Rosters...")
    for tid in team_ids:
        try:
            roster = commonteamroster.CommonTeamRoster(team_id=tid, season=current_season).get_data_frames()[0]
            for _, row in roster.iterrows():
                players_to_train.append({
                    'id': row['PLAYER_ID'],
                    'name': row['PLAYER']
                })
        except Exception as e:
            print(f"Error fetching roster for team {tid}: {e}")

    total = len(players_to_train)
    print(f"\nFound {total} players active today.")
    print("Starting Bulk Training... (This may take a while)")
    
    for i, p in enumerate(players_to_train):
        pid = p['id']
        pname = p['name']
        print(f"\n[{i+1}/{total}] Training Model for {pname} (ID: {pid})...")
        try:
            # Check if model exists? 
            # If force is False, maybe skip? But user DELETED them, so we assume we need to train.
            train_model(target_player_id=pid, debug=False, pretrain=False)
        except Exception as e:
            print(f"Failed to train for {pname}: {e}")

    print("\nBulk Training Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, help='YYYY-MM-DD date to train players for (default: today)')
    args = parser.parse_args()
    
    train_active_players(args.date)
