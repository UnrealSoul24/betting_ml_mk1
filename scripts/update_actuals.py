
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Setup Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calibration_tracker import CalibrationTracker
from src.data_fetch import fetch_player_game_logs

def update_actuals():
    tracker = CalibrationTracker()
    df_history = tracker.load_history(days=365) # Load plenty
    
    # Identify pending
    pending = df_history[df_history['actual'].isna()]
    
    if pending.empty:
        print("No pending predictions to update.")
        return
        
    print(f"Found {len(pending)} pending predictions. Fetching latest game logs...")
    
    # Fetch logs for current season (assuming current context is 2025-26)
    # We might need multiple seasons if pending spans seasons, but likely just current.
    # Current season logic:
    now = datetime.now()
    # Simple logic: if month > 9 (Oct), year-nextyear. Else prevyear-year.
    if now.month >= 10:
        season = f"{now.year}-{str(now.year+1)[-2:]}"
    else:
        season = f"{now.year-1}-{str(now.year)[-2:]}"
        
    # Hardcode/Config override if needed, but lets assume 2025-26 as per user context
    season = "2025-26" 
    
    # Fetch logs
    logs = fetch_player_game_logs(season=season)
    
    if logs.empty:
        print("No game logs found from API.")
        return
        
    # Preprocess logs
    # Logs have: PLAYER_ID, GAME_DATE (string), PTS, REB, AST, FG3M, BLK, STL
    # Ensure date format matches history
    logs['GAME_DATE'] = pd.to_datetime(logs['GAME_DATE']).dt.strftime('%Y-%m-%d')
    cols_map = {
        'PTS': 'PTS', 'REB': 'REB', 'AST': 'AST',
        'FG3M': '3PM', 'BLK': 'BLK', 'STL': 'STL'
    }
    
    # Create a unified lookup map: (PID, DATE) -> {Stat: Val}
    # But prediction history is "Stat Type" based.
    # We can iterate pending and lookup.
    
    # Optimized:
    # Set index on logs for fast lookup
    logs.set_index(['PLAYER_ID', 'GAME_DATE'], inplace=True)
    
    updated_count = 0
    
    # We need to update the CSV. 
    # CalibrationTracker.save_prediction appends. 
    # It doesn't update.
    # We need to rewrite the history file or append a configured "Update"?
    # The requirement said "Update prediction history CSV with actual values".
    # Since save_prediction only appends, we might need a method to rewrite the file or we implement it here.
    
    # Let's load the FULL history, update in memory, and rewrite.
    full_history = pd.read_csv(tracker.history_file)
    
    for idx, row in full_history.iterrows():
        if pd.notna(row['actual']):
            continue
            
        pid = row['player_id']
        date = row['game_date']
        stat = row['stat_type']
        
        try:
            if (pid, date) in logs.index:
                # Handle duplicate games? NBA API usually unique per day unless double header (rare/handled)
                # logs.loc might return DataFrame if multiple matches
                log_row = logs.loc[(pid, date)]
                if isinstance(log_row, pd.DataFrame):
                    log_row = log_row.iloc[0]
                
                # Map stat type
                # Stat types in history: PTS, REB, AST, 3PM, BLK, STL
                # Unify
                if stat == '3PM': 
                    val = log_row['FG3M']
                elif stat in cols_map:
                    val = log_row[cols_map[stat]]
                else:
                    continue
                    
                full_history.at[idx, 'actual'] = val
                updated_count += 1
        except Exception as e:
            pass
            
    if updated_count > 0:
        # Backup
        import shutil
        shutil.copy(tracker.history_file, tracker.history_file + ".bak")
        
        # Save
        full_history.to_csv(tracker.history_file, index=False)
        print(f"Updated {updated_count} predictions with actuals.")
    else:
        print("No matches found in latest logs yet.")

if __name__ == "__main__":
    update_actuals()
