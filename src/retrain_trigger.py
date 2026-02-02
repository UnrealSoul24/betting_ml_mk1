"""
Retrain Trigger - Intelligent injury detection and retraining orchestration.
Enhances the system with star player identification, impact radius calculation,
priority queuing, and historical accuracy tracking.
"""

import json
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
import asyncio

# Adjust path to include project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nba_api.stats.static import players
from src.injury_service import load_injury_cache, get_injury_report, compare_injury_states, save_injury_cache
from src.feature_engineer import FeatureEngineer
from src.calibration_tracker import CalibrationTracker

# --- Configuration & Constants ---
DATA_DIR = Path(__file__).parent / "../data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
RETRAINING_DIR = CACHE_DIR / "retraining"

# Thresholds
MIN_IMPACT_SCORE = 2.0  # Minimum usage shift score to trigger teammate retraining
ACCURACY_DEGRADATION_THRESHOLD = 0.20  # 20% MAE increase triggers retraining
MIN_POST_INJURY_SAMPLES = 5  # Minimum predictions before accuracy check
STAR_PLAYER_TOP_N = 3  # Top N players per team considered "stars"

# Priority Scores
PRIORITY_CRITICAL = 100 # Star OUT
PRIORITY_HIGH = 70      # Star DOUBTFUL or Rotation OUT
PRIORITY_MEDIUM = 40    # Rotation DOUBTFUL or Star QUESTIONABLE
PRIORITY_LOW = 10       # Bench or QUESTIONABLE

# Batch Limits
MAX_BATCH_SIZE = 50
MAX_PLAYERS_PER_TEAM = 10

# Log Files
INJURY_EVENT_LOG = RETRAINING_DIR / "injury_events.csv"
RETRAINING_HISTORY = RETRAINING_DIR / "retraining_history.csv"
ACCURACY_TRACKING = RETRAINING_DIR / "post_injury_accuracy.csv"

# --- Infrastructure ---

def _ensure_log_files_exist():
    """Initializes CSV logs with headers if they don't exist."""
    RETRAINING_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INJURY_EVENT_LOG.exists():
        pd.DataFrame(columns=[
            'timestamp', 'player_id', 'player_name', 'injury_status', 
            'severity_weight', 'retraining_triggered', 'priority_tier', 'affected_teammates_count'
        ]).to_csv(INJURY_EVENT_LOG, index=False)
        
    if not RETRAINING_HISTORY.exists():
        pd.DataFrame(columns=[
            'timestamp', 'player_id', 'priority', 'tier', 'reason', 'batch_id', 'status'
        ]).to_csv(RETRAINING_HISTORY, index=False)

    if not ACCURACY_TRACKING.exists():
        pd.DataFrame(columns=[
            'timestamp', 'player_id', 'post_injury_mae', 'pre_injury_mae', 'degradation_pct', 'trigger_retrain'
        ]).to_csv(ACCURACY_TRACKING, index=False)

# --- Core Logic ---

def identify_star_players(season='2025-26') -> Dict[int, List[int]]:
    """
    Identifies star players for each team based on usage metrics.
    Returns: {team_id: [star_player_ids]}
    """
    cache_file = RETRAINING_DIR / "star_players_cache.json"
    
    # Check cache (24h TTL)
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                mtime = os.path.getmtime(cache_file)
                if (datetime.now().timestamp() - mtime) < 86400: # 24 hours
                    # keys in json are strings, convert to int
                    return {int(k): v for k, v in data.items()}
        except Exception as e:
            print(f"Error loading star cache: {e}")

    print("Identifying star players from recent logs...")
    game_logs = sorted(RAW_DIR.glob(f'game_logs_{season}*.csv'), reverse=True)
    if not game_logs:
        return {}
        
    df = pd.read_csv(game_logs[0])
    
    # Calculate usage metrics per player
    # Simple composite: PTS (50%) + MIN (50%) proxy for Usage if USG_PCT not avail
    # Group by Team, then Player
    team_stars = {}
    
    for team_id in df['TEAM_ID'].unique():
        team_df = df[df['TEAM_ID'] == team_id]
        player_stats = team_df.groupby('PLAYER_ID').agg({
            'PTS': 'mean',
            'MIN': 'mean', 
            'PLAYER_NAME': 'first'
        }).reset_index()
        
        # Filter garbage time checks
        active_players = player_stats[player_stats['MIN'] > 20].copy()
        
        if active_players.empty:
            continue
            
        # Score: PTS * 1.0 + MIN * 0.5
        active_players['score'] = active_players['PTS'] + (active_players['MIN'] * 0.5)
        
        # Top N
        top_players = active_players.nlargest(STAR_PLAYER_TOP_N, 'score')
        team_stars[int(team_id)] = top_players['PLAYER_ID'].tolist()
        
    # Save cache
    try:
        with open(cache_file, 'w') as f:
            json.dump(team_stars, f)
    except Exception:
        pass
        
    return team_stars

def calculate_impact_radius(injured_pid: int, team_id: int, severity_weight: float) -> List[Dict]:
    """
    Identifies teammates significantly affected by the injured player's absence.
    Returns list of dicts: {'player_id': int, 'impact_score': float}
    """
    # Use FeatureEngineer to check for split data availability
    # In a real run, we'd query the 'with_without' cache directly.
    # For now, we simulate strictly based on rotation logic if we can't easily access the heavy FE cache here
    # or instantiate FE if cheap. FE is heavy. 
    
    # Let's try to infer rotation from game logs.
    season = '2025-26'
    game_logs = sorted(RAW_DIR.glob(f'game_logs_{season}*.csv'), reverse=True)
    if not game_logs:
        return []
        
    df = pd.read_csv(game_logs[0])
    
    # Get Teammates
    team_df = df[df['TEAM_ID'] == team_id]
    
    # Calculate simple correlation proxy? 
    # Or just return top usage players who are NOT the injured player.
    # If a Star is out, high usage players absorb volume.
    
    teammates = team_df[team_df['PLAYER_ID'] != injured_pid].groupby('PLAYER_ID').agg({
        'MIN': 'mean'
    }).reset_index()
    
    affected = []
    
    for _, row in teammates.iterrows():
        # Heuristic: High minutes players are affected most by rotation tightening/loosening
        if row['MIN'] > 15:
            # Impact Score = Base Importance (Min/10) * Severity
            score = (row['MIN'] / 10.0) * severity_weight
            if score > MIN_IMPACT_SCORE:
                affected.append({
                    'player_id': int(row['PLAYER_ID']),
                    'impact_score': score
                })
                
    # Sort by impact
    affected.sort(key=lambda x: x['impact_score'], reverse=True)
    return affected

def calculate_retraining_priority(player_id: int, status: str, team_stars: Dict[int, List[int]], team_id: int) -> Dict:
    """
    Calculates priority score and tier for a player based on injury status.
    """
    is_star = False
    if team_id in team_stars and player_id in team_stars[team_id]:
        is_star = True
        
    status = status.lower()
    
    # Severity Weight
    if "out" in status:
        severity = 1.0
        base_tier = PRIORITY_CRITICAL if is_star else PRIORITY_HIGH
        tier_name = "CRITICAL" if is_star else "HIGH"
    elif "doubtful" in status:
        severity = 0.7
        base_tier = PRIORITY_HIGH if is_star else PRIORITY_MEDIUM
        tier_name = "HIGH" if is_star else "MEDIUM"
    elif "questionable" in status:
        severity = 0.3
        base_tier = PRIORITY_MEDIUM if is_star else PRIORITY_LOW
        tier_name = "MEDIUM" if is_star else "LOW"
    else:
        severity = 0.1
        base_tier = PRIORITY_LOW
        tier_name = "LOW"
        
    
    # Composite Score: Tier Base * Severity * (1 + Impact Boost?)
    # Actually just pass pure severity weight for now as requested.
    # The prompt asked for: "base_tier * severity_weight"
    
    final_score = base_tier * severity
    
    return {
        'priority_score': final_score,
        'tier': tier_name,
        'severity_weight': severity
    }

def batch_retraining_jobs(jobs: List[Dict]) -> List[Dict]:
    """
    Deduplicates and batches jobs by team.
    Input: List of {'player_id': int, 'priority': float, 'team_id': int}
    Output: List of {'team_id': int, 'players': [ids...]}
    """
    # Dedup keeping highest priority
    unique_jobs = {}
    for job in jobs:
        pid = job['player_id']
        if pid not in unique_jobs or job['priority'] > unique_jobs[pid]['priority']:
            unique_jobs[pid] = job
            
    # Group by Team
    by_team = {}
    for pid, job in unique_jobs.items():
        tid = job.get('team_id', 0)
        if tid not in by_team:
            by_team[tid] = []
        by_team[tid].append(job)
        
    batches = []
    
    curr_batch = []
    curr_count = 0
    
    # Simple strategy: Flatten team batches
    for tid, team_jobs in by_team.items():
        # Sort by priority
        team_jobs.sort(key=lambda x: x['priority'], reverse=True)
        # Limit per team
        team_jobs = team_jobs[:MAX_PLAYERS_PER_TEAM]
        
        player_ids = [j['player_id'] for j in team_jobs]
        
        # Add to global batching logic (could make this more complex)
        # For now, just return per-team batches or aggregated?
        # The prompt says: "Return batched structure: [{'team_id': int, 'players': [{'id': int, 'priority': float}]}]"
        
        batch_entry = {
            'team_id': tid,
            'players': [{'id': j['player_id'], 'priority': j['priority']} for j in team_jobs]
        }
        batches.append(batch_entry)
        
    return batches

def detect_injury_changes() -> Dict:
    """
    Main orchestrator: Detects changes, calculates impacts, returns jobs.
    """
    _ensure_log_files_exist()
    
    # Load Cache & Report
    old_cache = load_injury_cache()
    new_report = get_injury_report(force_refresh=True)
    changes = compare_injury_states(old_cache, new_report)
    
    if not changes['all_affected']:
        return {'changes': changes, 'retraining_jobs': [], 'stats': {}}
        
    # Helpers
    from src.retrain_trigger import get_player_id_by_name, get_player_team # Self import to use existing helpers if needed, or redefine
    # Actually I should redefine helpers here or use the ones I defined below
    
    stars_map = identify_star_players()
    jobs = []
    
    stats = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'total_events': 0}
    
    # Process
    for pname in changes['all_affected']:
        pid = get_player_id_by_name(pname)
        if not pid: continue
        
        tid = get_player_team(pid)
        if not tid: continue
        
        status = new_report.get(pname, "Unknown")
        
        # Priority
        prio_info = calculate_retraining_priority(pid, status, stars_map, tid)
        
        # Log Event
        # Log Event (Placeholder count, updated later if impact found)
        # We need to know impact count BEFORE logging or update it.
        # Let's calculate impact first if needed.
        
        impact_count = 0
        teammates = []
        
        if prio_info['priority_score'] >= (PRIORITY_MEDIUM * 0.3): # Check adjusted threshold
             teammates = calculate_impact_radius(pid, tid, prio_info['severity_weight'])
             impact_count = len(teammates)
        
        # Boost priority if high impact?
        # "include it in the returned priority_score"
        if impact_count > 0:
            prio_info['priority_score'] *= (1.0 + (impact_count * 0.05))
            
        with open(INJURY_EVENT_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now()},{pid},{pname},{status},{prio_info['severity_weight']},True,{prio_info['tier']},{impact_count}\n")
            
        stats['total_events'] += 1
        stats[prio_info['tier'].lower()] += 1
        
        # Add Self
        jobs.append({
            'player_id': pid, 
            'priority': prio_info['priority_score'], 
            'tier': prio_info['tier'],
            'team_id': tid,
            'reason': f"Injury Change: {status}"
        })
        
        
        # Impact Radius (Teammates) already calculated above
        if teammates:
            
            for tm in teammates:
                jobs.append({
                    'player_id': tm['player_id'],
                    'priority': prio_info['priority_score'] * 0.5, # Lower priority for secondary effects
                    'tier': 'SECONDARY',
                    'team_id': tid,
                    'reason': f"Teammate Impact: {pname} ({status})"
                })
                
    # Batching
    batched = batch_retraining_jobs(jobs)
    
    # Return flat list for compatible API or structured?
    # The queue expects simple list of IDs usually.
    # But let's return rich structure for analysis.
    
    # Flat list of all unique IDs for the queue
    all_ids = set()
    for b in batched:
        for p in b['players']:
            all_ids.add(p['id'])
            
    return {
        'changes': changes,
        'affected_player_ids': list(all_ids),
        'batched_jobs': batched,
        'stats': stats
    }

async def queue_injury_retraining(affected_ids: List[int]):
    """
    Submits jobs to the retraining queue.
    """
    from src.retraining_queue import get_queue
    queue = get_queue()
    await queue.queue_retraining(affected_ids, use_v2=True)
    
    # Log to History
    with open(RETRAINING_HISTORY, 'a') as f:
        ts = datetime.now()
        for pid in affected_ids:
            # We don't have per-pid metadata here easily unless specific passed.
            # Just log simple
            f.write(f"{ts},{pid},0,AUTO,Injury Trigger,0,QUEUED\n")

# --- Helpers from original file (kept for compatibility) ---

def get_player_id_by_name(player_name: str) -> Optional[int]:
    all_players = players.get_players()
    pk = player_name.lower().strip()
    for p in all_players:
        if p['full_name'].lower().strip() == pk:
            return p['id']
    # fuzzy
    for p in all_players:
        if pk in p['full_name'].lower():
            return p['id']
    return None

def get_player_team(player_id: int) -> Optional[int]:
    game_logs = sorted(RAW_DIR.glob('game_logs_*.csv'), reverse=True)
    if not game_logs: return None
    df = pd.read_csv(game_logs[0])
    rows = df[df['PLAYER_ID'] == player_id]
    if rows.empty: return None
    return int(rows.iloc[-1]['TEAM_ID'])

# --- Accuracy Tracking ---

def track_post_injury_accuracy(player_ids: List[int]=None) -> List[int]:
    """
    Analyzes prediction accuracy for players recently marked as injured.
    Triggers 'ACCURACY_CORRECTION' retraining if MAE degrades significantly.
    """
    _ensure_log_files_exist()
    tracker = CalibrationTracker()
    try:
        history = tracker.load_history(days=45) # Load enough for pre/post comparison
    except Exception:
        return []
    
    if history.empty: return []
    
    # Identify relevant players (those with injuries in last 14 days)
    try:
        events = pd.read_csv(INJURY_EVENT_LOG)
        recent_injuries = events[
            pd.to_datetime(events['timestamp']) > (datetime.now() - timedelta(days=14))
        ]['player_id'].unique()
    except Exception:
        events = pd.DataFrame(columns=['player_id', 'timestamp'])
        recent_injuries = []
    
    targets = player_ids if player_ids else recent_injuries
    triggers = []
    
    for pid in targets:
        # Find first injury event for this player in recent logs
        p_events = events[events['player_id'] == pid]
        if p_events.empty: continue
        
        first_injury_date = pd.to_datetime(p_events.iloc[0]['timestamp'])
        
        # Ensure history dates are datetime
        history['game_date'] = pd.to_datetime(history['game_date'])
        
        pre_mask = (history['player_id'] == pid) & \
                   (history['game_date'] < first_injury_date) & \
                   (history['game_date'] >= (first_injury_date - timedelta(days=30)))
                   
        post_mask = (history['player_id'] == pid) & \
                    (history['game_date'] >= first_injury_date)
                    
        pre_data = history[pre_mask]
        post_data = history[post_mask]
        
        if len(post_data) < MIN_POST_INJURY_SAMPLES:
            continue
            
        # Comparison Metric: MAE of PTS (Primary)
        # We need actuals.
        pre_data = pre_data.dropna(subset=['actual'])
        post_data = post_data.dropna(subset=['actual'])
        
        if pre_data.empty or post_data.empty: continue
        
        pre_mae = (pre_data['predicted'] - pre_data['actual']).abs().mean()
        post_mae = (post_data['predicted'] - post_data['actual']).abs().mean()
        
        degradation = (post_mae - pre_mae) / pre_mae if pre_mae > 0 else 0.0
        
        trigger = False
        if degradation > ACCURACY_DEGRADATION_THRESHOLD:
            trigger = True
            triggers.append(int(pid))
            
        # Save to Accuracy Log
        with open(ACCURACY_TRACKING, 'a') as f:
            f.write(f"{datetime.now()},{pid},{post_mae:.2f},{pre_mae:.2f},{degradation:.2f},{trigger}\n")
            
    return triggers

def check_accuracy_triggers() -> List[int]:
    """Scans for accuracy triggers and returns list."""
    return track_post_injury_accuracy()

# --- Reporting ---

def generate_injury_impact_report():
    _ensure_log_files_exist()
    try:
        df = pd.read_csv(INJURY_EVENT_LOG)
        recent = df.tail(50).to_dict('records')
        stats = df['priority_tier'].value_counts().to_dict()
        return {'recent_events': recent, 'stats': stats}
    except Exception:
        return {}

# --- Test ---
if __name__ == "__main__":
    print("Running Retrain Trigger Test...")
    _ensure_log_files_exist()
    res = detect_injury_changes()
    print(json.dumps(res, indent=2, default=str))
