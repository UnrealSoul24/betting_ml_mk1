"""
Advanced Stats Service - Fetches advanced tracking data from NBA API.

Key Features:
- Potential Assists: Passes that could have been assists
- Rebound Chances: Opportunities to grab rebounds  
- Shot Quality: Open vs Contested shot percentages

Endpoints Used:
- playerdashptpass: Passing stats with potential assists
- playerdashptreb: Rebounding chances
- playerdashptshots: Shot charts by defense proximity
"""

import os
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

try:
    from nba_api.stats.endpoints import playerdashptpass, playerdashptreb, playerdashptshotdefend
    from nba_api.stats.static import players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("[AdvancedStats] Warning: nba_api not available")

CACHE_DIR = Path(__file__).parent / "../data/cache/advanced_stats"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_current_season():
    """Returns current NBA season in 'YYYY-YY' format."""
    now = datetime.now()
    if now.month >= 10:
        return f"{now.year}-{str(now.year + 1)[-2:]}"
    else:
        return f"{now.year - 1}-{str(now.year)[-2:]}"

def _get_cache_path(player_id: int, stat_type: str):
    """Returns cache path for player's advanced stats."""
    season = _get_current_season()
    return CACHE_DIR / f"{player_id}_{stat_type}_{season}.json"

def _load_cached(player_id: int, stat_type: str, max_age_hours: int = 24):
    """Loads cached data if fresh enough."""
    cache_path = _get_cache_path(player_id, stat_type)
    if cache_path.exists():
        mtime = cache_path.stat().st_mtime
        age_hours = (time.time() - mtime) / 3600
        if age_hours < max_age_hours:
            with open(cache_path, 'r') as f:
                return json.load(f)
    return None

def _save_cache(player_id: int, stat_type: str, data: dict):
    """Saves data to cache."""
    cache_path = _get_cache_path(player_id, stat_type)
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def get_potential_assists(player_id: int, season: str = None) -> dict:
    """
    Fetches potential assists data for a player.
    
    Potential Assists = Passes that led to shots (made or missed).
    High Potential AST with low Actual AST = teammates not converting.
    
    Returns:
        dict with:
        - potential_assists: float
        - assist_adj_passes: float  
        - passes_made: int
        - found: bool
    """
    if not NBA_API_AVAILABLE:
        return {'found': False, 'reason': 'NBA API not available'}
    
    season = season or _get_current_season()
    
    # Check cache
    cached = _load_cached(player_id, 'passing')
    if cached:
        return cached
    
    try:
        time.sleep(0.6)  # Rate limit
        data = playerdashptpass.PlayerDashPtPass(
            player_id=player_id,
            season=season,
            per_mode_simple='PerGame'
        )
        
        df = data.get_data_frames()[0]  # Passing stats
        
        if df.empty:
            return {'found': False, 'reason': 'No data'}
        
        result = {
            'found': True,
            'potential_assists': float(df['POTENTIAL_AST'].sum()) if 'POTENTIAL_AST' in df.columns else 0,
            'passes_made': int(df['PASS'].sum()) if 'PASS' in df.columns else 0,
            'ast_adj_passes': float(df['AST_ADJ'].sum()) if 'AST_ADJ' in df.columns else 0,
        }
        
        _save_cache(player_id, 'passing', result)
        return result
        
    except Exception as e:
        return {'found': False, 'reason': str(e)}

def get_rebound_chances(player_id: int, season: str = None) -> dict:
    """
    Fetches rebounding chances data for a player.
    
    Rebound Chances = Loose balls where player could have grabbed rebound.
    High Chances with low actual REB = poor positioning or boxing out issues.
    
    Returns:
        dict with:
        - oreb_chances: float
        - dreb_chances: float
        - reb_chances: float
        - found: bool
    """
    if not NBA_API_AVAILABLE:
        return {'found': False, 'reason': 'NBA API not available'}
    
    season = season or _get_current_season()
    
    cached = _load_cached(player_id, 'rebounding')
    if cached:
        return cached
    
    try:
        time.sleep(0.6)
        data = playerdashptreb.PlayerDashPtReb(
            player_id=player_id,
            season=season,
            per_mode_simple='PerGame'
        )
        
        df = data.get_data_frames()[0]
        
        if df.empty:
            return {'found': False, 'reason': 'No data'}
        
        result = {
            'found': True,
            'oreb_chances': float(df['OREB_CHANCES'].iloc[0]) if 'OREB_CHANCES' in df.columns else 0,
            'dreb_chances': float(df['DREB_CHANCES'].iloc[0]) if 'DREB_CHANCES' in df.columns else 0,
            'reb_chances': float(df['REB_CHANCES'].iloc[0]) if 'REB_CHANCES' in df.columns else 0,
        }
        
        _save_cache(player_id, 'rebounding', result)
        return result
        
    except Exception as e:
        return {'found': False, 'reason': str(e)}

def get_shot_quality(player_id: int, season: str = None) -> dict:
    """
    Fetches shot quality data based on defender proximity.
    
    Returns:
        dict with:
        - open_shot_pct: % of shots taken with defender 6+ feet away
        - tight_shot_pct: % of shots with defender 0-2 feet
        - open_fg_pct: FG% on open shots
        - found: bool
    """
    if not NBA_API_AVAILABLE:
        return {'found': False, 'reason': 'NBA API not available'}
    
    season = season or _get_current_season()
    
    cached = _load_cached(player_id, 'shot_quality')
    if cached:
        return cached
    
    try:
        time.sleep(0.6)
        data = playerdashptshotdefend.PlayerDashPtShotDefend(
            player_id=player_id,
            season=season,
            per_mode_simple='PerGame'
        )
        
        df = data.get_data_frames()[0]
        
        if df.empty:
            return {'found': False, 'reason': 'No data'}
        
        total_fga = df['FGA'].sum() if 'FGA' in df.columns else 1
        
        # Group by defender distance
        open_fga = df[df['CLOSE_DEF_DIST'].isin(['6+ Feet - Wide Open', '4-6 Feet - Open'])]['FGA'].sum() if 'CLOSE_DEF_DIST' in df.columns else 0
        tight_fga = df[df['CLOSE_DEF_DIST'].isin(['0-2 Feet - Very Tight', '2-4 Feet - Tight'])]['FGA'].sum() if 'CLOSE_DEF_DIST' in df.columns else 0
        
        result = {
            'found': True,
            'open_shot_pct': float(open_fga / max(total_fga, 1)),
            'tight_shot_pct': float(tight_fga / max(total_fga, 1)),
            'total_fga': int(total_fga),
        }
        
        _save_cache(player_id, 'shot_quality', result)
        return result
        
    except Exception as e:
        return {'found': False, 'reason': str(e)}

def get_all_advanced_stats(player_id: int) -> dict:
    """
    Fetches all advanced stats for a player.
    Combines potential assists, rebound chances, and shot quality.
    """
    passing = get_potential_assists(player_id)
    rebounding = get_rebound_chances(player_id)
    shots = get_shot_quality(player_id)
    
    return {
        'player_id': player_id,
        'passing': passing,
        'rebounding': rebounding,
        'shot_quality': shots,
        'available': any([passing.get('found'), rebounding.get('found'), shots.get('found')])
    }


# Quick test
if __name__ == "__main__":
    # Test with LeBron James (player_id: 2544)
    print("Testing Advanced Stats Service...")
    
    stats = get_all_advanced_stats(2544)
    print(f"\nLeBron Advanced Stats:")
    print(f"Passing: {stats['passing']}")
    print(f"Rebounding: {stats['rebounding']}")
    print(f"Shot Quality: {stats['shot_quality']}")
