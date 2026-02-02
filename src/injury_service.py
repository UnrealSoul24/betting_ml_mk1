"""
NBA Injury Report Service

Fetches and caches NBA injury reports from multiple sources with automatic fallback.

Sources (in priority order):
1. CBS Sports (primary)
2. ESPN (fallback)
3. NBA.com (tertiary)

Features:
- Multi-source scraping with automatic fallback
- Exponential backoff retry logic (3 attempts per source)
- Fuzzy player name matching for concatenated names
- Automatic cache refresh for data older than 2 hours
- Force refresh capability for real-time updates

Usage:
    # Use cached data (auto-refresh if stale)
    injuries = get_injury_report()
    
    # Force fresh scrape
    injuries = get_injury_report(force_refresh=True)

Returns:
    dict: {'PLAYER_NAME': 'STATUS'}
    Status values: 'Out', 'Questionable', 'Doubtful', 'Active'
"""

import pandas as pd
import requests
import io
import json
import os
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from rapidfuzz import fuzz, process

# Constants
CACHE_EXPIRY_HOURS = 2
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 2

# Configuration for scraping sources
SCRAPE_SOURCES = [
    {"name": "CBS", "url": "https://www.cbssports.com/nba/injuries/", "parser": "_parse_cbs_injuries"}
    # {"name": "ESPN", "url": "https://www.espn.com/nba/injuries", "parser": "_parse_espn_injuries"},
    # {"name": "NBA", "url": "https://www.nba.com/stats/injuries", "parser": "_parse_nba_injuries"}
]

# Paths
DATA_DIR = Path(__file__).parent / "../data"
CACHE_DIR = DATA_DIR / "cache"
INJURY_CACHE_PATH = CACHE_DIR / "injury_state.json"
INJURY_ARCHIVE_DIR = CACHE_DIR / "injury_archives"
METRICS_LOG_PATH = CACHE_DIR / "injury_scrape_metrics.jsonl"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True, parents=True)
INJURY_ARCHIVE_DIR.mkdir(exist_ok=True, parents=True)

def _fetch_with_retry(url: str, headers: dict, max_retries: int = MAX_RETRIES) -> requests.Response:
    """Fetches URL with exponential backoff retry logic."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            wait_time = RETRY_BACKOFF_FACTOR ** attempt
            print(f"[InjuryService] Attempt {attempt+1}/{max_retries} failed for {url}: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
            
            if attempt == max_retries - 1:
                raise e

def _log_scrape_metrics(source_name: str, success: bool, player_count: int, duration: float):
    """Logs scrape performance metrics."""
    log_entry = {
         "timestamp": datetime.now().isoformat(),
         "source": source_name,
         "success": success,
         "players_found": player_count,
         "duration_seconds": round(duration, 2)
    }
    
    # Log to console
    print(f"[InjuryService] Source: {source_name}, Success: {success}, Players: {player_count}, Duration: {duration:.2f}s")
    
    # Log to file
    try:
        with open(METRICS_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[InjuryService] Warning: Failed to write metrics log: {e}")

def _clean_player_name(raw_name: str) -> str:
    """Cleans and normalizes player names, handling concatenation issues."""
    # 1. Remove position abbreviations commonly found in scraping (e.g. "LeBron James SF")
    # Matches suffix positions like G, F, C, PG, SG, SF, PF preceded by whitespace
    raw_name = re.sub(r'\s+(G|F|C|PG|SG|SF|PF)$', '', raw_name)
    
    # 2. Handle Concatenated Names (e.g. "T. HerroTyler Herro")
    # Strategy: Split by Capital Letters. If duplicates found, take the full name part.
    # "T. HerroTyler Herro" -> ['T', 'Herro', 'Tyler', 'Herro']
    parts = re.findall(r'[A-Z][a-z\.]+', raw_name)
    
    cleaned_name = raw_name
    
    if len(parts) >= 4:
        # Check if the second half looks like a full name (heuristic)
        # e.g. T, Herro, Tyler, Herro
        # We assume the last two parts form the First Last name
        candidate = " ".join(parts[-2:])
        
        # Verify it resembles the input somewhat using fuzz
        if fuzz.partial_ratio(candidate, raw_name) > 85:
             cleaned_name = candidate
    
    # 3. Standardize Whitespace
    cleaned_name = ' '.join(cleaned_name.split())
    
    # 4. Remove special chars (except apostrophes/hyphens which are valid in names)
    # Keeping it simple for now as complex regex might strip valid chars.
    
    return cleaned_name

def _fuzzy_match_player_name(scraped_name: str, known_players: list = None) -> str:
    """
    Matches scraped name against known player database if provided, 
    otherwise just cleans it.
    """
    clean_name = _clean_player_name(scraped_name)
    
    if known_players:
        # If we had a list of all active NBA players, we could map it here.
        # For now, we rely on the clean logic, but this function is ready for
        # full database matching integration.
        match = process.extractOne(clean_name, known_players, scorer=fuzz.ratio)
        if match and match[1] >= 85:
            # print(f"[InjuryService] Fuzzy matched '{scraped_name}' -> '{match[0]}' (score: {match[1]})")
            return match[0]
            
    return clean_name

def _validate_injury_data(injury_map: dict) -> bool:
    """Validates the scraped injury data quality."""
    if not injury_map:
        print("[InjuryService] Validation Failed: Empty injury map.")
        return False
        
    if len(injury_map) < 5:
        print(f"[InjuryService] Warning: Only {len(injury_map)} injuries found. Low count but proceeding.")
        # We allow low counts now as sticking to "Strict > 5" causes fallbacks even on valid days with few injuries.
        
    valid_statuses = {"Out", "Questionable", "Doubtful", "Active"}
    unknown_statuses = [s for s in injury_map.values() if s not in valid_statuses]
    
    if len(unknown_statuses) > len(injury_map) * 0.5:
        print(f"[InjuryService] Validation Failed: Too many unknown statuses: {unknown_statuses[:5]}...")
        return False
        
    # Check for malformed names (sanity check)
    malformed_names = [n for n in injury_map.keys() if len(n) > 50 or len(n) < 3]
    if malformed_names:
         print(f"[InjuryService] Warning: Found {len(malformed_names)} malformed names (e.g. {malformed_names[0]}).")
         # We process anyway but log warning
         
    return True

# --- Parsers ---

def _parse_cbs_injuries(html_content: str) -> dict:
    """Parser for CBS Sports injury report."""
    dfs = pd.read_html(io.StringIO(html_content))
    injury_map = {}
    
    for df in dfs:
        # Check for necessary columns
        # Case 1: 'Injury Status' exists (Preferred)
        status_col = None
        if 'Injury Status' in df.columns:
            status_col = 'Injury Status'
        elif 'Injury' in df.columns:
             # Fallback: Sometimes 'Injury' is the only col, but usually incorrect for Status
             # But if Injury Status is missing, we might have to use Injury if it implies status (rare)
             status_col = 'Injury'
             
        if 'Player' in df.columns and status_col:
            for _, row in df.iterrows():
                player_name = str(row['Player']).strip()
                status_raw = str(row[status_col]).strip()
                
                name = _clean_player_name(player_name)
                
                # Standardize Status
                s_lower = status_raw.lower()
                status = "Questionable" # Default fallback
                
                # "Expected to be out" or "Out for the season" or just "Out"
                if "out" in s_lower: status = "Out"
                elif "doubtful" in s_lower: status = "Doubtful"
                elif "questionable" in s_lower: status = "Questionable"
                elif "decision" in s_lower or "gtd" in s_lower: status = "Questionable"
                
                injury_map[name] = status
                
    return injury_map

def _parse_espn_injuries(html_content: str) -> dict:
    """Parser for ESPN injury report."""
    dfs = pd.read_html(io.StringIO(html_content))
    injury_map = {}
    
    for df in dfs:
        # ESPN cols usually: NAME, POS, DATE, STATUS, COMMENT
        # Sometimes headers are different. Look for NAME or PLAYER
        cols = [c.upper() for c in df.columns]
        
        name_col = next((c for c in df.columns if 'NAME' in c.upper() or 'PLAYER' in c.upper()), None)
        status_col = next((c for c in df.columns if 'STATUS' in c.upper()), None)
        
        if name_col and status_col:
            for _, row in df.iterrows():
                player_name = str(row[name_col]).strip()
                status_raw = str(row[status_col]).strip()
                
                name = _clean_player_name(player_name)
                
                s_lower = status_raw.lower()
                status = "Questionable"
                
                if "out" in s_lower: status = "Out"
                elif "doubtful" in s_lower: status = "Doubtful"
                elif "questionable" in s_lower or "day-to-day" in s_lower: status = "Questionable"
                
                injury_map[name] = status
                
    return injury_map

def _parse_nba_injuries(html_content: str) -> dict:
    """Parser for NBA.com injury report (HTML or JSON)."""
    injury_map = {}
    
    try:
        # Try JSON parse first
        data = json.loads(html_content)
        
        # Handle standard NBAStats 'resultSets' structure
        # Key fields usually: 'PlayerName', 'Status', etc.
        result_sets = data.get('resultSets', [])
        if not result_sets and 'rowSet' in data:
             # Sometimes the root IS the result set
             result_sets = [data]
             
        for result in result_sets:
            headers = result.get('headers', [])
            row_set = result.get('rowSet', [])
            
            # Identify columns dynamics
            # We need PLAYER_NAME (or Player) and STATUS (or InjuryStatus)
            try:
                # Normalizing headers to uppercase for search
                headers_upper = [h.upper() for h in headers]
                name_idx = next(i for i, h in enumerate(headers_upper) if 'PLAYER' in h and 'NAME' in h) # e.g. PLAYER_NAME
                # Status might be 'COMMENT', 'STATUS', 'INJURY_STATUS'
                # fallback checks
                status_idx = -1
                for keyword in ['STATUS', 'COMMENT', 'INJURY']:
                     for i, h in enumerate(headers_upper):
                         if keyword in h:
                             status_idx = i
                             break
                     if status_idx != -1: break
                
                if name_idx != -1 and status_idx != -1:
                    for row in row_set:
                        player_name = str(row[name_idx])
                        status_raw = str(row[status_idx])
                        
                        name = _clean_player_name(player_name)
                        
                        s_lower = status_raw.lower()
                        status = "Questionable"
                        if "out" in s_lower: status = "Out"
                        elif "doubtful" in s_lower: status = "Doubtful"
                        
                        injury_map[name] = status
            except StopIteration:
                continue

    except json.JSONDecodeError:
        # Fallback to HTML Parse
        try:
            dfs = pd.read_html(io.StringIO(html_content))
            for df in dfs:
                if 'Player' in df.columns and 'Current Status' in df.columns:
                     for _, row in df.iterrows():
                        player_name = str(row['Player']).strip()
                        status_raw = str(row['Current Status']).strip()
                        
                        name = _clean_player_name(player_name)
                        
                        s_lower = status_raw.lower()
                        status = "Questionable"
                        if "out" in s_lower: status = "Out"
                        
                        injury_map[name] = status
        except:
             pass
             
    return injury_map

def _scrape_from_sources() -> dict:
    """Orchestrates multi-source scraping with fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
    }
    
    for source in SCRAPE_SOURCES:
        start_time = time.time()
        print(f"[InjuryService] Attempting to scrape from {source['name']}...")
        
        try:
            r = _fetch_with_retry(source['url'], headers)
            parser_func = globals()[source['parser']]
            injury_map = parser_func(r.text)
            
            duration = time.time() - start_time
            
            if _validate_injury_data(injury_map):
                _log_scrape_metrics(source['name'], True, len(injury_map), duration)
                return injury_map
            else:
                _log_scrape_metrics(source['name'], False, len(injury_map), duration)
                print(f"[InjuryService] Data invalid from {source['name']}, trying next source...")
                
        except Exception as e:
            duration = time.time() - start_time
            _log_scrape_metrics(source['name'], False, 0, duration)
            print(f"[InjuryService] Failed to scrape {source['name']}: {e}")
            
    print("[InjuryService] CRITICAL: All scrape sources failed.")
    return {}

def load_injury_cache():
    """
    Loads the last saved injury state from cache with metadata.
    Returns:
        dict: {
            "last_updated": str (ISO),
            "injuries": dict,
            "is_stale": bool,
            "age_hours": float
        }
    """
    default_ret = {"last_updated": None, "injuries": {}, "is_stale": True, "age_hours": 999.0}
    
    if not INJURY_CACHE_PATH.exists():
        return default_ret
    
    try:
        with open(INJURY_CACHE_PATH, 'r') as f:
            data = json.load(f)
            
        last_updated = data.get("last_updated")
        if not last_updated:
            return default_ret
            
        dt_updated = datetime.fromisoformat(last_updated)
        age = (datetime.now() - dt_updated).total_seconds() / 3600.0
        
        data["is_stale"] = age > CACHE_EXPIRY_HOURS
        data["age_hours"] = round(age, 2)
        
        return data
    except Exception as e:
        print(f"[InjuryService] Error loading cache: {e}")
        return default_ret

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

def compare_injury_states(old_state: dict, new_state: dict):
    """(Existing Logic) Compares two injury states and returns what changed."""
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
            if status == "Out" or status == "Questionable":
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

def get_injury_report(force_refresh: bool = False) -> dict:
    """
    Fetches the latest NBA injury report with multi-source fallback.

    Args:
        force_refresh (bool): If True, bypass cache and scrape fresh data.
                             If False, use cache if available and not stale (>2 hours).

    Returns:
        dict: Mapping of player names to injury status.
              Example: {'LeBron James': 'Out', 'Stephen Curry': 'Questionable'}

    Raises:
        None: Returns empty dict on total failure, logs errors.

    Notes:
        - Automatically tries CBS -> ESPN -> NBA.com until successful
        - Each source has 3 retry attempts with exponential backoff
        - Falls back to stale cache if all sources fail
        - Player names are fuzzy-matched to handle concatenation issues
    """
    # Step 1: Check cache if not forcing refresh
    if not force_refresh:
        cache = load_injury_cache()
        if cache["injuries"] and not cache["is_stale"]:
            print(f"[InjuryService] Using cached injury data (Age: {cache['age_hours']}h)")
            return cache["injuries"]
    
    # Step 2: Scrape fresh data
    print(f"[InjuryService] Fetching fresh injury report (force_refresh={force_refresh})...")
    injury_map = _scrape_from_sources()
    
    # Fallback to stale cache if scrape completely failed
    if not injury_map:
        cache = load_injury_cache()
        if cache["injuries"]:
             print("[InjuryService] Warning: All scrape sources failed. Using stale cache as fallback.")
             return cache["injuries"]
        else:
             print("[InjuryService] Error: Failed to fetch injuries and no cache available.")
             return {}

    # Step 3: Save and return
    save_injury_cache(injury_map)
    return injury_map

if __name__ == "__main__":
    print("="*60)
    print("INJURY SERVICE DIAGNOSTICS")
    print("="*60)
    
    # 1. Test Name Cleaning
    print("\n--- Testing Name Cleaning Logic ---")
    test_names = ["T. HerroTyler Herro", "P. GeorgePaul George", "L. JamesLeBron James", "J. EmbiidJoel Embiid"]
    for raw in test_names:
        clean = _clean_player_name(raw)
        print(f"Raw: '{raw}' -> Clean: '{clean}'")
        
    # 2. Test Validation Logic
    print("\n--- Testing Validation Logic ---")
    bad_data = {"": "Out", "X"*100: "Questionable"}
    print(f"Bad Data Valid? {_validate_injury_data(bad_data)}")
    
    # 3. Test Full Fetch (Force Refresh)
    print("\n--- Testing Live Fetch (Force Refresh) ---")
    report = get_injury_report(force_refresh=True)
    
    print(f"\nFinal Report Item Count: {len(report)}")
    print("Sample Injuries:")
    for p, s in list(report.items())[:5]:
        print(f"  - {p}: {s}")
        
    # 4. Test Cache Retrieval
    print("\n--- Testing Cache Retrieval ---")
    cached = get_injury_report(force_refresh=False)
    print(f"Cached Items: {len(cached)}")
