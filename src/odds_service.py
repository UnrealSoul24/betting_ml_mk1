"""
Odds Service - Fetches and processes player prop odds from SportsGameOdds API.

Key Functions:
- get_todays_odds(): Fetches all NBA odds for today
- get_player_prop_odds(player_name, stat_type): Returns (line, odds, ev) for a player
- american_to_implied_prob(odds): Converts American odds to probability
- calculate_ev(model_prob, american_odds): Calculates Expected Value

API Key: Set SPORTGAMEODDSAPIKEY in .env file
"""

import os
import re
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_BASE = "https://api.sportsgameodds.com/v2"
API_KEY = os.getenv("SPORTGAMEODDSAPIKEY")
CACHE_DIR = Path(__file__).parent / "../data/cache"
CACHE_DIR.mkdir(exist_ok=True)

# Stat type mapping
STAT_MAP = {
    'PTS': 'points',
    'REB': 'rebounds', 
    'AST': 'assists',
    'STL': 'steals',
    'BLK': 'blocks',
    '3PM': 'threesMade',
    'RA': 'rebounds-assists',
    'PR': 'points-rebounds',
    'PA': 'points-assists',
    'PRA': 'points-rebounds-assists'
}

def _get_cache_path():
    """Returns cache file path for today's odds."""
    today = datetime.now().strftime("%Y-%m-%d")
    return CACHE_DIR / f"nba_odds_{today}.json"

def _load_from_cache():
    """Loads odds from today's cache if available."""
    cache_path = _get_cache_path()
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def _save_to_cache(data):
    """Saves odds data to cache."""
    cache_path = _get_cache_path()
    with open(cache_path, 'w') as f:
        json.dump(data, f)

def _player_name_to_api_id(player_name: str) -> str:
    """
    Converts player name like "Joel Embiid" to API format "JOEL_EMBIID_1_NBA".
    Note: The "_1" suffix is a consistent player ID in this API.
    """
    # Remove special characters, convert to uppercase, replace spaces with underscores
    clean_name = re.sub(r"[^a-zA-Z\s]", "", player_name)
    parts = clean_name.strip().upper().split()
    api_id = "_".join(parts) + "_1_NBA"
    return api_id

def american_to_decimal(american_odds: str) -> float:
    """Converts American odds string to decimal odds."""
    try:
        odds = int(american_odds.replace('+', ''))
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    except:
        return 2.0  # Default to even odds

def american_to_implied_prob(american_odds: str) -> float:
    """Converts American odds to implied probability."""
    try:
        odds = int(american_odds.replace('+', ''))
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    except:
        return 0.5

def calculate_ev(model_prob: float, american_odds: str) -> float:
    """
    Calculates Expected Value.
    EV = (P_win * Decimal_Payout) - 1
    Returns as a percentage (e.g., 0.05 = 5% edge).
    """
    decimal_odds = american_to_decimal(american_odds)
    ev = (model_prob * decimal_odds) - 1
    return ev

def fetch_nba_odds(force_refresh=False):
    """
    Fetches all NBA odds from API.
    Uses cache if available and not forcing refresh.
    """
    if not force_refresh:
        cached = _load_from_cache()
        if cached:
            print(f"[OddsService] Loaded odds from cache ({len(cached.get('data', []))} events)")
            return cached
    
    if not API_KEY:
        print("[OddsService] Warning: SPORTGAMEODDSAPIKEY not set in .env")
        return None
    
    url = f"{API_BASE}/events"
    params = {
        "leagueID": "NBA",
        "oddsAvailable": "true"
    }
    headers = {
        "x-api-key": API_KEY
    }
    
    try:
        print("[OddsService] Fetching live odds from API...")
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        _save_to_cache(data)
        print(f"[OddsService] Fetched {len(data.get('data', []))} events")
        return data
    except requests.RequestException as e:
        print(f"[OddsService] API Error: {e}")
        return None

def get_game_total(event: dict) -> float:
    """
    Extracts the game total O/U from an event object.
    Returns the total (e.g., 221.5) or None if not found.
    """
    odds = event.get('odds', {})
    
    # Look for the game total O/U
    over_key = 'points-all-game-ou-over'
    if over_key in odds:
        return float(odds[over_key].get('bookOverUnder', 0))
    
    return None

def get_game_total_for_matchup(team_abbrev: str, odds_data=None) -> dict:
    """
    Finds the game total for a team's matchup.
    
    Args:
        team_abbrev: Team abbreviation (e.g., 'PHI', 'LAL')
        odds_data: Optional pre-fetched odds data
    
    Returns:
        dict with:
        - found: bool
        - total: float (e.g., 221.5)
        - home_team: str
        - away_team: str
    """
    if odds_data is None:
        odds_data = fetch_nba_odds()
    
    if not odds_data or 'data' not in odds_data:
        return {'found': False}
    
    team_abbrev_upper = team_abbrev.upper()
    
    for event in odds_data.get('data', []):
        teams = event.get('teams', {})
        home = teams.get('home', {}).get('names', {}).get('short', '')
        away = teams.get('away', {}).get('names', {}).get('short', '')
        
        if team_abbrev_upper in [home.upper(), away.upper()]:
            total = get_game_total(event)
            if total:
                return {
                    'found': True,
                    'total': total,
                    'home_team': home,
                    'away_team': away,
                    'event_id': event.get('eventID')
                }
    
    return {'found': False, 'reason': f'No game found for {team_abbrev}'}

def get_player_prop_odds(player_name: str, stat_type: str = 'PTS', odds_data=None):
    """
    Retrieves player prop odds from the cached/fetched data.
    
    Args:
        player_name: e.g., "Joel Embiid"
        stat_type: One of 'PTS', 'REB', 'AST', 'STL', 'BLK', '3PM'
        odds_data: Optional pre-fetched odds data
    
    Returns:
        dict with keys:
        - line: Over/Under line (e.g., 25.5)
        - odds_over: American odds for over (e.g., "-110")
        - odds_under: American odds for under
        - implied_prob_over: Implied probability of over
        - best_book_over: Best bookmaker for over bet
        - found: bool if odds were found
    """
    if odds_data is None:
        odds_data = fetch_nba_odds()
    
    if not odds_data or 'data' not in odds_data:
        return {'found': False, 'reason': 'No odds data available'}
    
    # Convert player name to API format
    player_api_id = _player_name_to_api_id(player_name)
    stat_api_id = STAT_MAP.get(stat_type, 'points')
    
    # Search for the player's prop in all events
    over_key = f"{stat_api_id}-{player_api_id}-game-ou-over"
    under_key = f"{stat_api_id}-{player_api_id}-game-ou-under"
    
    for event in odds_data.get('data', []):
        odds = event.get('odds', {})
        
        if over_key in odds:
            over_data = odds[over_key]
            under_data = odds.get(under_key, {})
            
            # Find best bookmaker odds
            best_book = None
            best_odds = None
            for book, book_data in over_data.get('byBookmaker', {}).items():
                if book_data.get('available'):
                    if best_odds is None or int(book_data['odds'].replace('+', '')) > int((best_odds or '-110').replace('+', '')):
                        best_odds = book_data['odds']
                        best_book = book
            
            return {
                'found': True,
                'line': float(over_data.get('bookOverUnder', 0)),
                'odds_over': over_data.get('bookOdds', '-110'),
                'odds_under': under_data.get('bookOdds', '-110'),
                'implied_prob_over': american_to_implied_prob(over_data.get('bookOdds', '-110')),
                'best_book_over': best_book,
                'best_odds_over': best_odds,
                'player_api_id': player_api_id,
                'stat': stat_type
            }
    
    return {
        'found': False,
        'reason': f'No odds found for {player_name} ({player_api_id}) {stat_type}',
        'player_api_id': player_api_id
    }

def normal_cdf(x, mu=0, sigma=1):
    """Cumulative Distribution Function for Normal Distribution."""
    import math
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def get_ev_for_prediction(player_name: str, stat_type: str, prediction: float, std_dev: float = None, model_confidence: float = None, odds_data=None):
    """
    Calculates Expected Value for a prediction vs market odds.
    
    Args:
        player_name: e.g., "Joel Embiid"
        stat_type: 'PTS', 'REB', ...
        prediction: Model's predicted value (e.g., 28.5 points)
        std_dev: Model's uncertainty (Standard Deviation). Preferred over model_confidence.
        model_confidence: Deprecated/Fallback (0-1).
        odds_data: Optional pre-fetched odds
    
    Returns:
        dict with EV calculation details
    """
    prop = get_player_prop_odds(player_name, stat_type, odds_data)
    
    if not prop.get('found'):
        return {'found': False, 'reason': prop.get('reason')}
    
    line = prop['line']
    
    # Calculate Probability of Over
    if std_dev and std_dev > 0:
        # Gaussian method (Robust)
        # P(Over) = 1 - CDF(line)
        # We use line implies > line. 
        # But strictly Over 25.5 means 26+. Continuous CDF at 25.5 is correct.
        p_over = 1 - normal_cdf(line, mu=prediction, sigma=std_dev)
    else:
        # Fallback Sigmoid Logic (if only model_confidence provided)
        # Assuming std_dev proxy of ~15% of prediction if missing
        sigma_proxy = prediction * 0.15 if prediction > 0 else 1.0
        p_over = 1 - normal_cdf(line, mu=prediction, sigma=sigma_proxy)
        
        # Adjust using the old confidence scalar if provided
        if model_confidence:
             p_over = 0.5 + (p_over - 0.5) * model_confidence
    
    # Calculate EV
    ev_over = calculate_ev(p_over, prop['odds_over'])
    p_under = 1 - p_over
    ev_under = calculate_ev(p_under, prop['odds_under'])
    
    return {
        'found': True,
        'player': player_name,
        'stat': stat_type,
        'prediction': prediction,
        'market_line': line,
        'std_dev': std_dev,
        'p_over': p_over,
        'p_under': p_under,
        'ev_over': ev_over,
        'ev_under': ev_under,
        'odds_over': prop['odds_over'],
        'odds_under': prop['odds_under'],
        'best_book': prop.get('best_book_over'),
        'recommendation': 'OVER' if ev_over > ev_under and ev_over > 0 else ('UNDER' if ev_under > 0 else 'PASS')
    }

# Quick test
if __name__ == "__main__":
    # Test with local cache
    print("Testing Odds Service...")
    
    # Attempt to get Joel Embiid's points line
    result = get_player_prop_odds("Joel Embiid", "PTS")
    print(f"\nJoel Embiid PTS: {result}")
    
    # Test EV calculation
    if result.get('found'):
        ev = get_ev_for_prediction("Joel Embiid", "PTS", 28.0, model_confidence=0.65)
        print(f"\nEV Analysis: {ev}")
