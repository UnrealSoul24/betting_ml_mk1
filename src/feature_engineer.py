import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict
from src.injury_service import _fuzzy_match_player_name


# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Tuple

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
WITH_WITHOUT_CACHE_PATH = os.path.join(DATA_DIR, 'cache', 'with_without_splits.joblib')

INJURY_SEVERITY_WEIGHTS = {'Out': 1.0, 'Doubtful': 0.7, 'Questionable': 0.3, 'Active': 0.0}
MIN_SPLIT_SAMPLE_SIZE = 5
MAX_OPPORTUNITY_BOOST = 15.0

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

class FeatureEngineer:
    def __init__(self):
        self.player_encoder = LabelEncoder()
        self.team_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_all_data(self) -> pd.DataFrame:
        """Loads and concatenates all available season logs."""
        files = [f for f in os.listdir(RAW_DIR) if f.startswith('game_logs_')]
        dfs = []
        for f in files:
            path = os.path.join(RAW_DIR, f)
            print(f"Loading {path}...")
            df = pd.read_csv(path)
            # Parse Season Year from filename (e.g. game_logs_2023-24.csv -> 2024)
            try:
                # filename is just 'f'
                # game_logs_2023-24.csv
                s_str = f.split('_')[-1].replace('.csv', '') # 2023-24
                # We want 2024
                s_year = int(s_str.split('-')[0]) + 1
                df['SEASON_YEAR'] = s_year
            except:
                print(f"Failed to parse season from {f}")
                df['SEASON_YEAR'] = 2024 # Fallback
            dfs.append(df)
            
        if not dfs:
             raise FileNotFoundError("No game logs found in raw directory.")
             
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Filter Data (User Request: Just 24-25 and 25-26)
        # 2024-25 -> SEASON_YEAR 2025
        # 2025-26 -> SEASON_YEAR 2026
        full_df = full_df[full_df['SEASON_YEAR'] >= 2025].copy()
        print(f"Loaded {len(full_df)} rows (Filtered for 2025+).")
        
        return full_df

    def load_player_positions(self) -> pd.DataFrame:
        """Loads and combines player position data for all seasons."""
        dfs = []
        # Get all position files
        import glob
        files = glob.glob(os.path.join(RAW_DIR, 'player_positions_*.csv'))
        for f in files:
            df = pd.read_csv(f)
            # Add season year from filename
            season = os.path.basename(f).split('_')[-1].replace('.csv', '')
            df['SEASON_YEAR'] = int(season.split('-')[0]) + 1 # 2023-24 -> 2024
            
            # Key: Player + Season mapping
            dfs.append(df)
            
        if not dfs:
            print("Warning: No position files found.")
            return pd.DataFrame()
            
        return pd.concat(dfs)

    def calculate_dvp(self, df: pd.DataFrame, pos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Defense vs Position (DvP).
        1. Merge Player positions into Game Logs.
        2. Group by Opponent Team + Opponent Position + Date.
        3. Calculate Rolling Avg of stats allowed.
        """
        print("Calculating Defense vs. Position (DvP)...")
        
        # 1. Merge Position to Game Logs
        # We need to match Player + Season. 
        # DF has SEASON_YEAR (e.g. 2024 for 2023-24). 
        # Position DF also has SEASON_YEAR.
        # But Position DF might have duplicates if traded. Take last?
        pos_df = pos_df.drop_duplicates(subset=['PLAYER_ID', 'SEASON_YEAR'], keep='last')
        
        df = df.merge(pos_df[['PLAYER_ID', 'SEASON_YEAR', 'POSITION_SIMPLE']], 
                      on=['PLAYER_ID', 'SEASON_YEAR'], 
                      how='left')
                      
        df['POSITION_SIMPLE'] = df['POSITION_SIMPLE'].fillna('F') # Default to F if missing
        
        # 2. Aggregating: What did Team X allow to Position Y?
        # Use OPP_TEAM_ABBREVIATION instead of ID
        agg_key = 'OPP_TEAM_ABBREVIATION'
        
        defensive_stats = df.groupby(['GAME_DATE', agg_key, 'POSITION_SIMPLE'])[['PTS', 'REB', 'AST']].sum().reset_index()
        defensive_stats.columns = ['GAME_DATE', agg_key, 'POSITION_SIMPLE', 'def_PTS', 'def_REB', 'def_AST']
        
        # Now calculating rolling averages for each Team + Position combo
        defensive_stats = defensive_stats.sort_values('GAME_DATE')
        
        cols = ['def_PTS', 'def_REB', 'def_AST']
        for col in cols:
            defensive_stats[f'ROLL_{col}'] = defensive_stats.groupby([agg_key, 'POSITION_SIMPLE'])[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            
        # 3. Merge back
        df = df.merge(defensive_stats[['GAME_DATE', agg_key, 'POSITION_SIMPLE', 'ROLL_def_PTS', 'ROLL_def_REB', 'ROLL_def_AST']],
                      on=['GAME_DATE', agg_key, 'POSITION_SIMPLE'],
                      how='left')
                      
        # Fill NaNs (start of season)
        for col in ['ROLL_def_PTS', 'ROLL_def_REB', 'ROLL_def_AST']:
            df[col] = df[col].fillna(df[col].mean()) # Fill with global average
        
        # 4. Per-Minute DvP (NEW)
        # We need sum of MIN for the position group to calculate rates
        # Re-aggregate including MIN
        def_stats_advanced = df.groupby(['GAME_DATE', agg_key, 'POSITION_SIMPLE'])[['PTS', 'REB', 'AST', 'MIN']].sum().reset_index()
        def_stats_advanced.columns = ['GAME_DATE', agg_key, 'POSITION_SIMPLE', 'def_PTS', 'def_REB', 'def_AST', 'def_MIN']
        
        # Calculate Per-Minute Rates for the Defense
        # def_PTS_PER_MIN = def_PTS / def_MIN
        # Avoid div/0
        def_stats_advanced['def_MIN'] = def_stats_advanced['def_MIN'].replace(0, 1) # Safety
        
        for stat in ['PTS', 'REB', 'AST']:
            def_stats_advanced[f'def_{stat}_PER_MIN'] = def_stats_advanced[f'def_{stat}'] / def_stats_advanced['def_MIN']
            
        # Rolling Avg of these Rates
        def_stats_advanced = def_stats_advanced.sort_values('GAME_DATE')
        for stat in ['PTS', 'REB', 'AST']:
            col = f'def_{stat}_PER_MIN'
            def_stats_advanced[f'ROLL_{col}'] = def_stats_advanced.groupby([agg_key, 'POSITION_SIMPLE'])[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            
        # Merge Per-Minute DvP
        merge_cols = ['GAME_DATE', agg_key, 'POSITION_SIMPLE'] + [f'ROLL_def_{s}_PER_MIN' for s in ['PTS', 'REB', 'AST']]
        df = df.merge(def_stats_advanced[merge_cols], on=['GAME_DATE', agg_key, 'POSITION_SIMPLE'], how='left')
        
        # Fill NaNs for new cols
        for stat in ['PTS', 'REB', 'AST']:
            col = f'ROLL_def_{stat}_PER_MIN'
            if col in df.columns:
                 df[col] = df[col].fillna(df[col].mean())

        print("DvP Features Added (including Per-Minute).")
        return df

    def calculate_h2h(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Head-to-Head (H2H)...")
        # Rolling avg against specific opponent
        # Group by Player, Opponent
        cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']
                
        # Ensure cols exist
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        
        # Sort by date to ensure rolling window is correct
        df = df.sort_values('GAME_DATE')
        
        # Optimize: Group ONCE for all columns instead of looping
        # This avoids re-sorting/hashing for every column (speedup 10x+)
        grouped = df.groupby(['PLAYER_ID', 'OPP_TEAM_ABBREVIATION'])[cols]
        
        # Apply transformation to all columns at once
        h2h_features = grouped.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        
        # Rename columns to H2H_ prefix
        h2h_features.columns = [f'H2H_{c}' for c in cols]
        
        # Merge back (concat is safe because index is preserved by transform)
        df = pd.concat([df, h2h_features], axis=1)
             
        # Fill NaNs with global mean
        for col in cols:
            val = df[col].mean()
            df[f'H2H_{col}'] = df[f'H2H_{col}'].fillna(val)
            
        print("H2H Features Added.")
        return df

    def _identify_stars(self, df: pd.DataFrame) -> dict:
        """
        Identifies top 2 scorers per team per season.
        Returns: Dict { (TeamID, Season) : [PlayerID1, PlayerID2] }
        """
        print("Identifying Star Players...")
        summary = df.groupby(['SEASON_YEAR', 'TEAM_ID', 'PLAYER_ID'])['PTS'].mean().reset_index()
        
        stars_map = {}
        
        for (season, team), group in summary.groupby(['SEASON_YEAR', 'TEAM_ID']):
            top_scorers = group.sort_values('PTS', ascending=False).head(2)
            stars_map[(team, season)] = top_scorers['PLAYER_ID'].tolist()
            
        return stars_map

    def calculate_teammate_absence(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Teammate Impact (Stars Out)...")
        stars_map = self._identify_stars(df)
        
        # Optimize: Group by Team/Date to get roster
        present_players = df.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID'].apply(set).to_dict()
        
        def count_stars_out(row):
            team = row['TEAM_ID']
            season = row['SEASON_YEAR']
            date = row['GAME_DATE']
            me = row['PLAYER_ID']
            
            stars = stars_map.get((team, season), [])
            if not stars: return 0
                
            present = present_players.get((team, date), set())
            
            missing = 0
            for star_id in stars:
                if star_id == me: continue 
                if star_id not in present:
                    missing += 1
            return missing

        df['STARS_OUT'] = df.apply(count_stars_out, axis=1)
        print("Teammate Impact Features Added.")
        return df

    def _identify_rotation_players(self, df: pd.DataFrame, min_minutes: float = 15.0) -> dict:
        """
        Identifies rotation players (avg > min_minutes) per team per season.
        Returns: Dict { (TeamID, Season) : Set(PlayerIDs) }
        """
        print("Identifying Rotation Players...")
        summary = df.groupby(['SEASON_YEAR', 'TEAM_ID', 'PLAYER_ID'])['MIN'].mean().reset_index()
        
        rotation_map = {}
        for (season, team), group in summary.groupby(['SEASON_YEAR', 'TEAM_ID']):
            rotation_players = group[group['MIN'] > min_minutes]['PLAYER_ID'].unique()
            rotation_map[(team, season)] = set(rotation_players)
            
        return rotation_map

    def calculate_missing_players(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Calculating Specific Missing Players...")
        rotation_map = self._identify_rotation_players(df)
        present_players = df.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID'].apply(set).to_dict()
        
        def get_missing(row):
            team = row['TEAM_ID']
            season = row['SEASON_YEAR']
            date = row['GAME_DATE']
            me = row['PLAYER_ID']
            
            # Expected roster
            expected = rotation_map.get((team, season), set())
            if not expected: return []
            
            # Who actually played
            present = present_players.get((team, date), set())
            
            missing = []
            for pid in expected:
                if pid == me: continue # Don't count myself as missing
                if pid not in present:
                    missing.append(pid)
            
            return missing

        # Apply and store as list
        # Using list in DataFrame cell is possible but requires care with saving/loading (CSV converts to string).
        # We should serialize to string "ID1_ID2_ID3" for CSV safety.
        df['MISSING_PLAYER_IDS'] = df.apply(get_missing, axis=1)
        # Convert to string for CSV compatibility
        df['MISSING_PLAYER_IDS'] = df['MISSING_PLAYER_IDS'].apply(lambda x: "_".join(map(str, x)) if x else "NONE")
        
        print("Missing Players Identified.")
        return df

    def calculate_per_minute_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates per-minute versions of base stats.
        STAT_PER_MIN = STAT / MAX(MIN, 1.0)
        """
        print("Calculating Per-Minute Stats...")
        
        targets = ['PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL']
        
        # Ensure MIN is safe for division
        # Create a temporary safe MIN column
        safe_min = df['MIN'].apply(lambda x: max(x, 1.0))
        
        for stat in targets:
            if stat in df.columns:
                df[f'{stat}_PER_MIN'] = df[stat] / safe_min
                # If original stat was 0, it remains 0.
                
        return df

    def calculate_minutes_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling minutes statistics for the Minutes Predictor.
        """
        print("Calculating Minutes Features...")
        
        if 'MIN' not in df.columns:
            return df
            
        # Recent Avg (Last 10)
        df['recent_min_avg'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        
        # Recent Std (Last 10) - captures volatility
        df['recent_min_std'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).std())
        
        # Season Avg (Expanding)
        # shift(1) to avoid leaking today's minutes
        df['season_min_avg'] = df.groupby('PLAYER_ID')['MIN'].transform(lambda x: x.shift(1).expanding().mean())
        
        # Fill NaNs
        # First game: Use global average or 0? 
        # Usually we want some reasonable default.
        # Let's fill with 20.0 (bench/rotation avg) or just 0 if unknown.
        df['recent_min_avg'] = df['recent_min_avg'].fillna(20.0)
        df['recent_min_std'] = df['recent_min_std'].fillna(0.0)
        df['season_min_avg'] = df['season_min_avg'].fillna(20.0)
        
        return df

    def _derive_opponent(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses MATCHUP to get Opponent Abbreviation."""
        # MATCHUP: "LAL vs. TOR" or "LAL @ TOR"
        # Split by ' ' and take last element.
        df['OPP_TEAM_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        return df

    def _load_with_without_cache(self) -> dict:
        """
        Loads with/without splits and star maps from persistent cache.
        Returns dict with 'splits', 'stars_map', and optional 'names'.
        """
        if os.path.exists(WITH_WITHOUT_CACHE_PATH):
            try:
                data = joblib.load(WITH_WITHOUT_CACHE_PATH)
                return data # Return full object (splits, names, stars_map)
            except Exception as e:
                print(f"Error loading with/without cache: {e}")
                return {}
        return {}

    def _save_with_without_cache(self, splits: dict, names: dict = None, stars_map: dict = None):
        """Saves with/without splits and star maps to persistent cache."""
        os.makedirs(os.path.dirname(WITH_WITHOUT_CACHE_PATH), exist_ok=True)
        data = {
            'last_updated': pd.Timestamp.now(),
            'splits': splits,
            'names': names or {},
            'stars_map': stars_map or {}
        }
        joblib.dump(data, WITH_WITHOUT_CACHE_PATH)
        print(f"Saved With/Without Splits & Star Map to {WITH_WITHOUT_CACHE_PATH}")

    def _build_with_without_splits(self, df: pd.DataFrame):
        """
        Builds historical performance splits.
        """
        print("Building With/Without Splits Cache...")
        
        # Capture ID->Name Map
        id_to_name = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates().set_index('PLAYER_ID')['PLAYER_NAME'].to_dict()
        
        # ... (rest is same, but pass id_to_name to save)

        # We need per-season identification
        stars_map = {} # (Team, Season) -> [PIDs]
        
        # Calculate Season Avg stats first if not present
        if 'USG_PCT' not in df.columns:
            # Should have been calculated by now if called in correct order
            return 
            
        summary = df.groupby(['SEASON_YEAR', 'TEAM_ID', 'PLAYER_ID'])[['USG_PCT', 'MIN']].mean().reset_index()
        
        for (season, team), group in summary.groupby(['SEASON_YEAR', 'TEAM_ID']):
            # Filter low minute players (garbage time usage outliers)
            valid = group[group['MIN'] > 15]
            if valid.empty: valid = group
            
            top_usage = valid.sort_values('USG_PCT', ascending=False).head(3)
            stars_map[(team, season)] = top_usage['PLAYER_ID'].tolist()
            
        # 2. Iterate through each team's games
        # This is computationally expensive, so we optimize.
        
        splits = {} # {pid: {teammate_id: {'with': stats, 'without': stats}}}
        
        # Group by Team to process roster combos
        # We only care about "Active" players impact on "Active" players
        
        present_players = df.groupby(['TEAM_ID', 'GAME_DATE'])['PLAYER_ID'].apply(set).to_dict()
        
        # Iterate over all players who played? No, iterate over team games.
        # Check all rotation players.
        
        metrics = ['USG_PCT', 'PTS', 'FGA', 'MIN', 'TOUCHES']
        
        # Loop through identifying stars and calculate their impact on everyone else
        # This is O(Games * RosterSize * KeyTeammates)
        
        # Optimization: Pre-calculate per-game stats dict for fast lookup
        # (pid, date) -> {stats}
        # But DataFrame lookup with multi-index is okay if sorted.
        
        # Let's reduce scope to just "Rotation Players" affected by "Stars"
        # We already have _build_rotation_map logic inside identify_rotation?
        # Let's just do it dynamically.
        
        for (season, team), stars in stars_map.items():
            # Get all games for this team-season
            team_season_df = df[(df['TEAM_ID'] == team) & (df['SEASON_YEAR'] == season)]
            if team_season_df.empty: continue
            
            # Get list of all players who played significant minutes
            roster = team_season_df.groupby('PLAYER_ID')['MIN'].mean()
            roster = roster[roster > 10].index.tolist()
            
            for star_id in stars:
                # Identify games where Star was IN vs OUT
                # Star IN: star_id in present_players[(team, date)]
                # Star OUT: star_id NOT in present_players AND (team, date) is valid game
                
                # We need all dates this team played
                team_dates = team_season_df['GAME_DATE'].unique()
                
                games_with = []
                games_without = []
                
                for d in team_dates:
                    roster_today = present_players.get((team, d), set())
                    if star_id in roster_today:
                        games_with.append(d)
                    else:
                        games_without.append(d)
                
                if not games_without: continue # Star never missed a game
                if len(games_without) < MIN_SPLIT_SAMPLE_SIZE: continue # Sample too small
                
                # Now calculate impact on Teammates
                for teammate_id in roster:
                    if teammate_id == star_id: continue
                    
                    if teammate_id not in splits: splits[teammate_id] = {}
                    
                    # Get stats for Teammate in 'with' games
                    p_with = team_season_df[(team_season_df['PLAYER_ID'] == teammate_id) & (team_season_df['GAME_DATE'].isin(games_with))]
                    p_without = team_season_df[(team_season_df['PLAYER_ID'] == teammate_id) & (team_season_df['GAME_DATE'].isin(games_without))]
                    
                    if p_without.empty: continue
                    
                    # Calculate aggregates
                    stats_with = p_with[metrics].mean().to_dict()
                    stats_without = p_without[metrics].mean().to_dict()
                    
                    # Store
                    splits[teammate_id][star_id] = {
                        'with': stats_with,
                        'without': stats_without,
                        'sample_without': len(p_without)
                    }
                    
                    
        self._save_with_without_cache(splits, names=id_to_name, stars_map=stars_map)

    def calculate_usage_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Usage Rate (USG%) for each player-game.
        Formula: 100 * ((FGA + 0.44 * FTA + TOV) * (Team_MIN / 5)) / (MIN * (Team_FGA + 0.44 * Team_FTA + Team_TOV))
        """
        print("Calculating Usage Rates...")
        
        # 1. Aggregate Team Stats per Game needed for formula
        # We need Team FGA, Team FTA, Team TOV, Team MIN
        team_stats = df.groupby(['GAME_ID', 'TEAM_ID'])[['FGA', 'FTA', 'TOV', 'MIN']].sum().reset_index()
        team_stats = team_stats.rename(columns={
            'FGA': 'TEAM_FGA', 
            'FTA': 'TEAM_FTA', 
            'TOV': 'TEAM_TOV', 
            'MIN': 'TEAM_MIN'
        })
        
        # 2. Merge back to player rows
        df = df.merge(team_stats, on=['GAME_ID', 'TEAM_ID'], how='left')
        
        # 3. Calculate USG%
        # Handle zero MIN to avoid div/0
        
        # (FGA + 0.44 * FTA + TOV)
        player_poss = df['FGA'] + 0.44 * df['FTA'] + df['TOV']
        
        # (Team_FGA + 0.44 * Team_FTA + Team_TOV)
        team_poss = df['TEAM_FGA'] + 0.44 * df['TEAM_FTA'] + df['TEAM_TOV']
        
        term1 = player_poss * (df['TEAM_MIN'] / 5)
        term2 = df['MIN'] * team_poss
        
        df['USG_PCT'] = 100 * (term1 / term2)
        
        df['USG_PCT'] = 100 * (term1 / term2)
        
        # Fill NaNs/Infs (e.g. 0 minutes played)
        df['USG_PCT'] = df['USG_PCT'].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate Proxy TOUCHES: 
        # FGA + (FTA * 0.44) + TOV + AST
        # (Passes are touches, but we only have AST. This is a rough proxy for "Ball Dominance")
        if 'AST' in df.columns:
            df['TOUCHES'] = df['FGA'] + (df['FTA'] * 0.44) + df['TOV'] + df['AST']
        else:
            df['TOUCHES'] = df['FGA'] + (df['FTA'] * 0.44) + df['TOV']
            
        df['TOUCHES'] = df['TOUCHES'].fillna(0)

        # Added per-minute version for opportunity analysis
        safe_min = df['MIN'].apply(lambda x: max(x, 1.0))
        df['TOUCHES_PER_MIN'] = df['TOUCHES'] / safe_min
        
        # 4. Rolling USG%
        # df is already sorted by date in process() usually, but safe to verify
        df['ROLL_USG_PCT'] = df.groupby('PLAYER_ID')['USG_PCT'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        df['ROLL_USG_PCT'] = df['ROLL_USG_PCT'].fillna(df['USG_PCT'].mean())
        
        print("Usage Rate Features Added.")
        return df

    def _apply_injury_severity_weights(self, df: pd.DataFrame, injury_report: dict) -> pd.DataFrame:
        """
        Adjusts opportunity boosts based on injury severity from the report.
        {'Out': 1.0, 'Doubtful': 0.7, 'Questionable': 0.3}
        """
        if not injury_report:
            return df
            
        print("Applying Injury Severity Weights...")
        
        # 1. Map Player IDs to Names (reverse lookup needed or fuzzy match)
        # MISSING_PLAYER_IDS contains IDs like "123_456".
        # Injury Report has Names "LeBron James": "Out".
        
        # We need to lookup the Status of each Missing ID.
        # This requires ID -> Name mapping.
        # We can build a quick map from the DF itself (PLAYER_ID -> PLAYER_NAME)
        # But MISSING_PLAYER_IDS are Teammates, not necessarily in the current row's Player Name.
        # However, they should exist in the dataframe somewhere if they are relevant.
        
        id_to_name = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates().set_index('PLAYER_ID')['PLAYER_NAME'].to_dict()
        
        def calculate_severity_total(missing_str):
            if not missing_str or missing_str == "NONE": return 0.0
            
            ids = missing_str.split('_')
            total_weight = 0.0
            
            for pid_str in ids:
                try:
                    pid = int(pid_str)
                    pname = id_to_name.get(pid)
                    if not pname: continue
                    
                    # Fuzzy match against injury report
                    # We can use the cached fuzzy match helper if available or direct lookup
                    # optimization: try direct first
                    status = injury_report.get(pname)
                    if not status:
                        # Try fuzzy
                        match_name = _fuzzy_match_player_name(pname, injury_report.keys())
                        status = injury_report.get(match_name, "Active")
                        
                    weight = INJURY_SEVERITY_WEIGHTS.get(status, 0.0)
                    total_weight += weight
                except:
                    continue
                    
            return total_weight

        # Calculate Total Severity Score for the row (Missing Players context)
        # This represents "How Verified/Severe is the absence?"
        # If ID is in MISSING_PLAYER_IDS (calculated from boxscores), it means they ARE missing in historical data.
        # But for *Prediction* (overrides), we rely on Injury Report.
        # For Training, we assume Severity = 1.0 (Confirmed Out).
        # So this method assumes we are in Prediction mode or treating historicals as fully confirmed.
        # Actually, for training, strict "Out" is via boxscore check.
        
        # Applying weight to the BOOST features.
        # If we are training, we trust the boxscore: Weight = 1.0.
        # If we are predicting, we trust the report: Weight = Status.
        
        df['INJURY_SEVERITY_TOTAL'] = df['MISSING_PLAYER_IDS'].apply(calculate_severity_total)
        
        # Scale the boosts
        # If Severity is 0 (Active), Boost becomes 0.
        # If Severity is 0.3 (Questionable), Boost is dampened.
        
        # Note: If multiple players missing, we need per-player weight * per-player boost.
        # The current aggregation in calculate_star_usage_shift sums them up.
        # Ideally we apply weight during summation.
        # But here we do a post-hoc adjustment?
        # Approximation: Weighted Score = Raw Score * (Avg Severity)?
        # Or Severity Total?
        # Let's simple-scale for now: Weighted = Raw * (Total Severity / Count Missing) ?
        # No, let's keep it simple: Weighted Opportunity = Raw Opportunity * (Severity of the Missing Star).
        # Since Raw Opportunity is a sum, we can't easily disentangle without re-looping.
        # Let's adjust calculate_star_usage_shift to accept injury_report and do it there!
        
        return df

    def calculate_star_usage_shift(self, df: pd.DataFrame, injury_report: dict = None) -> pd.DataFrame:
        """
        Quantifies opportunity redistribution when stars are missing.
        Calculates delta in USG, PTS, FGA based on historical 'without' splits.
        """
        print("Calculating Star Usage Shift (Opportunity)...")
        cache_data = self._load_with_without_cache()
        
        # cache_data is now dict: {'splits': ..., 'names': ..., 'stars_map': ...} or empty
        splits = cache_data.get('splits', {})
        cached_names = cache_data.get('names', {})
        stars_map = cache_data.get('stars_map', {})
        
        if not splits:
            print("No with/without splits found. Skipping opportunity features.")
            columns_to_clear = ['OPP_SCORE', 'OPP_USG_BOOST', 'OPP_MIN_BOOST', 'OPP_FGA_BOOST', 'OPP_TOUCHES_BOOST', 'INJURY_SEVERITY_TOTAL', 
                                'OPP_SCORE_WEIGHTED', 'OPP_USG_WEIGHTED', 'STARS_OUT']
            for c in columns_to_clear:
                df[c] = 0.0
            return df
            
        # Helper map for ID -> Name
        # 1. From DF (present players) - reliable for Present
        id_to_name = df[['PLAYER_ID', 'PLAYER_NAME']].drop_duplicates().set_index('PLAYER_ID')['PLAYER_NAME'].to_dict()
        
        # 2. Merge with Cache (Legacy/Missing players)
        # Cache wins if DF is missing it? Or DF wins? DF is strictly current. Cache is broad.
        # Use cache as base, update with DF.
        full_id_to_name = cached_names.copy()
        full_id_to_name.update(id_to_name)
        id_to_name = full_id_to_name
        
        
        def calculate_boost(row):
            me = int(row['PLAYER_ID'])
            team = int(row['TEAM_ID'])
            season = int(row['SEASON_YEAR'])
            missing_str = row['MISSING_PLAYER_IDS']
            
            if not missing_str or missing_str == "NONE":
                return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
            missing_ids = [int(i) for i in missing_str.split('_')]
            
            usg_delta_sum = 0.0
            min_delta_sum = 0.0
            fga_delta_sum = 0.0
            touches_delta_sum = 0.0
            severity_total = 0.0
            stars_out_count = 0.0
            
            my_splits = splits.get(me, {})
            team_stars = stars_map.get((team, season), [])
            
            for missing_id in missing_ids:
                try:
                    # 1. Determine Weight (Matches previous logic)
                    weight = 1.0
                    if injury_report:
                        pname = id_to_name.get(missing_id)
                        if pname:
                            status = injury_report.get(pname)
                            if not status:
                                 match_name = _fuzzy_match_player_name(pname, injury_report.keys())
                                 status = injury_report.get(match_name, "Active")
                            if status:
                                weight = INJURY_SEVERITY_WEIGHTS.get(status, 1.0)
                    
                    severity_total += weight
                    
                    # Check if this missing player is a star
                    if missing_id in team_stars:
                        stars_out_count += weight if injury_report else 1.0
                    
                    # 2. Get Deltas
                    if missing_id in my_splits:
                        s = my_splits[missing_id]
                        
                        # With vs Without
                        # Delta = Without - With
                        w_usg = s['with'].get('USG_PCT', 0)
                        wo_usg = s['without'].get('USG_PCT', 0)
                        
                        w_min = s['with'].get('MIN', 0)
                        wo_min = s['without'].get('MIN', 0)
                        
                        w_fga = s['with'].get('FGA', 0)
                        wo_fga = s['without'].get('FGA', 0)
                        
                        w_tch = s['with'].get('TOUCHES', 0)
                        wo_tch = s['without'].get('TOUCHES', 0)
                        
                        usg_delta_sum += (wo_usg - w_usg) * weight
                        min_delta_sum += (wo_min - w_min) * weight
                        fga_delta_sum += (wo_fga - w_fga) * weight
                        touches_delta_sum += (wo_tch - w_tch) * weight
                        
                except Exception as e:
                    continue
            
            # Weighted Score
            # (USG_BOOST * 0.4) + (MIN_BOOST * 0.2) + (FGA_BOOST * 0.2) + (TOUCHES_BOOST * 0.2)
            opp_score = (usg_delta_sum * 0.4) + (min_delta_sum * 0.2) + (fga_delta_sum * 0.2) + (touches_delta_sum * 0.2)
            
            # Cap realistic boosts
            usg_delta_sum = min(max(usg_delta_sum, -5.0), MAX_OPPORTUNITY_BOOST)
            
            return opp_score, usg_delta_sum, min_delta_sum, fga_delta_sum, touches_delta_sum, severity_total, stars_out_count

        # Apply
        results = df.apply(calculate_boost, axis=1)
        
        # Unpack
        df['OPP_SCORE'] = [x[0] for x in results]
        df['OPP_USG_BOOST'] = [x[1] for x in results]
        df['OPP_MIN_BOOST'] = [x[2] for x in results]
        df['OPP_FGA_BOOST'] = [x[3] for x in results]
        df['OPP_TOUCHES_BOOST'] = [x[4] for x in results]
        df['INJURY_SEVERITY_TOTAL'] = [x[5] for x in results]
        df['STARS_OUT'] = [x[6] for x in results]
        
        # Weighted features (Directly redundant if weight applied above? 
        
        df['OPP_SCORE_WEIGHTED'] = df['OPP_SCORE'] # Already weighted
        df['OPP_USG_WEIGHTED'] = df['OPP_USG_BOOST']
        
        print("Star Usage Shift Features Added.")
        return df

    def calculate_team_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Team-Level Advanced Stats (Off Rtg, Def Rtg, Pace)
        and merges them as 'Opponent Stats' for the player.
        """
        print("Calculating Team Advanced Stats...")
        
        # 1. Aggregate Player Stats to Team-Game Level
        # Sum targets: PTS, FGA, FTA, OREB, TOV, MIN
        # We need DREB? Total REB - OREB = DREB usually.
        # Poss formula: 0.96 * (FGA + 0.008*FTA - 1.07*(OREB/ (OREB + OppDREB))) ... too complex.
        # Simple NBA Poss: FGA + 0.44*FTA - OREB + TOV
        
        team_game_stats = df.groupby(['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'TEAM_ABBREVIATION'])[['PTS', 'FGA', 'FTA', 'OREB', 'TOV', 'MIN']].sum().reset_index()
        
        # 2. Calculate Basic Team Metrics
        team_game_stats['POSS'] = team_game_stats['FGA'] + 0.44 * team_game_stats['FTA'] - team_game_stats['OREB'] + team_game_stats['TOV']
        # Normalize Pace to 48 mins: Poss / (MIN / 5) * 48. (5 players on court)
        # Avoid div by zero
        team_game_stats['PACE'] = 48 * (team_game_stats['POSS'] / (team_game_stats['MIN'] / 5))
        team_game_stats['OFF_RTG'] = 100 * (team_game_stats['PTS'] / team_game_stats['POSS'])
        
        # 3. Find Opponent to get DEF_RTG (Opponent's Off Rtg)
        # Self-join on GAME_ID
        game_opps = team_game_stats.merge(team_game_stats[['GAME_ID', 'TEAM_ID', 'OFF_RTG', 'PACE']], 
                                          on='GAME_ID', suffixes=('', '_OPP'))
        
        # Filter out self-matches (Team A vs Team A) - usually game has 2 rows.
        game_opps = game_opps[game_opps['TEAM_ID'] != game_opps['TEAM_ID_OPP']]
        
        # Now we have: For Team A, DEF_RTG = OPP_OFF_RTG
        game_opps['DEF_RTG'] = game_opps['OFF_RTG_OPP']
        
        # We want to store these stats FOR THE TEAM, so we can look them up as an OPPONENT later.
        # Identify the Team we are tracking metrics FOR.
        # We want "How good is Team X?".
        # So we keep rows for Team X.
        
        cols_to_keep = ['GAME_DATE', 'TEAM_ABBREVIATION', 'OFF_RTG', 'DEF_RTG', 'PACE']
        team_history = game_opps[cols_to_keep].copy()
        team_history = team_history.sort_values('GAME_DATE')
        
        # 4. Calculate Rolling Stats
        # Group by Team
        roll_cols = ['OFF_RTG', 'DEF_RTG', 'PACE']
        for col in roll_cols:
            # Shift 1 to use PAST games only
            team_history[f'TEAM_ROLL_{col}'] = team_history.groupby('TEAM_ABBREVIATION')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
            
        # Fill NaNs with league average
        for col in roll_cols:
            mean_val = team_history[col].mean()
            team_history[f'TEAM_ROLL_{col}'] = team_history[f'TEAM_ROLL_{col}'].fillna(mean_val)
            
        # 5. Merge back to Player DF as "OPP_TEAM_ROLL_..."
        # df has 'OPP_TEAM_ABBREVIATION'.
        # We join team_history on [TEAM_ABBREV, GAME_DATE] -> [OPP_TEAM_ABBREV, GAME_DATE]
        
        # Prepare merge source
        merge_src = team_history[['GAME_DATE', 'TEAM_ABBREVIATION', 'TEAM_ROLL_OFF_RTG', 'TEAM_ROLL_DEF_RTG', 'TEAM_ROLL_PACE']]
        merge_src.columns = ['GAME_DATE', 'OPP_TEAM_ABBREVIATION', 'OPP_ROLL_OFF_RTG', 'OPP_ROLL_DEF_RTG', 'OPP_ROLL_PACE']
        
        # Merge
        df = df.merge(merge_src, on=['GAME_DATE', 'OPP_TEAM_ABBREVIATION'], how='left')
        
        # Fill missing (e.g. first game of season or no history)
        # Fill with global means from the merge_src (approx league avg)
        for col in ['OPP_ROLL_OFF_RTG', 'OPP_ROLL_DEF_RTG', 'OPP_ROLL_PACE']:
             df[col] = df[col].fillna(merge_src[col].mean())
             
        print("Team Advanced Stats Added.")
        return df


    def process(self, df: pd.DataFrame, is_training: bool = True, overrides: dict = None) -> pd.DataFrame:
        """
        overrides: Dict of {(PlayerID, Date_Str) : {'Col': Val}} to manually set feature values before scaling.
        Also supports 'INJURY_REPORT' in overrides for prediction context.
        """
        print("Starting Feature Engineering...")
        if is_training:
            # Filter garbage
            df = df.dropna(subset=['PTS', 'REB', 'AST'])
            
        # Ensure Date Format Early
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        
        # Derive Opponent Info EARLY
        df = self._derive_opponent(df)
        
        # Add Context Features (Home/Away, Rest Days)
        df = self._add_context_features(df)
        
        # Load Positions & Add DvP
        pos_df = self.load_player_positions()
        if not pos_df.empty:
            df = self.calculate_dvp(df, pos_df)
        else:
             print("Skipping DvP (No position data)")
             
        # Add H2H
        df = self.calculate_h2h(df)
        
        # Add Usage Rate (NEW)
        df = self.calculate_usage_rate(df)
        
        # Add Team Stats (Defense/Pace)
        df = self.calculate_team_stats(df)
        
        # Add Teammate Impact
        df = self.calculate_missing_players(df)
        
        # APPLY OVERRIDES (Pre-Scaling) -- Moved here per verification comment
        # Must apply overrides BEFORE per-minute/minutes calc so they use the override values.
        injury_report_override = None
        
        if overrides:
            print(f"Applying manual overrides...")
            for (pid, date_str), vals in overrides.items():
                target_dt = pd.to_datetime(date_str)
                mask = (df['PLAYER_ID'] == pid) & (df['GAME_DATE'] == target_dt)
                if mask.any():
                    for col, val in vals.items():
                        # Extract INJURY_REPORT if present
                        if col == 'INJURY_REPORT':
                            injury_report_override = val
                            continue
                            
                        # If overriding MISSING_PLAYER_IDS, ensure string format
                        if col == 'MISSING_PLAYER_IDS' and isinstance(val, list):
                            val = "_".join(map(str, val))
                        
                        df.loc[mask, col] = val
                        # print(f"Overrode {col} for {pid} on {date_str}")
        
        # Add Per-Minute Stats (NEW)
        df = self.calculate_per_minute_stats(df)
        
        # Add Minutes Features (NEW)
        df = self.calculate_minutes_features(df)


        
        # Build Splits Cache (Training Only) (NEW) - ORDER FIXED: Build BEFORE Shift Calc
        if is_training:
            self._build_with_without_splits(df)
            
        # Add Star Usage Shift Features (NEW)
        # Pass injury report if available from overrides (Prediction Mode)
        # Now cache should be populated for the current run if is_training was True
        df = self.calculate_star_usage_shift(df, injury_report=injury_report_override)
        
        # Sort
        # df['GAME_DATE'] is already datetime
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # 1. Rolling Averages (Player Form)
        # Shift by 1 so we don't cheat using today's stats
        # Add Per-Minute Stats to rolling window
        cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS', 'MIN',
                'USG_PCT',
                'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'FG3M_PER_MIN', 'BLK_PER_MIN', 'STL_PER_MIN'] # Added Per-Minute
        for col in cols:
             if col in df.columns:
                 # Standard Last 5
                 df[f'ROLL_{col}'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
                 
                 # Extended Windows for Per-Minute Stats (Comment 2)
                 if '_PER_MIN' in col:
                     # Add Last 10
                     df[f'ROLL_{col}_10'] = df.groupby('PLAYER_ID')[col].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
             
        # 2. Rest Days
        df['GAME_DATE_DT'] = df['GAME_DATE']
        df['PREV_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE_DT'].shift(1)
        df['DAYS_REST'] = (df['GAME_DATE_DT'] - df['PREV_GAME_DATE']).dt.days
        df['DAYS_REST'] = df['DAYS_REST'].fillna(7)
        df['DAYS_REST'] = df['DAYS_REST'].clip(upper=7)
        
        # Capture RAW features for Minutes Predictor before scaling
        min_pred_features = ['recent_min_avg', 'recent_min_std', 'season_min_avg', 'INJURY_SEVERITY_TOTAL', 'STARS_OUT', 'OPP_ROLL_PACE', 'IS_HOME', 'DAYS_REST']
        for f in min_pred_features:
            if f in df.columns:
                df[f'{f}_RAW'] = df[f]
        
        # 3. Apply Scaling / Encoding
        if is_training:
            df = self._fit_artifacts(df, is_training=True) 
        else:
            # For inference, use saved scalers
            df = self._apply_transformations(df, is_training=False)
        
        # Drop NaNs
        # Important: Rolling stats take 1 game to start. First row is NaN.
        # If inference key row is the first row (impossible if history loaded), it might be dropped.
        if is_training:
            df = df.dropna(subset=['ROLL_PTS']) # Drop rows without features
        else:
            # Keep everything, let caller extract the last row
            pass
        
        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Home/Away
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if ' vs. ' in x else 0)
        
        # Rest Days
        df['REST_DAYS'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
        df['REST_DAYS'] = df['REST_DAYS'].fillna(3) # Assume 3 days rest for first game
        df['IS_B2B'] = (df['REST_DAYS'] == 1).astype(int)
        
        # Parse Season Year (for weighting)
        # SEASON_ID is usually like 22023 -> 2023.
        # Robust handling for float strings (e.g. '22024.0')
        s_ids = df['SEASON_ID'].fillna(22025).astype(str)
        s_ids = s_ids.str.replace(r'\.0$', '', regex=True)
        # Consistent with load_all_data: 22025 -> 2026 (2025-26 Season)
        df['SEASON_YEAR'] = s_ids.str[-4:].astype(int) + 1
        
        return df

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        rolling_metrics = ['PTS', 'REB', 'AST', 'MIN', 'FGM', 'FGA', 'FG3M', 'FG3A']
        windows = [5, 10, 20]
        
        # Vectorized Rolling would be faster but GroupBy is safer for strict player isolation
        # Using a simplistic loop for clarity/safety.
        
        # Pre-sort (already done)
        for metric in rolling_metrics:
            if metric not in df.columns: continue
            
            # Group shift first to prevent data leak
            shifted = df.groupby('PLAYER_ID')[metric].shift(1)
            
            for w in windows:
                col_name = f'ROLL_{metric}_{w}'
                # Calculate rolling on shifted data
                df[col_name] = shifted.groupby(df['PLAYER_ID']).rolling(window=w, min_periods=1).mean().reset_index(0, drop=True)
                
            # Season Average
            col_name_season = f'AVG_{metric}_SEASON'
        return df

    def _fit_artifacts(self, df: pd.DataFrame, is_training: bool = True):
        if not is_training:
             # Just load and transform? actually _fit_artifacts implies fitting.
             # For inference we should use separate method.
             # But let's just ignore for now or return df.
             # Actually, if is_training=False, we shouldn't be here?
             # 'process' calls it.
             # Let's just allow the arg.
             pass
             
        print("Fitting scalers and encoders...")
        
        # Categoricals
        player_encoder = LabelEncoder()
        df['PLAYER_IDX'] = player_encoder.fit_transform(df['PLAYER_ID'].astype(str))
        
        team_encoder = LabelEncoder()
        df['TEAM_IDX'] = team_encoder.fit_transform(df['OPP_TEAM_ABBREVIATION'].astype(str))
        
        # Continuous Features
        # Exclude ID, Date, Target, and categorical strings
        # Keep ROLL_*, H2H_*, DAYS_REST
        exclude = ['PLAYER_ID', 'SEASON_ID', 'DEV_SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 
                   'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'VIDEO_AVAILABLE', 
                   'PLAYER_NAME', 'OPP_TEAM_ABBREVIATION', 'SEASON_YEAR', 
                   'PLAYER_IDX', 'TEAM_IDX', 'GAME_DATE_DT', 'PREV_GAME_DATE',
                   'POSITION_SIMPLE', 'def_PTS', 'def_REB', 'def_AST',
                   'POSS', 'OFF_RTG', 'DEF_RTG', 'PACE', 'TEAM_ABBREVIATION_OPP'] # Exclude intermediate calc
                   
        # Also exclude Targets
        exclude += ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'FGM', 'FGA', 'FG_PCT', 
                    'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'PLUS_MINUS', 'FANTASY_PTS',
                    'USG_PCT'] # Exclude raw USG_PCT as it is a target-derived stat unavailable at inference time
        
        # Also exclude raw Per-Minute stats (Targets)
        exclude += ['PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'FG3M_PER_MIN', 'BLK_PER_MIN', 'STL_PER_MIN']
        
        feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
        print(f"Identified {len(feature_cols)} feature columns: {feature_cols}")
        
        # Fit SELF.scaler
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols].fillna(0))
        
        # Save Artifacts
        joblib.dump(player_encoder, os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        joblib.dump(team_encoder, os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        joblib.dump(self.scaler, os.path.join(MODELS_DIR, 'scaler.joblib')) # Save self.scaler
        joblib.dump(feature_cols, os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        return df

    def _load_artifacts(self):
        self.player_encoder = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        self.team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        self.scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))

    def _apply_transformations(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        if is_training:
            return df # Already transformed in _fit_artifacts
            
        # 1. Feature Columns
        self._load_artifacts() # Ensure scaler/encoders are loaded
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Scale continuous
        # Ensure all columns exist (fill with 0 if missing - e.g. new features in old model?)
        # But we are retraining so feature_cols should match.
        for c in feature_cols:
            if c not in df.columns:
                 df[c] = 0.0
                 
        df[feature_cols] = self.scaler.transform(df[feature_cols].fillna(0))
        
        # 2. Embeddings Indices
        # Player (Handle unknown)
        df['PLAYER_ID_STR'] = df['PLAYER_ID'].astype(str)
        # Map unknown to 'UNKNOWN'
        # Map unknown to a known class (fallback) to avoid crash
        known_players = set(self.player_encoder.classes_)
        # Use first class as generic fallback if current ID is unknown
        fallback = self.player_encoder.classes_[0] 
        df.loc[~df['PLAYER_ID_STR'].isin(known_players), 'PLAYER_ID_STR'] = fallback
        
        df['PLAYER_IDX'] = self.player_encoder.transform(df['PLAYER_ID_STR'])
        
        # Team
        df['OPP_TEAM_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split(' ')[-1])
        known_teams = set(self.team_encoder.classes_)
        fallback_team = self.team_encoder.classes_[0]
        df.loc[~df['OPP_TEAM_ABBREVIATION'].isin(known_teams), 'OPP_TEAM_ABBREVIATION'] = fallback_team
        
        df['TEAM_IDX'] = self.team_encoder.transform(df['OPP_TEAM_ABBREVIATION'])
        
        return df

    def save_features(self, df: pd.DataFrame, filename: str = 'processed_features.csv'):
        # Save minimal dataset for Training to save space/IO
        # But we need targets (PTS, REB, AST) and Season Year
        # We must include the new columns!
        # feature_cols joblib already has the names? 
        # Yes, _fit_artifacts logic below needs to include them in the 'continuous_cols' list.
        # But here we explicitly list non-feature metadata key columns.
        cols = ['PLAYER_ID', 'PLAYER_IDX', 'TEAM_IDX', 'SEASON_YEAR', 'GAME_DATE', 'PTS', 'REB', 'AST', 'FG3M', 'BLK', 'STL', 'MIN', 'MISSING_PLAYER_IDS', 'USG_PCT', 'PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'FG3M_PER_MIN', 'BLK_PER_MIN', 'STL_PER_MIN']
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Ensure all columns exist
        final_cols = list(set(cols + feature_cols)) # Dedup just in case
        final_df = df[final_cols]
        final_df.to_csv(os.path.join(PROCESSED_DIR, filename), index=False)
        print(f"Saved processed features to {filename}")


if __name__ == "__main__":
    fe = FeatureEngineer()
    try:
        df = fe.load_all_data()
        df_processed = fe.process(df)
        fe.save_features(df_processed)
        print(f"Processed {len(df_processed)} rows.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
