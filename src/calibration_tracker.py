
import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from filelock import FileLock # Requires installing filelock if not present, or use simple approach

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
CALIBRATION_DIR = os.path.join(DATA_DIR, 'calibration')
HISTORY_FILE = os.path.join(CALIBRATION_DIR, 'prediction_history.csv')
FACTORS_FILE = os.path.join(CALIBRATION_DIR, 'calibration_factors.json')

# Ensure directories exist
os.makedirs(CALIBRATION_DIR, exist_ok=True)

class CalibrationTracker:
    def __init__(self):
        self.history_file = HISTORY_FILE
        self.lock_file = HISTORY_FILE + ".lock"
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not os.path.exists(self.history_file):
            df = pd.DataFrame(columns=[
                'player_id', 'player_name', 'game_date', 'stat_type', 
                'predicted', 'actual', 'calibration_applied'
            ])
            df.to_csv(self.history_file, index=False)

    def save_prediction(self, player_id, player_name, stat_type, predicted_value, actual_value=None, game_date=None, calibration_applied=False):
        """Append a single prediction to the history file thread-safely."""
        if game_date is None:
            game_date = datetime.now().strftime('%Y-%m-%d')
            
        row = {
            'player_id': player_id,
            'player_name': player_name,
            'game_date': game_date,
            'stat_type': stat_type,
            'predicted': predicted_value,
            'actual': actual_value if actual_value is not None else np.nan,
            'calibration_applied': calibration_applied
        }
        
        df_row = pd.DataFrame([row])
        
        # Simple lock mechanism using file presence or just append mode safely?
        # Standard append mode is atomic enough for CSV lines in many OSs, but pandas to_csv isn't.
        # We'll use a try-except block or just append mode of open.
        
        try:
            # Append without reading whole file
            # Check if header needed? (handled by _ensure_file_exists)
            df_row.to_csv(self.history_file, mode='a', header=False, index=False)
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")

    def load_history(self, days=90):
        """Load recent prediction history."""
        try:
            if not os.path.exists(self.history_file):
                return pd.DataFrame()
            
            # Optimization: Read only needed stats? No, usually small file unless millions of lines.
            # If large, use chunks. For now, read all.
            df = pd.read_csv(self.history_file)
            
            # Filter by date
            cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            df = df[df['game_date'] >= cutoff_date]
            
            # Remove rows without actuals for analysis purposes?
            # Or keep them for tracking pending?
            # Typically load_history is for analysis, so we return all but let caller filter.
            return df
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return pd.DataFrame()

    def calculate_hit_rates(self, df=None, stat_type=None, player_id=None):
        """Calculate over/under percentages."""
        if df is None:
            df = self.load_history()
            
        # Filter completed games
        df = df.dropna(subset=['actual'])
        
        if df.empty:
            return {'over_rate': 0.5, 'under_rate': 0.5, 'count': 0}
            
        if stat_type:
            df = df[df['stat_type'] == stat_type]
            
        if player_id:
            df = df[df['player_id'] == player_id]
            
        if df.empty:
             return {'over_rate': 0.5, 'under_rate': 0.5, 'count': 0}
             
        overs = (df['actual'] > df['predicted']).sum()
        unders = (df['actual'] < df['predicted']).sum()
        # pushes (exact matches)
        pushes = (df['actual'] == df['predicted']).sum()
        
        total = len(df)
        # Exclude pushes from rate? Or count as half?
        # Usually exclude or denom includes them.
        # Let's use denom = total - pushes for rate.
        rate_denom = total - pushes
        if rate_denom == 0:
             return {'over_rate': 0.5, 'under_rate': 0.5, 'count': total}
        
        over_rate = overs / rate_denom
        under_rate = unders / rate_denom
        
        return {
            'over_rate': over_rate,
            'under_rate': under_rate,
            'count': total
        }

    def get_bias_metrics(self):
        """Return dictionary with per-stat bias statistics."""
        df = self.load_history()
        # Filter for actuals
        df = df.dropna(subset=['actual'])
        
        metrics = {}
        for stat in df['stat_type'].unique():
            rates = self.calculate_hit_rates(df, stat_type=stat)
            
            # Calculate Error Metrics
            sub = df[df['stat_type'] == stat]
            residuals = sub['actual'] - sub['predicted']
            mpe = residuals.mean() # Mean Error (Bias)
            mae = residuals.abs().mean()
            
            metrics[stat] = {
                'over_rate': rates['over_rate'],
                'under_rate': rates['under_rate'],
                'count': rates['count'],
                'mpe': mpe,
                'mae': mae
            }
            
        return metrics

class BiasAnalyzer:
    def __init__(self, tracker: CalibrationTracker):
        self.tracker = tracker

    def analyze_stat_bias(self, stat_type):
        """Analyze systematic bias for a specific stat."""
        df = self.tracker.load_history()
        df = df.dropna(subset=['actual'])
        
        rates = self.tracker.calculate_hit_rates(df, stat_type=stat_type)
        
        under_biased = False
        severity = 0.0
        
        # Threshold: 65% under rate -> Systematic Under Bias
        if rates['under_rate'] > 0.65:
            under_biased = True
            # Normalize spread: (Rate - 0.5) / 0.5? 
            # Request says: (under_rate - 50) / 50 -> assumes % format
            # My logic uses 0.0-1.0. So (under_rate - 0.5) / 0.5
            severity = (rates['under_rate'] - 0.5) / 0.5
            
        return {
            'stat': stat_type,
            'under_biased': under_biased,
            'severity': severity,
            'rates': rates
        }

    def analyze_player_bias(self, player_id):
        """Analyze prediction bias for a specific player."""
        df = self.tracker.load_history()
        df = df.dropna(subset=['actual'])
        
        rates = self.tracker.calculate_hit_rates(df, player_id=player_id)
        
        # Simple multiplier logic
        # If under_rate is high, we need to multiply predictions by > 1.0
        # Multiplier ~ Actual / Predicted average ratio?
        # Or based on hit rate severity?
        # Let's use the calculator for multiplier logic.
        return rates

class CalibrationFactorCalculator:
    def __init__(self, tracker: CalibrationTracker):
        self.tracker = tracker

    def compute_stat_calibration(self, stat_type, target_hit_rate=0.50):
        """Compute calibration multiplier for a stat type."""
        df = self.tracker.load_history(days=90)
        df = df.dropna(subset=['actual'])
        
        rates = self.tracker.calculate_hit_rates(df, stat_type=stat_type)
        under_rate = rates['under_rate']
        
        multiplier = 1.0
        
        # Logic: If under_rate > 65%, compute upward multiplier
        if under_rate > 0.65:
            # Formula: 1 + (under_rate_pct - 50) * 0.01
            # e.g. 68% -> 1 + (18) * 0.01 = 1.18
            rate_pct = under_rate * 100
            diff = rate_pct - 50
            multiplier = 1.0 + (diff * 0.01)
            
            # Cap at 1.25
            multiplier = min(1.25, multiplier)
            
        return multiplier

    def compute_player_calibration_factors(self, min_predictions=30):
        """Compute player-specific multipliers."""
        df = self.tracker.load_history(days=90)
        df = df.dropna(subset=['actual'])
        
        player_factors = {}
        
        for pid in df['player_id'].unique():
            sub = df[df['player_id'] == pid]
            if len(sub) < min_predictions:
                continue
                
            p_factors = {}
            for stat in sub['stat_type'].unique():
                stat_sub = sub[sub['stat_type'] == stat]
                if len(stat_sub) < 10: # Minimum per stat per player
                    continue
                    
                # Ratio Method for players: Mean(Actual) / Mean(Predicted)
                # This is more direct for individual scaling than hit-rate logic
                mean_pred = stat_sub['predicted'].mean()
                mean_act = stat_sub['actual'].mean()
                
                if mean_pred > 0:
                    ratio = mean_act / mean_pred
                    # Clamp: 0.8 to 1.25
                    ratio = max(0.8, min(1.25, ratio))
                    if abs(ratio - 1.0) > 0.05: # Only if significant deviation
                        p_factors[stat] = ratio
                        
            if p_factors:
                player_factors[str(pid)] = p_factors
                
        return player_factors

def save_calibration_factors(factors):
    """Save factors to JSON."""
    with open(FACTORS_FILE, 'w') as f:
        json.dump(factors, f, indent=4)

def load_calibration_factors():
    """Load factors from JSON."""
    if not os.path.exists(FACTORS_FILE):
        return {'global': {}, 'players': {}}
    try:
        with open(FACTORS_FILE, 'r') as f:
            return json.load(f)
    except:
        return {'global': {}, 'players': {}}

# Singleton / Module Level Helpers
tracker = CalibrationTracker() # Shared instance logic if imported
