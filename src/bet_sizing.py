import numpy as np

def calculate_bet_quality(
    prop_name: str, 
    val: float, 
    mae: float, 
    line: float,
    recent_stats: dict = None
) -> dict:
    """
    Calculates the quality of a bet, returning units, badge, and consistency score.
    Logic extracted from batch_predict.py for reuse.
    """
    
    # Defaults
    if recent_stats is None:
        recent_stats = {}
    
    # 1. Edge Calculation
    # Edge = (Prediction - Line) for Over? 
    # Not exactly. We use simple MAE logic mainly, but let's formalize.
    # If val > line (Over), we want val - line > mae ideally.
    
    # For unit sizing, we primarily look at Signal-to-Noise (val vs MAE) 
    # AND Consistency.
    
    # Confidence Calculation (Inverse CV equivalent)
    # Higher is better.
    if val < 0.1: 
        cv_score = 0
    else:
        cv_score = val / (mae + 0.1)
    
    # 2. Consistency Score
    std_dev = recent_stats.get('std_pra', 10.0)
    avg_l5 = recent_stats.get('avg_pra', 1.0)
    
    consistency_score = 1.0
    if avg_l5 > 0:
        cv_val = std_dev / avg_l5
        # Map CV 0.1 -> 1.0, CV 0.5 -> 0.0
        consistency_score = max(0.0, 1.0 - (cv_val * 2))
        
    # 3. Scientific Unit Sizing (Kelly w/ heuristics)
    base_prob = 0.51
    
    consistency_boost = consistency_score * 0.15
    model_boost = min(0.08, cv_score / 50.0) 
    
    # Penalize high variance
    if std_dev > avg_l5 * 0.3:
        consistency_boost *= 0.5
        
    win_prob = base_prob + consistency_boost + model_boost
    win_prob = min(0.70, win_prob)
    
    # Kelly
    b = 0.91 # -110 odds
    q = 1 - win_prob
    f_star = ((b * win_prob) - q) / b
    
    kelly_fraction = 0.125
    recommended_bankroll_pct = max(0.0, f_star * kelly_fraction)
    
    raw_units = recommended_bankroll_pct * 100.0
    units = round(raw_units * 4) / 4 # Round to 0.25
    
    # Gates
    if consistency_score < 0.4:
        units = min(units, 0.25)
    elif consistency_score < 0.6:
        units = min(units, 0.50)
    elif consistency_score < 0.75:
        units = min(units, 0.75)
        
    if units > 1.0: units = 1.0
    
    # Badge
    badge = "LEAN"
    if units >= 0.5: badge = "STANDARD"
    if units >= 0.75: badge = "GOLD 🥇"
    if units >= 1.0: badge = "DIAMOND 💎"
    
    if units < 0.25:
        if f_star > 0: units = 0.25
        else: units = 0.0
        badge = "NO BET"
        
    # Recalculate Safe Lines for display
    line_low = val - mae
    line_high = val + mae
    
    return {
        'prop': prop_name,
        'val': val,
        'line_low': line_low,
        'line_high': line_high,
        'units': units,
        'badge': badge,
        'consistency': consistency_score,
        'mae': mae,
        'win_prob': win_prob
    }
