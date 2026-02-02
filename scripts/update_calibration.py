
import sys
import os
import argparse
from datetime import datetime

# Setup Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.calibration_tracker import CalibrationTracker, CalibrationFactorCalculator, save_calibration_factors, BiasAnalyzer

def update_calibration(days=90, min_predictions=50, target_hit_rate=0.50):
    print("Starting Calibration Update...")
    tracker = CalibrationTracker()
    bias_analyzer = BiasAnalyzer(tracker)
    calculator = CalibrationFactorCalculator(tracker)
    
    # 1. Global Factors
    stats = ['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL']
    global_factors = {}
    
    print("\nGlobal Bias Analysis:")
    print(f"{'Stat':<5} | {'Under%':<6} | {'Factor':<6}")
    print("-" * 25)
    
    for stat in stats:
        # Analyze
        analysis = bias_analyzer.analyze_stat_bias(stat)
        
        # Compute Factor
        factor = calculator.compute_stat_calibration(stat)
        
        global_factors[stat] = factor
        
        under_pct = analysis['rates']['under_rate'] * 100
        print(f"{stat:<5} | {under_pct:<6.1f} | {factor:<6.3f}")
        
    # 2. Player Factors
    print("\nComputing Player Factors...")
    player_factors = calculator.compute_player_calibration_factors(min_predictions=min_predictions)
    print(f"Computed specific factors for {len(player_factors)} players.")
    
    # 3. Save
    all_factors = {
        'meta': {
            'updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'history_days': days
        },
        'global': global_factors,
        'players': player_factors
    }
    
    save_calibration_factors(all_factors)
    print("\nCalibration factors updated and saved.")
    
    # 4. Generate Report
    report_dir = os.path.dirname(tracker.history_file)
    report_path = os.path.join(report_dir, 'calibration_report.md')
    
    # Use helper method or manual write
    # Assuming manual write for simplicity here
    with open(os.path.abspath(report_path), 'w') as f:
        f.write("# Calibration Report\n")
        f.write(f"Updated: {datetime.now()}\n\n")
        f.write("## Global Factors\n")
        f.write("| Stat | Factor | Under Bias % |\n")
        f.write("|---|---|---|\n")
        for stat, factor in global_factors.items():
            ana = bias_analyzer.analyze_stat_bias(stat)
            f.write(f"| {stat} | {factor:.3f} | {ana['rates']['under_rate']*100:.1f}% |\n")
            
        f.write(f"\n\n## Player Corrections: {len(player_factors)} players calibrated.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=90)
    parser.add_argument('--min-predictions', type=int, default=30)
    args = parser.parse_args()
    
    update_calibration(days=args.days, min_predictions=args.min_predictions)
