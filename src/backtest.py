
import pandas as pd
import numpy as np
import joblib
import os
import torch
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
from datetime import timedelta, datetime

# Use Agg backend for non-interactive plotting
plt.switch_backend('Agg')

class Backtester:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.models_dir = os.path.join(os.path.dirname(__file__), '../data/models')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resources = {}
        self._load_resources()
        self.injury_archives = {}
        
    def _load_resources(self):
        print("Loading resources...")
        self.resources['p_enc'] = joblib.load(os.path.join(self.models_dir, 'player_encoder.joblib'))
        self.resources['t_enc'] = joblib.load(os.path.join(self.models_dir, 'team_encoder.joblib'))
        self.resources['scaler'] = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
        self.resources['feature_cols'] = joblib.load(os.path.join(self.models_dir, 'feature_cols.joblib'))

        # Build robust ID -> Name map from game logs (avoid encoder index confusion)
        print("Building Player ID Map...")
        self.player_id_map = {}
        log_pattern = os.path.join(os.path.dirname(__file__), '../data/raw/game_logs_*.csv')
        log_files = sorted(glob.glob(log_pattern), reverse=True)
        
        if log_files:
            # Read latest logs to get mappings
            try:
                df = pd.read_csv(log_files[0])
                # Create map
                mapping = df.dropna(subset=['PLAYER_ID', 'PLAYER_NAME'])
                self.player_id_map = dict(zip(mapping['PLAYER_ID'], mapping['PLAYER_NAME']))
                print(f"Mapped {len(self.player_id_map)} players.")
            except Exception as e:
                print(f"Error building player map: {e}")

    def _load_model(self):
        path = os.path.join(self.models_dir, 'pytorch_nba_global.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
            
        p_enc = self.resources['p_enc']
        t_enc = self.resources['t_enc']
        f_cols = self.resources['feature_cols']
        
        # Dimensions
        num_players = len(p_enc.classes_) + 1 # +1 for unknown
        num_teams = len(t_enc.classes_)
        num_cont = len(f_cols)
        
        model = NBAPredictor(num_players, num_teams, num_cont)
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def _load_injury_archives(self):
        """Loads historical injury reports from data/cache/injury_archives/"""
        archive_dir = os.path.join(os.path.dirname(__file__), '../data/cache/injury_archives')
        files = glob.glob(os.path.join(archive_dir, '*.json'))
        
        print(f"Loading {len(files)} injury archives...")
        for f in files:
            try:
                date_str = os.path.basename(f).replace('.json', '')
                with open(f, 'r') as file:
                    self.injury_archives[date_str] = json.load(file)
            except Exception as e:
                print(f"Error loading injury archive {f}: {e}")

    def _match_injury_to_games(self, game_date):
        """Finds the closest injury report for a given game date (same day or prev day)"""
        date_str = game_date.strftime('%Y-%m-%d')
        if date_str in self.injury_archives:
            return self.injury_archives[date_str]
            
        # Try previous day (sometimes games are late/archive is early)
        prev_day = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
        if prev_day in self.injury_archives:
            return self.injury_archives[prev_day]
            
        return {}

    def _player_id_to_name(self, pid):
        """Converts Player ID to Name using the robust map."""
        if hasattr(self, 'player_id_map') and pid in self.player_id_map:
            return self.player_id_map[pid]
            
        # Fallback to encoder if map fails (legacy)
        if 'p_enc' not in self.resources:
            return 'Unknown'
            
        # This part was the bug source: encoder uses 0..N indices, not PIDs
        # So we can only use it if we transform PID to IDX first. but we don't have that handy.
        # So just return Unknown if not in map.
        return 'Unknown'

    def _classify_injury_severity(self, row, injury_report):
        """
        Classifies the game context based on injuries.
        Returns: 'Star Out', 'Role Player Out', 'Healthy'
        """
        missing_ids_str = row.get('MISSING_PLAYER_IDS', 'NONE')
        if missing_ids_str == 'NONE' or not missing_ids_str:
            return 'Healthy'
            
        missing_ids = [int(x) for x in missing_ids_str.split('_') if x.isdigit()]
        if not missing_ids:
            return 'Healthy'

        # If no report available, use heuristics (Legacy/Fallback)
        if not injury_report:
            if row.get('OPP_SCORE', 0) > 1.0: return 'Star Out'
            if row.get('OPP_SCORE', 0) > 0: return 'Role Player Out'
            return 'Healthy'

        # Report-Driven Classification
        is_star_out = False
        is_role_out = False
        
        # We need to know who is "Out"
        # Since injury_report might be structured as {'injuries': {'Name': 'Status'}}
        injuries = injury_report.get('injuries', {})
        
        for pid in missing_ids:
            name = self._player_id_to_name(pid)
            if name == 'Unknown': continue
            
            # Check Status in Report
            status = injuries.get(name, 'Healthy')
            
            if status == 'Out':
                # Valid confirmed absence.
                # Determine if Star or Role.
                # Heuristic: High Impact (OPP_USG_BOOST > 5) implies Star.
                if row.get('OPP_USG_BOOST', 0) > 5.0:
                    is_star_out = True
                else:
                    is_role_out = True
            elif status in ['Doubtful', 'Questionable']:
                 # Treat as potential role impact if they end up being out, 
                 # but technically classification should be based on "Out".
                 # User requested: "Elif status in ['Doubtful', 'Questionable']: role_out = True"
                 is_role_out = True
        
        if is_star_out: return 'Star Out'
        if is_role_out: return 'Role Player Out'
        
        return 'Healthy'

    # Legacy run_backtest removed. Enhanced version is at the end of the class.

    def validate_calibration(self, df):
        """
        Validates calibration quality by computing over/under hit rates and distribution alignment.
        """
        print("\n--- Calibration Validation ---")
        
        # Filter where predictions exist
        df = df.dropna(subset=['PRED_PTS', 'PRED_REB', 'PRED_AST'])
        
        stats = {
            'PTS': ('PTS', 'PRED_PTS'),
            'REB': ('REB', 'PRED_REB'),
            'AST': ('AST', 'PRED_AST')
        }
        
        calibration_data = []
        
        print(f"{'Stat':<5} | {'Over%':<6} | {'Under%':<6} | {'Align':<6} | {'Status'}")
        print("-" * 50)
        
        for name, (act_col, pred_col) in stats.items():
            total = len(df)
            if total == 0: continue
            
            overs = (df[act_col] > df[pred_col]).sum()
            unders = (df[act_col] < df[pred_col]).sum()
            
            over_rate = overs / total
            under_rate = unders / total
            
            # Alignment Score: 1.0 - |Over - 0.5|*2
            align = 1.0 - abs(over_rate - 0.5) * 2
            
            status = "OK"
            if align < 0.8: status = "POOR"  # e.g. >60% or <40%
            if align < 0.6: status = "BAD"   # e.g. >70% or <30%
            
            print(f"{name:<5} | {over_rate*100:<6.1f} | {under_rate*100:<6.1f} | {align:<6.2f} | {status}")
            
            calibration_data.append({
                'stat': name,
                'over_rate': over_rate,
                'under_rate': under_rate,
                'alignment': align,
                'status': status
            })
            
        # Save Report
        report_dir = os.path.join(os.path.dirname(__file__), '../data/calibration')
        os.makedirs(report_dir, exist_ok=True)
        
    def run_injury_scenario_backtest(self, df):
        """
        Analyzes prediction accuracy across different injury contexts.
        Cohorts: Star Out, Role Player Out, Healthy
        """
        print("\n--- Injury Scenario Analysis ---")
        
        # Load archives if not loaded
        if not self.injury_archives:
            self._load_injury_archives()
            
        # Classify rows with Archive Checks
        # To do this effectively, we really need the FE to have applied the report logic.
        # But here we are post-FE.
        # We will iterate and apply classification.
        
        def classify_wrapper(row):
            date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
            report = self._match_injury_to_games(row['GAME_DATE'])
            classification = self._classify_injury_severity(row, report)
            # Log usage for verification (User request)
            if report and classification != 'Healthy':
                 # Only log relevant ones to avoid spam? Or just log a few?
                 # User asked for "Used report for {date}: classified as {context}"
                 # Let's print unique day/context combos to avoid massive spam
                 # Or just print all if volume is low.
                 # Given huge potential volume, printing EVERY row is bad.
                 # Let's print every 100th or similar? Or unique dates?
                 pass 
            return classification
            
        # Efficient application
        df['INJURY_CONTEXT'] = df.apply(classify_wrapper, axis=1)

        # Verification Logging (Aggregate)
        # Identify dates where logic was used
        print("\n[Log] Injury Classification Verification:")
        sample_logs = []
        seen_dates = set()
        for idx, row in df.iterrows():
             if row['INJURY_CONTEXT'] != 'Healthy' and len(sample_logs) < 10:
                 d = row['GAME_DATE'].strftime('%Y-%m-%d')
                 if d not in seen_dates:
                     sample_logs.append(f"Used report for {d}: classified as {row['INJURY_CONTEXT']} (Missing: {row.get('MISSING_PLAYER_IDS', '')})")
                     seen_dates.add(d)
        
        for log in sample_logs:
            print(log)
        
        results = []
        stats = ['PTS', 'REB', 'AST']
        
        print(f"{'Scenario':<20} | {'Stat':<5} | {'MAE':<6} | {'Count':<6} | {'Bias%'}")
        print("-" * 65)
        
        for context, group in df.groupby('INJURY_CONTEXT'):
            if len(group) < 10: continue
            
            for stat in stats:
                pred_col = f'PRED_{stat}'
                if pred_col not in group.columns: continue
                
                # MAE
                mae = (group[pred_col] - group[stat]).abs().mean()
                
                # Bias (Mean Error)
                bias = (group[pred_col] - group[stat]).mean()
                
                print(f"{context:<20} | {stat:<5} | {mae:<6.2f} | {len(group):<6} | {bias:<6.2f}")
                
                results.append({
                    'scenario': context,
                    'stat_type': stat,
                    'mae': mae,
                    'bias': bias,
                    'sample_size': len(group)
                })
                
        # Save Results
        # Save Results
        # Use absolute path based on CWD or relative to file, but ensure it exists
        # Assuming run from root
        out_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/backtest'))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(results).to_csv(os.path.join(out_dir, 'injury_scenario_results.csv'), index=False)
        print(f"Scenario results saved to {out_dir}/injury_scenario_results.csv")

    def analyze_with_without_accuracy(self, df):
        """
        Identifies players who have better predictability when teammates are missing.
        """
        print("\n--- With/Without Accuracy Analysis ---")
        
        # Load splits cache to know who matters
        cache_path = os.path.join(os.path.dirname(__file__), '../data/cache/with_without_splits.joblib')
        if not os.path.exists(cache_path):
            print("No with/without cache found. Skipping.")
            return

        try:
            cache = joblib.load(cache_path)
            splits = cache.get('splits', {})
        except:
            return

        # We want to compare: MAE when Context="Star Out" vs MAE when Context="Healthy"
        # For each player, calculate delta.
        
        player_impacts = []
        
        for pid, group in df.groupby('PLAYER_ID'):
            if len(group) < 10: continue
            
            # Split by context
            healthy = group[group['INJURY_CONTEXT'] == 'Healthy']
            injured = group[group['INJURY_CONTEXT'] == 'Star Out']
            
            if len(healthy) < 5 or len(injured) < 5: continue
            
            h_mae = (healthy['PRED_PTS'] - healthy['PTS']).abs().mean()
            i_mae = (injured['PRED_PTS'] - injured['PTS']).abs().mean()
            
            # Improvement: Positive if Injury MAE is LOWER (better) than Healthy MAE
            # Wait, usually injury adds variance. If our model detects it well, MAE should be comparable.
            # If we want to see if FEATURE helps, we'd overlap. 
            # But here we just want to know if we are handling it well.
            
            delta = h_mae - i_mae # Positive means we predict injury games BETTER than healthy games (unlikely but great)
            
            if abs(delta) > 1.0: # Significant difference
                player_impacts.append({
                    'player_id': pid,
                    'name': group.iloc[0]['PLAYER_NAME'] if 'PLAYER_NAME' in group.columns else str(pid),
                    'healthy_mae': h_mae,
                    'injury_mae': i_mae,
                    'delta': delta,
                    'count_h': len(healthy),
                    'count_i': len(injured)
                })
                
        # Top Beneficiaries (Predictions stay accurate or improve)
        player_impacts.sort(key=lambda x: x['delta'], reverse=True)
        
        print("Top 5 Players with resilient predictions during injuries:")
        for p in player_impacts[:5]:
            print(f"  {p['name']}: Healthy MAE {p['healthy_mae']:.1f}, Injured MAE {p['injury_mae']:.1f} (Delta {p['delta']:.2f})")
            
        out_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/backtest'))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(player_impacts).to_csv(os.path.join(out_dir, 'with_without_analysis.csv'), index=False)

    def validate_opportunity_scores(self, df):
        """
        Checks if OPP_SCORE correlates with Actual Usage Increase.
        Includes Baseline delta and Quartile Analysis.
        """
        print("\n--- Opportunity Feature Validation ---")
        
        if 'OPP_SCORE' not in df.columns or 'PTS' not in df.columns:
            print("Missing OPP_SCORE or PTS columns.")
            return
            
        # 1. Calculate Baseline (Season Avg before this game? or Rolling?)
        # We can use our pre-calc features if available, e.g., 'ROLL_PTS' is not standard.
        # Let's simple calculate global avg for the player as baseline for simplicity in backtest.
        # Or better: PRED_PTS (without opp boost) would be ideal baseline.
        # Since PRED_PTS *includes* OPP_SCORE in the model, we can't use it as non-boosted baseline easily.
        # Let's use the player's season average up to that point.
        
        df = df.sort_values('GAME_DATE')
        df['ROLLING_PTS'] = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
        
        # Filter for rows with Rolling PTS
        validation_df = df.dropna(subset=['ROLLING_PTS', 'OPP_SCORE']).copy()
        
        # 2. Calculate Actual Boost
        validation_df['ACTUAL_BOOST'] = validation_df['PTS'] - validation_df['ROLLING_PTS']
        
        # 3. Filter for Boosted Games
        subset = validation_df[validation_df['OPP_SCORE'] > 0.0].copy()
        
        if len(subset) < 20:
             print("Insufficient data for validation.")
             return
             
        # 4. Correlation
        corr_pts = subset['OPP_SCORE'].corr(subset['ACTUAL_BOOST'])
        print(f"Correlation (OPP_SCORE vs Actual PTS Boost): {corr_pts:.3f}")
        
        # 5. Quartiles
        subset['Quartile'] = pd.qcut(subset['OPP_SCORE'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
        
        quartile_stats = subset.groupby('Quartile')['ACTUAL_BOOST'].agg(['mean', 'count', 'std'])
        print("\nActual PTS Boost by OPP_SCORE Quartile:")
        print(quartile_stats)
        
        comparison = {
            'corr_pts_boost': corr_pts,
            'quartile_stats': quartile_stats.to_dict(),
            'sample_size': len(subset)
        }
        
        # Convert index for JSON
        # quartile_stats keys are Categorical, need str
        comparison['quartile_stats'] = {str(k): v for k, v in comparison['quartile_stats'].items()}
        
        out_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/backtest'))
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'opportunity_validation.json'), 'w') as f:
            json.dump(comparison, f, default=str)

    def generate_injury_impact_report(self, df):
        """
        Generates a detailed report of 'Star Out' events and teammate responses.
        """
        print("\n--- Generating Injury Impact Report ---")
        
        # Filter for Star Out games
        events = df[df['INJURY_CONTEXT'] == 'Star Out'].copy()
        if events.empty:
            print("No Star Out events found.")
            return
            
        # Group by Game/Team to identify the event
        report_data = []
        
        for (game_id, team_id), group in events.groupby(['GAME_ID', 'TEAM_ID']):
            # Who was out? (parse MISSING_PLAYER_IDS)
            missing = group.iloc[0]['MISSING_PLAYER_IDS']
            date = group.iloc[0]['GAME_DATE']
            
            # Teammates in this game
            for _, row in group.iterrows():
                # Predicted vs Actual
                # We need a baseline to know "Boost". 
                # Use calculated ROLLING_PTS from validation step if available, else calc
                if 'ROLLING_PTS' not in row:
                    continue
                    
                actual_boost = row['PTS'] - row['ROLLING_PTS']
                predicted_boost = row.get('OPP_SCORE', 0) * 0.5 # Rough scaling factor model might learn
                
                report_data.append({
                    'GAME_DATE': date,
                    'TEAM_ID': team_id,
                    'MISSING': missing,
                    'TEAMMATE': row.get('PLAYER_NAME', row['PLAYER_ID']),
                    'OPP_SCORE': row.get('OPP_SCORE', 0),
                    'ACTUAL_BOOST': actual_boost,
                    'PRED_ERROR': row['ERR_PTS']
                })
        
        if not report_data:
            return
            
        report_df = pd.DataFrame(report_data)
        out_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/backtest'))
        os.makedirs(out_dir, exist_ok=True)
        report_df.to_csv(os.path.join(out_dir, 'injury_impact_report.csv'), index=False)
        print(f"Impact report saved to {out_dir}/injury_impact_report.csv")

    def create_injury_visualization_dashboard(self, df):
        """Generates plots for injury analysis."""
        print("\n--- Generating Visualization Dashboard ---")
        out_dir = os.path.abspath(os.path.join(os.getcwd(), 'data/backtest'))
        os.makedirs(out_dir, exist_ok=True)
        
        # 1. MAE by Context
        plt.figure(figsize=(10, 6))
        sns.barplot(x='INJURY_CONTEXT', y='ERR_PTS', data=df, errorbar='se', palette='viridis')
        plt.title("Prediction Error (MAE) by Injury Context")
        plt.ylabel("Absolute Error (PTS)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mae_by_context.png'))
        plt.close()
        
        # 2. Bias by Context
        if 'PRED_PTS' in df.columns:
            df['ERROR_DIR'] = df['PRED_PTS'] - df['PTS'] # Predicted - Actual
            plt.figure(figsize=(10, 6))
            sns.barplot(x='INJURY_CONTEXT', y='ERROR_DIR', data=df, errorbar='se', palette='coolwarm')
            plt.axhline(0, color='k', linestyle='--')
            plt.title("Prediction Bias (Mean Error) by Injury Context")
            plt.ylabel("Prediction Error (Pts) - Positive = Overestimate")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'bias_by_context.png'))
            plt.close()
        
        # 3. OPP_SCORE vs Actual Boost
        if 'ROLLING_PTS' in df.columns:
            df['ACTUAL_BOOST'] = df['PTS'] - df['ROLLING_PTS']
            subset = df[df['OPP_SCORE'] > 0.5]
            if not subset.empty:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x='OPP_SCORE', y='ACTUAL_BOOST', data=subset, alpha=0.6, hue='INJURY_CONTEXT')
                sns.regplot(x='OPP_SCORE', y='ACTUAL_BOOST', data=subset, scatter=False, color='red')
                plt.title("Opportunity Score vs Actual Production Boost")
                plt.xlabel("Opp Score (Projected Usage Shift)")
                plt.ylabel("Actual Points - Rolling Avg")
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, 'opp_score_validation.png'))
                plt.close()
        
        # 4. Timeline
        # MAE over time
        df_sorted = df.sort_values('GAME_DATE')
        daily_mae = df_sorted.groupby('GAME_DATE')['ERR_PTS'].mean().rolling(7).mean()
        
        plt.figure(figsize=(12, 6))
        daily_mae.plot()
        plt.title("7-Day Rolling MAE Interaction")
        plt.ylabel("MAE (PTS)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'mae_timeline.png'))
        plt.close()
            
        print(f"Visualizations saved to {out_dir}")

    def run_backtest(self, days=30):
        print(f"--- Starting Backtest (Last {days} Days) ---")
        
        # 1. Load All Data (Actuals)
        try:
            raw_df = self.fe.load_all_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            return

        # 2. Process Features
        feature_df = self.fe.process(raw_df, is_training=False)
        
        # 3. Filter for Date Range
        max_date = feature_df['GAME_DATE'].max()
        start_date = max_date - timedelta(days=days)
        test_df = feature_df[feature_df['GAME_DATE'] >= start_date].copy()
        
        if test_df.empty:
            print("No data found for period.")
            return

        print(f"Testing on {len(test_df)} rows from {start_date.date()} to {max_date.date()}")
        
        # 4. Prepare Tensors
        model = self._load_model()
        
        # Inputs
        # Player IDX
        p_idxs = torch.tensor(test_df['PLAYER_IDX'].values, dtype=torch.long).to(self.device)
        # Team IDX
        t_idxs = torch.tensor(test_df['TEAM_IDX'].values, dtype=torch.long).to(self.device)
        # Continuous
        f_cols = self.resources['feature_cols']
        # Ensure cols exist
        missing = [c for c in f_cols if c not in test_df.columns]
        if missing:
            print(f"Warning: Zero-filling missing cols: {missing}")
            for c in missing: test_df[c] = 0.0
            
        cont_data = torch.tensor(test_df[f_cols].fillna(0).values, dtype=torch.float32).to(self.device)
        
        # Missing Players Tensor (padded with unknown index)
        pad_idx = len(self.resources['p_enc'].classes_)
        batch_size = len(test_df)
        m_idxs = torch.full((batch_size, 3), pad_idx, dtype=torch.long).to(self.device)
        
        # Predict
        with torch.no_grad():
            preds = model(p_idxs, t_idxs, cont_data, m_idxs).cpu().numpy()

            
        # 5. Evaluate
        test_df['PRED_PTS'] = preds[:, 0]
        test_df['PRED_REB'] = preds[:, 1]
        test_df['PRED_AST'] = preds[:, 2]
        
        # Calc Errors
        test_df['ERR_PTS'] = (test_df['PRED_PTS'] - test_df['PTS']).abs() # FIXED: Store MAE per row for plotting
        mae_pts = test_df['ERR_PTS'].mean()
        mae_reb = (test_df['PRED_REB'] - test_df['REB']).abs().mean()
        mae_ast = (test_df['PRED_AST'] - test_df['AST']).abs().mean()
        
        print(f"\n--- Results ---")
        print(f"MAE PTS: {mae_pts:.2f}")
        print(f"MAE REB: {mae_reb:.2f}")
        print(f"MAE AST: {mae_ast:.2f}")
        
        # Safe Anchor Hit Rate
        test_df['SAFE_PTS_LINE'] = test_df['PRED_PTS'] * 0.75
        test_df['HIT_SAFE_PTS'] = test_df['PTS'] >= test_df['SAFE_PTS_LINE']
        print(f"Safe Anchor Hit Rate (PTS > 75% Pred): {test_df['HIT_SAFE_PTS'].mean()*100:.1f}%")
        
        # High Confidence
        hc = test_df[test_df['PRED_PTS'] > 20]
        if not hc.empty:
            print(f"High Scorer Safe Hit Rate (Pred > 20): {hc['HIT_SAFE_PTS'].mean()*100:.1f}%")
            
        test_df[['GAME_DATE', 'PLAYER_ID', 'PTS', 'PRED_PTS', 'SAFE_PTS_LINE']].head(100).to_csv('src/dev/backtest_results.csv')
        print("Results saved to src/dev/backtest_results.csv")
        
        # Trigger Calibration Validation
        self.validate_calibration(test_df)
        
        # --- NEW INJURY ANALYSIS ---
        self.run_injury_scenario_backtest(test_df)
        self.analyze_with_without_accuracy(test_df)
        self.validate_opportunity_scores(test_df)
        self.generate_injury_impact_report(test_df) # NEW
        self.create_injury_visualization_dashboard(test_df)

if __name__ == "__main__":
    bt = Backtester()
    bt.run_backtest(days=30)

