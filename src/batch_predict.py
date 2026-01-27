import pandas as pd
import numpy as np
import os
import torch
import joblib
from datetime import datetime
from src.data_fetch import fetch_daily_scoreboard
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
from src.sgp_model import SGPMetaModel
from src.odds_service import fetch_nba_odds, get_ev_for_prediction, get_game_total_for_matchup
from nba_api.stats.static import teams
import asyncio

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchPredictor:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.resources = {} # Cache resources
        self.models_cache = {} # Cache loaded PyTorch models

    def load_common_resources(self):
        print("Loading common resources...")
        self.resources['p_enc'] = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        self.resources['t_enc'] = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        self.resources['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        self.resources['feature_cols'] = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Load SGP Meta Model
        try:
            sgp_path = os.path.join(MODELS_DIR, 'sgp_meta_model.pth')
            scaler_path = os.path.join(MODELS_DIR, 'sgp_scaler.joblib')
            if os.path.exists(sgp_path) and os.path.exists(scaler_path):
                self.sgp_model = SGPMetaModel(input_dim=12).to(device) # 12 Features (6 Preds + 6 Stds)
                self.sgp_model.load_state_dict(torch.load(sgp_path, map_location=device))
                self.sgp_model.eval()
                self.sgp_scaler = joblib.load(scaler_path)
                self.sgp_loaded = True
                print("SGP Meta Model Loaded.")
            else:
                self.sgp_loaded = False
        except Exception as e:
            print(f"Failed to load SGP Model: {e}")
            self.sgp_loaded = False
        
    def _get_model(self, player_id: int):
        # Determine model path
        specific_path = os.path.join(MODELS_DIR, f'pytorch_player_{player_id}.pth')
        generic_path = os.path.join(MODELS_DIR, 'pytorch_nba_global_vNext.pth')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
        # Check cache
        if path_to_load in self.models_cache:
            return self.models_cache[path_to_load]
            
        # Load
        p_enc = self.resources['p_enc']
        t_enc = self.resources['t_enc']
        f_cols = self.resources['feature_cols']
        
        num_players = len(p_enc.classes_) + 1
        num_teams = len(t_enc.classes_)
        num_cont = len(f_cols)
        
        # Init Model (New Arch: 6 targets)
        model = NBAPredictor(num_players, num_teams, num_cont)
        
        # Safely load weights
        try:
            model.load_state_dict(torch.load(path_to_load, map_location=device))
            # print(f"[Model] Loaded model from {os.path.basename(path_to_load)}")
        except Exception as e:
            print(f"Error loading model {path_to_load}: {e}")
            # Fallback to generic if specific failed?
            if path_to_load != generic_path and os.path.exists(generic_path):
                 try:
                    model.load_state_dict(torch.load(generic_path, map_location=device))
                 except: pass
            
        model.to(device)
        model.eval()
        
        self.models_cache[path_to_load] = model
        return model

    def _get_metrics(self, player_id: int):
        """Loads MAE metrics for confidence intervals."""
        specific_path = os.path.join(MODELS_DIR, f'metrics_player_{player_id}.json')
        generic_path = os.path.join(MODELS_DIR, 'metrics_global.json')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
        # Default fallback
        metrics = {
            'mae_pts': 6.0, 'mae_reb': 2.5, 'mae_ast': 2.0,
            'baseline_pts': 7.0, 'baseline_reb': 3.0, 'baseline_ast': 2.5
        }
        
        try:
            import json
            if os.path.exists(path_to_load):
                with open(path_to_load, 'r') as f:
                    metrics = json.load(f)
        except:
             pass
             
        return metrics

    async def analyze_today_batch(self, date_input: str = None, execution_list: list = None, log_manager=None):
        """
        Main Batch Function.
        1. Loads History ONCE.
        2. Appends 'Phantom Rows' for ALL players in execution_list.
        3. Processes Features ONCE.
        4. Predicts in loop.
        """
        target_date = date_input if date_input else datetime.now().strftime('%Y-%m-%d')
        
        if log_manager: await log_manager.broadcast("Loading historical data from disk...")
        
        # 1. Load All Data (Heavy IO - done once)
        df_history = await asyncio.to_thread(self.fe.load_all_data)
        
        if log_manager: await log_manager.broadcast("Generating Prediction Rows...")
        
        # 2. Create Phantom Rows
        phantom_rows = []
        
        # Fetch team info cache
        team_abbr_cache = {}
        
        for item in execution_list:
            pid = item['pid']
            pname = item['pname']
            tid = item['team_id']
            # We need opponent ID. 
            # execution_list came from api.py which scanned the scoreboard.
            # But api.py only sent pid/tid. We need the GAME info (Opponent).
            # Let's pass the Full execution Item which might need to include Opponent Info.
            # OR we re-fetch scoreboard here? 
            # API.py has the scoreboard logic. Let's make API pass 'opp_id' and 'is_home' in execution_list.
            # Assuming api.py is updated to pass 'opp_id' and 'is_home'.
            
            # Fallback if keys missing (for safety during refactor)
            if 'opp_id' not in item:
                 continue
                 
            opp_id = item['opp_id']
            is_home = item['is_home']
            
            # Resolve Abbr
            if tid not in team_abbr_cache:
                try: team_abbr_cache[tid] = teams.find_team_name_by_id(tid)['abbreviation']
                except: team_abbr_cache[tid] = "UNK"
                
            if opp_id not in team_abbr_cache:
                try: team_abbr_cache[opp_id] = teams.find_team_name_by_id(opp_id)['abbreviation']
                except: team_abbr_cache[opp_id] = "UNK"
                
            own_abbr = team_abbr_cache[tid]
            opp_abbr = team_abbr_cache[opp_id]
            matchup = f"{own_abbr} vs. {opp_abbr}"
            
            new_row = {
                'PLAYER_ID': pid,
                'PLAYER_NAME': pname,
                'GAME_DATE': target_date,
                'MATCHUP': matchup,
                'TEAM_ID': tid,
                'PTS': np.nan, 'REB': np.nan, 'AST': np.nan,
                'MIN': 0, 'FGA': 0,
                'SEASON_YEAR': 2026
            }
            phantom_rows.append(new_row)
            
        if not phantom_rows:
            return []
            
        # Append all
        df_batch = pd.concat([df_history, pd.DataFrame(phantom_rows)], ignore_index=True)
        
        if log_manager: await log_manager.broadcast("Running Bulk Feature Engineering (This takes ~15s)...")
        
        # 3. Process Features (Heavy CPU - done once)
        df_processed = self.fe.process(df_batch, is_training=False)
        
        # Load resources for prediction
        self.load_common_resources()
        
        # 4. Predict Loop
        results_list = []
        target_dt = pd.to_datetime(target_date)
        
        # Extract just today's rows
        today_data = df_processed[df_processed['GAME_DATE'] == target_dt].copy()
        
        # Create map for fast lookup
        # today_data.set_index('PLAYER_ID', inplace=True) 
        # Actually duplicate PLAYER_IDs unlikely today but possible if double header? No.
        
        total_p = len(execution_list)
        
        # Pre-calculate Last Played Dates for Recency Filter
        # Ensure datetime
        df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
        last_played_map = df_history.groupby('PLAYER_ID')['GAME_DATE'].max().to_dict()
        
        target_dt = pd.to_datetime(target_date)
        
        if log_manager: await log_manager.broadcast("Inferencing Models...")
        
        # Fetch odds data ONCE at the start (used for Vegas totals in player loop)
        odds_data = fetch_nba_odds()
        
        processed_count = 0
        
        for item in execution_list:
            pid = item['pid']
            pname = item['pname']
            
            # RECENTLY ACTIVE CHECK
            # If player hasn't played in > 30 days, skip.
            last_date = last_played_map.get(pid)
            if last_date:
                days_inactive = (target_dt - last_date).days
                if days_inactive > 30:
                    # print(f"Skipping {pname} - Inactive for {days_inactive} days")
                    continue
            else:
                 # No history at all? Skip or keep?
                 # If no history, they are probably a rookie or new. Keep them or skip?
                 # If we have no history, we can't predict well anyway (using generic).
                 # Let's keep them if they are in the roster, maybe?
                 # Actually, if they have NO games in df_history (which covers multiple years), they are likely irrelevant or very new.
                 pass
            
            # Get row
            # mask = today_data['PLAYER_ID'] == pid ... fast enough?
            # optimization: use dictionary mapping
            # But let's just assume DataFrame lookup is fast enough for 200 items.
            
        # --- INJURY REPORT ---
        # --- INJURY REPORT ---
        from src.injury_service import load_injury_cache
        injury_cache = load_injury_cache()
        injury_map = injury_cache.get('injuries', {})
        if log_manager: await log_manager.broadcast(f"Loaded {len(injury_map)} injuries from cache for filtering.")
        # ---------------------
        # ---------------------
        
        player_ids = today_data['PLAYER_ID'].unique()
        
        for pid in player_ids:
            player_row = today_data[today_data['PLAYER_ID'] == pid]
            if player_row.empty:
                continue
                
            # Check Injury Status
            # Assume PLAYER_NAME is in the data (it should be)
            pname = player_row['PLAYER_NAME'].iloc[0]
            
            # Fuzzy Injury Lookup (Handle "T. HerroTyler Herro" case)
            injury_status = injury_map.get(pname)
            if not injury_status:
                # O(N) scan, but N is small (100-200 injuries)
                # Check if pname appears at end of any key (more likely) or just contained
                for k, v in injury_map.items():
                    if pname in k:
                        injury_status = v
                        break
            
            if injury_status == "Out":
                # if log_manager: await log_manager.broadcast(f"  🚫 Skipping {pname} (Out)") 
                # Commented out to reduce noise, but good for debug
                print(f"Skipping {pname} (Out)")
                continue
                
            processed_count += 1
            if processed_count % 10 == 0:
                 if log_manager: await log_manager.broadcast(f"Inference Progress: {processed_count}/{total_p}")
            
            # Load Model
            model = self._get_model(pid)
            
            # Prepare Tensor
            p_idx = torch.LongTensor(player_row['PLAYER_IDX'].values).to(device)
            t_idx = torch.LongTensor(player_row['TEAM_IDX'].values).to(device)
            x_cont = torch.FloatTensor(player_row[self.resources['feature_cols']].values).to(device)
            
            # Missing IDs embedding
            m_indices = []
            pad_idx = len(self.resources['p_enc'].classes_)
            m_indices += [pad_idx] * 3
            batch_size = len(player_row)
            m_idx_tensor = torch.LongTensor([m_indices] * batch_size).to(device)
            
            with torch.no_grad():
                preds = model(p_idx, t_idx, x_cont, m_idx_tensor)
                
            preds_np = preds.cpu().numpy()
            
            # Handle (6 targets: PTS, REB, AST, 3PM, BLK, STL)
            if preds_np.shape[1] == 12:  # 6 means + 6 logvars
                pred_pts = float(preds_np[0, 0])
                pred_reb = float(preds_np[0, 1])
                pred_ast = float(preds_np[0, 2])
                pred_3pm = float(preds_np[0, 3])
                pred_blk = float(preds_np[0, 4])
                pred_stl = float(preds_np[0, 5])
                
                # Extract learned std from log-variance
                std_pts = float(np.exp(0.5 * preds_np[0, 6]))
                std_reb = float(np.exp(0.5 * preds_np[0, 7]))
                std_ast = float(np.exp(0.5 * preds_np[0, 8]))
                std_3pm = float(np.exp(0.5 * preds_np[0, 9]))
                std_blk = float(np.exp(0.5 * preds_np[0, 10]))
                std_stl = float(np.exp(0.5 * preds_np[0, 11]))
                
                # Use learned std as MAE proxy
                mae_pts = std_pts
                mae_reb = std_reb
                mae_ast = std_ast
                mae_3pm = std_3pm
                mae_blk = std_blk
                mae_stl = std_stl
                
                mae_pra = mae_pts + mae_reb + mae_ast
            else:
                # Should not happen if models match, but fallback just in case
                pred_pts, pred_reb, pred_ast = 0,0,0
                pred_3pm, pred_blk, pred_stl = 0,0,0
                mae_pts, mae_reb, mae_ast = 1,1,1
                mae_3pm, mae_blk, mae_stl = 1,1,1
                mae_pra = 3
            
            # --- VEGAS GAME TOTAL ADJUSTMENT ---
            # High game totals (e.g., 230+) imply fast pace → more scoring opportunity
            # Low game totals (e.g., 205) imply slow grind → less stats
            # Average NBA total is ~218, so we adjust based on deviation
            vegas_adjustment = 1.0
            game_total = None
            
            team_abbrev = player_row['TEAM_ABBREVIATION'].iloc[0] if 'TEAM_ABBREVIATION' in player_row.columns else None
            if team_abbrev and isinstance(team_abbrev, str):
                game_total_info = get_game_total_for_matchup(team_abbrev, odds_data)
            else:
                game_total_info = {'found': False}
            if game_total_info.get('found'):
                game_total = game_total_info['total']
                # Scale: +/- 5% for every 10 points deviation from 218
                deviation = (game_total - 218) / 10
                vegas_adjustment = 1.0 + (deviation * 0.05)
                vegas_adjustment = max(0.85, min(1.15, vegas_adjustment))  # Cap at ±15%
                
                pred_pts *= vegas_adjustment
                pred_reb *= vegas_adjustment
                pred_ast *= vegas_adjustment
            
            pred_pra = pred_pts + pred_reb + pred_ast
            pred_pr = pred_pts + pred_reb
            pred_pa = pred_pts + pred_ast
            pred_ra = pred_reb + pred_ast
            
            # FILTER: Exclude Inactive Players
            if pred_pra < 1.0:
                continue

            # 2. Get Last 5 Games Context
            last_5 = self._get_last_5(pid, df_history, target_dt)

            # 3. Calculate Betting Lines & Units
            # We don't have real lines, so we simulate "Implied Lines" based on predictions
            # The "Buy" line is Pred - MAE (Consumer safety)
            # The "Sell" line is Pred + MAE
            
            # We calculate "Confidence Score" for the implied bet.
            # Since we define the line ourselves here as the "Safe Zone", 
            # We can treat the "Edge" as the gap we are giving.
            
            # Let's Standardize Unit Recommender based on "Projection Stability" (Low MAE = High Conf)
            # And "Recent Form" (If Trend aligns with Pred -> Boost)
            
            # Get Season Stats (Move UP for filtering)
            season_stats = self._get_season_stats(pid, df_history)
            season_pts = season_stats.get('pts', 0.0)
            
            # --- QUALITY CONTROL ---
            # 1. Clip Negatives
            pred_pts = max(0.0, float(pred_pts))
            pred_reb = max(0.0, float(pred_reb))
            pred_ast = max(0.0, float(pred_ast))
            pred_3pm = max(0.0, float(pred_3pm))
            pred_blk = max(0.0, float(pred_blk))
            pred_stl = max(0.0, float(pred_stl))
            pred_pra = pred_pts + pred_reb + pred_ast
            
            # 2. Relevance Filter (Restore Strict Filter per User Request)
            if season_pts < 8.0 and pred_pts < 10.0:
                 continue
            # if season_pts < 1.0 and pred_pts < 1.0: # Only skip absolute zeros
            #      continue

            valid_bets = []
            
            # Analyze Main Props
            props = [
                ('PTS', pred_pts, mae_pts, last_5.get('avg_pts', 0)),
                ('REB', pred_reb, mae_reb, last_5.get('avg_reb', 0)),
                ('AST', pred_ast, mae_ast, last_5.get('avg_ast', 0)),
                ('PRA', pred_pra, mae_pra, last_5.get('avg_pra', 0)),
                ('3PM', pred_3pm, mae_3pm, 0), # No last 5 data yet for these
                ('BLK', pred_blk, mae_blk, 0),
                ('STL', pred_stl, mae_stl, 0)
            ]
            
            # Smart Unit Logic
            # Base Unit = 1.0
            # Multipliers:
            # - Low Variance (MAE < X% of Pred): x1.2
            # - Recent Form Match (Last 5 avg is close to Pred): x1.2
            # - Inconsistency (Last 5 std dev high): x0.8
            
            best_prop = None
            highest_score = -1
            
            for p_name, val, mae, recent in props:
                # Simple Heuristic score for "Bet Quality"
                # We want High Val relative to MAE (Signal to Noise)
                # But mostly we want STABILITY.
                
                # Check for "Edge" relative to a hypothetical bookmaker line.
                # Since we don't have lines, we can't calculate true Edge.
                # We will output the "Safe Line" to bet vs.
                
                # Confidence Calculation
                # Z-Score equivalent: How many MAEs away from 0 is the prediction? 
                # (Not really relevant for Over/Under unless we have a line)
                # Let's use Inverse CV (Coefficient of Variation) = Mean / StdDev(MAE)
                # Higher is better.
                if val < 0.1: continue
                cv_score = val / (mae + 0.1) 
                
                # Unit Calc - UPDATED FOR CONSISTENCY
                # We want CONSISTENT players.
                # Measure: CV (StdDev / Mean) of Last 5 Games.
                # If CV is low (< 0.2), they are very consistent.
                
                std_dev = last_5.get('std_pra', 10.0) # Default high variance
                avg_l5 = last_5.get('avg_pra', 1.0)
                
                # Consistency Factor (Inverse of Variance)
                # 0.0 - 1.0 score (1.0 = Rock Solid)
                consistency_score = 1.0
                if avg_l5 > 0:
                     cv_val = std_dev / avg_l5
                     # Map CV 0.1 -> 1.0, CV 0.5 -> 0.0
                     consistency_score = max(0.0, 1.0 - (cv_val * 2))
                
                # --- SCIENTIFIC UNIT SIZING (KELLY CRITERION) ---
                # We replace heuristics with probability theory.
                
                # 1. Estimate Win Probability (P)
                # Base rate for NFL/NBA props is roughly 53-54% for breakeven.
                # We start at 51% (Conservative)
                base_prob = 0.51
                
                # Boosts
                consistency_boost = consistency_score * 0.15 # Up to +15% for rock solid players
                model_boost = min(0.08, cv_score / 50.0)    # Up to +8% for huge model edges
                
                # Penalty for High Variance (already in consistency, but let's be safe)
                if std_dev > avg_l5 * 0.3:
                    consistency_boost *= 0.5
                
                win_prob = base_prob + consistency_boost + model_boost
                
                # Cap prob at 70% (realistic ceiling for sports models)
                win_prob = min(0.70, win_prob)
                
                # 2. Apply Kelly Criterion
                # f = (bp - q) / b
                # b = odds - 1 (Assuming standard -110 lines => decimal 1.91 => b = 0.91)
                b = 0.91
                q = 1 - win_prob
                
                f_star = ((b * win_prob) - q) / b
                
                # 3. Fractional Kelly (Safety)
                # Eighth Kelly (Very Safe) because our P estimate is noisy.
                kelly_fraction = 0.125 
                recommended_bankroll_pct = max(0.0, f_star * kelly_fraction)
                
                # Convert to "Units" (Assuming 1 Unit = 1% of Bankroll)
                # If Kelly says bet 1% of bankroll, that is 1.0 Units.
                raw_units = recommended_bankroll_pct * 100.0
                
                # 4. Badging & Caps & STABILITY GATES
                units = round(raw_units * 4) / 4 # Round to nearest 0.25
                
                # --- CONSISTENCY GATES ---
                # You cannot be Diamond if you are erratic (D grade).
                if consistency_score < 0.4:
                    units = min(units, 0.25) # Max LEAN for D-Grade
                elif consistency_score < 0.6:
                    units = min(units, 0.50) # Max STANDARD for C-Grade
                elif consistency_score < 0.75:
                    units = min(units, 0.75) # Max GOLD for B-Grade
                    
                # Hard Cap (User Rule: Max 1.0u)
                if units > 1.0: units = 1.0
                
                # Determine Badge based on Final Units
                badge = "LEAN" # < 0.5
                if units >= 0.5: badge = "STANDARD"
                if units >= 0.75: badge = "GOLD 🥇"
                if units >= 1.0: badge = "DIAMOND 💎"
                
                # Filter out tiny bets? NO, keep them but mark as NO BET if 0
                if units < 0.25: 
                    if f_star > 0: units = 0.25
                    else: units = 0.0 # No Edge
                
                # Check against highest
                # If units > highest, take it.
                if units > highest_score:
                    highest_score = units
                    best_prop = {
                        'prop': p_name,
                        'val': val,
                        'line_low': val - mae,
                        'line_high': val + mae,
                        'units': units,
                        'badge': badge if units >= 0.25 else "NO BET", # Show NO BET
                        'consistency': consistency_score,
                        'mae': mae
                    }

            # Fallback if no prop found (e.g. all 0 units)
            if best_prop is None:
                 best_prop = {
                        'prop': 'PTS', # Default
                        'val': pred_pts,
                        'line_low': pred_pts - mae_pts,
                        'line_high': pred_pts + mae_pts,
                        'units': 0.0,
                        'badge': "NO BET",
                        'consistency': 0.5,
                        'mae': mae_pts
                    }

            # Get Season Stats (Already calculated above)
            # season_stats = self._get_season_stats(pid, df_history)

            # ---- SGP META MODEL INFERENCE ----
            sgp_info = {'active': False, 'prob': 0.0, 'legs': []}
            if self.sgp_loaded:
                try:
                    # Construct Input Vector (12 dims): [PTS_PRED, PTS_STD, REB_PRED, ...]
                    # Order must match training: PTS, REB, AST, 3PM, BLK, STL
                    sgp_feats = [
                        pred_pts, std_pts,
                        pred_reb, std_reb,
                        pred_ast, std_ast,
                        pred_3pm, std_3pm,
                        pred_blk, std_blk,
                        pred_stl, std_stl
                    ]
                    sgp_tensor = torch.FloatTensor([sgp_feats]).to(device)
                    # Scale (Manual transform using joblib scaler)
                    # Sklearn scaler expects (N, 12).
                    # Need to perform transform on CPU or replicate scaler logic?
                    # Faster to just use scaler.transform
                    scaled_feats = self.sgp_scaler.transform([sgp_feats])
                    scaled_tensor = torch.FloatTensor(scaled_feats).to(device)
                    
                    with torch.no_grad():
                        sgp_out = self.sgp_model(scaled_tensor)
                        sgp_probs = torch.sigmoid(sgp_out).cpu().numpy()[0] # [6]
                    
                    # Map back to names
                    stat_names = ['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL']
                    
                    # Create List of (Name, Prob, SafeVal)
                    legs = []
                    for i, name in enumerate(stat_names):
                        # Calculate Safe Line (Same as training)
                        margin = 2.0 if name == 'PTS' else 1.0
                        pred_val = sgp_feats[i*2]
                        safe_line = max(0.5, pred_val - margin) # Ensure at least 0.5?
                        legs.append({
                            'stat': name,
                            'prob': float(sgp_probs[i]),
                            'line': safe_line
                        })
                        
                    # Select Top 3 for Trixie
                    # User Logic: Model chooses "Main" + 2 "Safe".
                    # We sort by Probability descending.
                    legs.sort(key=lambda x: x['prob'], reverse=True)
                    top_3 = legs[:3]
                    
                    # Combined Probability (Naive Independence assumption, but model trained on individual hits)
                    # Actually Meta Model output is individual hit probability.
                    # We approximate Joint Prob = P1 * P2 * P3
                    combined_prob = top_3[0]['prob'] * top_3[1]['prob'] * top_3[2]['prob']
                    
                    sgp_info = {
                        'active': True,
                        'prob': combined_prob,
                        'legs': top_3,
                        'all_legs': legs
                    }
                except Exception as e:
                    print(f"SGP Inference Error: {e}")
            
            # ----------------------------------

            res = {
                'PLAYER_ID': int(pid),
                'PLAYER_NAME': str(pname),
                'GAME_DATE': target_date,
                'MATCHUP': str(player_row['MATCHUP'].iloc[0]),
                'OPPONENT': str(player_row['OPP_TEAM_ABBREVIATION'].iloc[0]),
                'IS_HOME': bool(item.get('is_home', False)),
                
                # Full Stats
                'PRED_PTS': float(pred_pts), 'PRED_REB': float(pred_reb), 'PRED_AST': float(pred_ast), 'PRED_PRA': float(pred_pra),
                'PRED_PR': float(pred_pr), 'PRED_PA': float(pred_pa), 'PRED_RA': float(pred_ra),
                'PRED_3PM': float(pred_3pm), 'PRED_BLK': float(pred_blk), 'PRED_STL': float(pred_stl),
                
                # Season Context 
                'SEASON_PTS': float(season_stats.get('pts', 0)),
                'SEASON_REB': float(season_stats.get('reb', 0)),
                'SEASON_AST': float(season_stats.get('ast', 0)),
                'SEASON_3PM': float(season_stats.get('3pm', 0)),
                'SEASON_BLK': float(season_stats.get('blk', 0)),
                'SEASON_STL': float(season_stats.get('stl', 0)),
                'SEASON_PRA': float(season_stats.get('pra', 0)),
                
                # MAE
                'MAE_PTS': float(mae_pts), 'MAE_REB': float(mae_reb), 'MAE_AST': float(mae_ast), 'MAE_PRA': float(mae_pra),
                'MAE_3PM': float(mae_3pm), 'MAE_BLK': float(mae_blk), 'MAE_STL': float(mae_stl),

                # Safe Lines (for UI Grid)
                'LINE_PTS_LOW': float(pred_pts - mae_pts), 'LINE_PTS_HIGH': float(pred_pts + mae_pts),
                'LINE_REB_LOW': float(pred_reb - mae_reb), 'LINE_REB_HIGH': float(pred_reb + mae_reb),
                'LINE_AST_LOW': float(pred_ast - mae_ast), 'LINE_AST_HIGH': float(pred_ast + mae_ast),
                'LINE_PRA_LOW': float(pred_pra - mae_pra), 'LINE_PRA_HIGH': float(pred_pra + mae_pra),
                
                # Best Bet (Primary Display)
                'BEST_PROP': best_prop['prop'],
                'BEST_VAL': best_prop['val'],
                'LINE_LOW': best_prop['line_low'],
                'LINE_HIGH': best_prop['line_high'],
                'UNITS': best_prop['units'],
                'BADGE': best_prop['badge'],
                'CONSISTENCY': best_prop['consistency'],
                
                # Context
                'LAST_5': last_5,
                
                # Vegas Context
                'VEGAS_TOTAL': game_total,
                'VEGAS_ADJUSTMENT': round(vegas_adjustment, 3) if vegas_adjustment != 1.0 else None,
                
                # Model Info
                'MODEL_VERSION': 'V2',
                'UNCERTAINTY_SOURCE': 'learned',
                
                # Market Odds (populated below)
                'MARKET_LINE': None,
                'MARKET_ODDS': None,
                'EV': None,
                'EV_RECOMMENDATION': None,

                # SGP / Trixie Info
                'SGP_PROB': float(sgp_info['prob']),
                'TRIXIE_LEGS': sgp_info['legs'] if sgp_info['active'] else [],
                'SGP_LEGS': sgp_info.get('all_legs', []) if sgp_info['active'] else [],
                'SGP_ACTIVE': sgp_info['active']
            }
            results_list.append(res)
        
        # 4a. Populate Market Odds & EV
        if log_manager: await log_manager.broadcast("Fetching market odds...")
        odds_data = fetch_nba_odds()
        
        for res in results_list:
            try:
                ev_result = get_ev_for_prediction(
                    player_name=res['PLAYER_NAME'],
                    stat_type=res['BEST_PROP'],
                    prediction=res['BEST_VAL'],
                    model_confidence=min(res.get('CONSISTENCY', 0.5), 0.75),  # Cap at 75%
                    odds_data=odds_data
                )
                if ev_result.get('found'):
                    res['MARKET_LINE'] = ev_result.get('market_line')
                    res['MARKET_ODDS'] = ev_result.get('odds_over')
                    res['EV'] = round(ev_result.get('ev_over', 0) * 100, 1)  # As percentage
                    res['EV_RECOMMENDATION'] = ev_result.get('recommendation')
            except Exception as e:
                pass  # Silently skip if odds lookup fails
            
        # 4. Generate "The SUPER Trixie" (3 Players x 3 Bets)
        # Sort by BADGE TIER then UNITS then CONSISTENCY
        # Tier Values: Diamond=4, Gold=3, Standard=2, Lean=1, No Bet=0
        
        def get_tier_val(badge):
            if "DIAMOND" in badge: return 4
            if "GOLD" in badge: return 3
            if "STANDARD" in badge: return 2
            if "LEAN" in badge: return 1
            return 0
            
        sorted_res = sorted(results_list, key=lambda x: (
            get_tier_val(x['BADGE']), 
            x['UNITS'], 
            x['CONSISTENCY']
        ), reverse=True)
        
        # DEBUG: Verify Sort Order
        print("[Sort Debug] Top 5 Sorted Players:")
        for i, p in enumerate(sorted_res[:5]):
             print(f"  {i+1}. {p['PLAYER_NAME']} | {p['BADGE']} | {p['UNITS']}u")
        
        # 4. Generate "Daily Trixie" (Top 3 Players by SGP Confidence)
        trixie = None
        if len(results_list) >= 3:
            # DEBUG
            try:
                debug_pts = [p.get('SEASON_PTS') for p in results_list[:5]]
                print(f"[TRIXIE DEBUG] Count: {len(results_list)}. Sample PTS: {debug_pts}")
            except: pass

            # Helper to score badges
            def get_badge_score(p):
                b = p.get('BADGE', '')
                if 'DIAMOND' in b: return 3
                if 'GOLD' in b: return 2
                if 'SILVER' in b: return 1
                return 0

            # Filter for Volume first (Trixie Strategy requires reliable stars)
            candidates = [p for p in results_list if p.get('SEASON_PTS', 0) >= 12.0]
            if len(candidates) < 3:
                candidates = results_list # Fallback
            
            # Sort by Badge Score desc, then SGP Prob desc (Prioritize Diamond/Gold)
            trixie_candidates = sorted(candidates, key=lambda x: (get_badge_score(x), x.get('SGP_PROB', 0.0)), reverse=True)
            top_3 = trixie_candidates[:3]
            
            legs = []
            combined_prob = 1.0
            
            for p in top_3:
                player_bets = []
                
                # 1. Main Bet (Best Prop)
                main_prop = p['BEST_PROP']
                # Format Main Bet Description
                main_desc = f"{p['BEST_PROP']} {p['BEST_VAL']:.1f}"
                
                player_bets.append({
                    'type': 'MAIN',
                    'prop': main_prop,
                    'val': p['BEST_VAL'],
                    'desc': main_desc,
                    'prob': 0.55 # Implicit
                })
                
                # 2. Safe Anchors (From SGP Meta Model)
                potential_anchors = p.get('SGP_LEGS', [])
                # Ensure sorted by prob
                potential_anchors.sort(key=lambda x: x['prob'], reverse=True)
                
                anchors_added = 0
                for anc in potential_anchors:
                    if anchors_added >= 2: break
                    
                    stat = anc['stat']
                    prob = anc['prob']
                    line = anc['line']
                    
                    # Skip if same as main prop
                    if stat == main_prop: continue
                    
                    # Filter Trivia
                    # User: "Lowest at 1". So we accept line >= 1.5 (which means betting Over 1.5 -> 2+)
                    # For PTS, we want at least a bucket or two (Over 3.5 -> 4+)
                    min_threshold = 3.5 if stat == 'PTS' else 1.5
                    
                    if line < min_threshold: continue

                    # Re-added Probability Floor (User: "25% confident is weird")
                    if prob < 0.55: continue
                    
                    # RELAXED: Removed 0.75 probability gate to ensure we populate the card.
                    # We rely on sorting by prob to get the best available.
                    
                    player_bets.append({
                        'type': 'SAFE',
                        'prop': stat,
                        'val': line,
                        'desc': f"{stat} {line:.1f}+",
                        'prob': prob
                    })
                    anchors_added += 1
                
                legs.append({
                    **p, # Include full stats for Modal
                    'player': p['PLAYER_NAME'],
                    'badge': p['BADGE'],
                    'units': p['UNITS'],
                    'bets': player_bets,
                    'sgp_prob': p.get('SGP_PROB', 0.5)
                })
                
                combined_prob *= p.get('SGP_PROB', 0.5)
            
            trixie = {
                'active': True,
                'combined_prob': combined_prob,
                'legs': legs,
                'title': "DAILY AI TRIXIE"
            }

        return {'predictions': sorted_res, 'trixie': trixie}

    def _get_season_stats(self, pid, df_history):
         try:
            p_rows = df_history[df_history['PLAYER_ID'] == pid]
            if p_rows.empty: return {}
            
            def safe_float(x):
                import math
                try:
                    v = float(x)
                    if math.isnan(v): return 0.0
                    return v
                except: return 0.0

            return {
                'pts': safe_float(p_rows['PTS'].mean()),
                'reb': safe_float(p_rows['REB'].mean()),
                'ast': safe_float(p_rows['AST'].mean()),
                '3pm': safe_float(p_rows['FG3M'].mean()),
                'blk': safe_float(p_rows['BLK'].mean()),
                'stl': safe_float(p_rows['STL'].mean()),
                'pra': safe_float((p_rows['PTS'] + p_rows['REB'] + p_rows['AST']).mean())
            }
         except:
            return {}

    def _get_last_5(self, pid, df_history, target_date):
        try:
            p_rows = df_history[
                (df_history['PLAYER_ID'] == pid) & 
                (df_history['GAME_DATE'] < target_date)
            ].sort_values('GAME_DATE', ascending=False).head(5)
            
            if p_rows.empty:
                return {}
                
            pra_series = p_rows['PTS'] + p_rows['REB'] + p_rows['AST']
            
            def safe_float(x):
                import math
                try:
                    v = float(x)
                    if math.isnan(v): return 0.0
                    return v
                except: return 0.0
                
            return {
                'avg_pts': safe_float(p_rows['PTS'].mean()),
                'avg_reb': safe_float(p_rows['REB'].mean()),
                'avg_ast': safe_float(p_rows['AST'].mean()),
                'avg_3pm': safe_float(p_rows['FG3M'].mean()),
                'avg_blk': safe_float(p_rows['BLK'].mean()),
                'avg_stl': safe_float(p_rows['STL'].mean()),
                'avg_pra': safe_float(pra_series.mean()),
                'std_pra': safe_float(pra_series.std()),
                'games': p_rows['GAME_DATE'].dt.strftime('%Y-%m-%d').tolist(),
                'pts_log': p_rows['PTS'].tolist()
            }
        except:
             return {}

