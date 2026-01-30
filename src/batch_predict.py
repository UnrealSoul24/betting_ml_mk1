import pandas as pd
import numpy as np
import os
import torch
import joblib
from datetime import datetime
from src.data_fetch import fetch_daily_scoreboard
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
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
        
        print("ANTIGRAVITY DEBUG: analyze_today_batch STARTED. Code Version: RELOADED") # DEBUG
        # Initialize result variables early
        trixie = None
        
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
            
            # Get Season Stats
            season_stats = self._get_season_stats(pid, df_history)
            season_pts = season_stats.get('pts', 0.0)

            # 3. Calculate Betting Lines & Units
            from src.bet_sizing import calculate_bet_quality

            props = [
                ('PTS', pred_pts, mae_pts, pred_pts),
                ('REB', pred_reb, mae_reb, pred_reb), # val placeholder
                ('AST', pred_ast, mae_ast, pred_ast),
                ('PRA', pred_pra, mae_pra, pred_pra),
                ('RA', pred_ra, mae_reb + mae_ast, pred_ra), 
                ('PR', pred_pr, mae_pts + mae_reb, pred_pr),
                ('PA', pred_pa, mae_pts + mae_ast, pred_pa),
                ('3PM', pred_3pm, mae_3pm, pred_3pm),
                ('BLK', pred_blk, mae_blk, pred_blk),
                ('STL', pred_stl, mae_stl, pred_stl)
            ]
            
            best_prop = None
            highest_score = -1
            
            for p_name, val, mae, _ in props: # _ was val repeat
                # Calculate quality
                # Safe Line: For display in grid.
                calc = calculate_bet_quality(p_name, val, mae, val, last_5)
                
                # Check for "Edge" relative to highest score
                if calc['units'] > highest_score:
                    highest_score = calc['units']
                    best_prop = calc

            # Fallback
            if best_prop is None:
                 best_prop = {
                        'prop': 'PTS',
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
                # Add others if needed for UI Grid
                
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
            }
            results_list.append(res)
        
        # 4a. Populate Market Odds & EV
        # 4a. Populate Market Odds & EV
        if log_manager: await log_manager.broadcast("Fetching market odds for comprehensive analysis...")
        try:
            odds_data = fetch_nba_odds()
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to fetch odds: {e}")
            if log_manager: await log_manager.broadcast(f"Error fetching odds: {e}")
            odds_data = None
        
        # 4a. Comprehensive Prop Analysis (Populate EV for ALL stats)
        for res in results_list:
            res['PROPS_ANALYSIS'] = {}
            
            # Map internal keys to stat types
            stats_to_analyze = [
                ('PTS', 'PRED_PTS', 'MAE_PTS'),
                ('REB', 'PRED_REB', 'MAE_REB'),
                ('AST', 'PRED_AST', 'MAE_AST'),
                ('3PM', 'PRED_3PM', 'MAE_3PM'),
                ('BLK', 'PRED_BLK', 'MAE_BLK'),
                ('STL', 'PRED_STL', 'MAE_STL'),
                ('RA', 'PRED_RA', 'MAE_REB'), # Approximating MAE for composites 
                ('PR', 'PRED_PR', 'MAE_REB'), 
                ('PA', 'PRED_PA', 'MAE_AST'),
                ('PRA', 'PRED_PRA', 'MAE_PRA')
            ]
            
            for stat_code, pred_key, mae_key in stats_to_analyze:
                try:
                    pred_val = res.get(pred_key, 0)
                    mae_val = res.get(mae_key, 1.0)
                    
                    ev_result = get_ev_for_prediction(
                        player_name=res['PLAYER_NAME'],
                        stat_type=stat_code,
                        prediction=pred_val,
                        std_dev=mae_val, # Use MAE as proxy for Standard Deviation
                        odds_data=odds_data
                    )
                    
                    if ev_result.get('found'):
                        res['PROPS_ANALYSIS'][stat_code] = {
                            'market_line': ev_result.get('market_line'),
                            'odds_over': ev_result.get('odds_over'),
                            'p_over': round(ev_result.get('p_over', 0.5) * 100, 1),
                            'ev_over': round(ev_result.get('ev_over', 0) * 100, 1),
                            'recommendation': ev_result.get('recommendation')
                        }
                        
                        # If this corresponds to the "Best Prop", update top-level fields for backwards capability
                        if stat_code == res['BEST_PROP']:
                            res['MARKET_LINE'] = ev_result.get('market_line')
                            recommendation = ev_result.get('recommendation')
                            
                            # Select the correct EV based on recommendation
                            if recommendation == 'UNDER':
                                final_ev = ev_result.get('ev_under', 0)
                                res['MARKET_ODDS'] = ev_result.get('odds_under') # Also show under odds
                            else:
                                final_ev = ev_result.get('ev_over', 0)
                                res['MARKET_ODDS'] = ev_result.get('odds_over') # Default to Over

                            res['EV'] = round(final_ev * 100, 1)
                            res['EV_RECOMMENDATION'] = recommendation
                            
                except Exception as e:
                    # print(f"Error analyzing {stat_code} for {res['PLAYER_NAME']}: {e}")
                    pass

            
        # 4. Generate "The SMART Trixie" (Mathematical SGP Optimizer)
        # Goal: Find 3 Players with the highest probability "SGP" (2-3 legs) that pays ~3.0x
        
        # Sort by BADGE TIER then UNITS then CONSISTENCY for general display
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

        trixie = None
        candidate_sgps = [] # tuples of (player_dict, best_sgp_dict, joint_prob)
        
        from itertools import combinations
        from src.odds_service import american_to_decimal
        
        # 1. Analyze EVERY player for their best possible SGP
        for p in sorted_res:
            # Gather all "Good Bets" for this player
            good_bets = []
            
            # Check stats (updated to include composites)
            stats_list = ['PTS', 'REB', 'AST', '3PM', 'STL', 'RA', 'PR', 'PA', 'PRA']
            
            for stat in stats_list:
                analysis = p.get('PROPS_ANALYSIS', {}).get(stat)
                if not analysis: continue
                
                rec = analysis.get('recommendation')
                if rec == 'PASS' or not rec: continue
                
                # Get Prob & Odds
                prob = analysis.get('p_over', 0)
                if rec == 'UNDER':
                    prob = 100 - prob # Convert to Under Prob if needed (data seems to be % already)
                    odds = analysis.get('odds_under')
                else:
                    odds = analysis.get('odds_over')
                
                # Strict Probability Gate
                if prob < 55.0: continue
                
                # Odds Filter (User Request: Exclude < 1.50 / -200)
                if american_to_decimal(odds) < 1.50: continue
                
                market_line = analysis.get('market_line')
                
                good_bets.append({
                    'prop': stat,
                    'direction': rec,
                    'line': market_line,
                    'prob': prob / 100.0, # Decimal prob
                    'odds_amer': odds,
                    'odds_dec': american_to_decimal(odds),
                    'desc': f"{stat} {rec[0]} {market_line}", # e.g. PTS O 25.5
                    'prediction': p.get(f'PRED_{stat}'),
                    'matchup': p.get('MATCHUP') # Store matchup for filtering
                })
                
            # If we don't have enough good bets to make a parlay, skip
            if len(good_bets) < 2:
                continue
                
            # 2. Find Best SGP Combination (2 or 3 legs)
            best_sgp = None
            best_sgp_score = -1
            
            # Test all 2-leg and 3-leg combinations
            for r in [2, 3]:
                for combo in combinations(good_bets, r):
                    joint_prob = 1.0
                    combined_odds = 1.0
                    
                    # Count "Safe Legs" (Prob > 60%) to boost score
                    safe_legs = 0
                    
                    for leg in combo:
                        joint_prob *= leg['prob']
                        combined_odds *= leg['odds_dec']
                        if leg['prob'] > 0.60: safe_legs += 1
                        
                    # Correlation Penalty
                    joint_prob *= 0.85
                    combined_odds *= 0.85 
                    
                    if combined_odds < 2.0: continue # Minimum odds floor
                        
                    # SCORE CALCULATION
                    ev = (joint_prob * combined_odds) - 1
                    score = ev * (joint_prob * 10) 
                    
                    # Correlation Boosts
                    directions = [leg['direction'] for leg in combo]
                    if all(d == 'UNDER' for d in directions):
                        score *= 1.30 
                    elif all(d == 'OVER' for d in directions):
                        score *= 1.15 
                    
                    if score > best_sgp_score:
                        best_sgp_score = score
                        best_sgp = {
                            'legs': combo,
                            'total_odds': round(combined_odds, 2),
                            'joint_prob': joint_prob,
                            'ev_edge': ev
                        }
            
            if best_sgp:
                candidate_sgps.append({
                    'player': p,
                    'sgp': best_sgp,
                    'score': best_sgp_score,
                    'matchup': p.get('MATCHUP') # Explicitly bubble up matchup
                })
                
        # 3. Select Top Players (Unique Game Constraint)
        candidate_sgps.sort(key=lambda x: x['score'], reverse=True)
        
        seen_matchups = set()
        primary_candidates = []
        
        # Select Primary (Top 3 Unique Matchups)
        for cand in candidate_sgps:
            if len(primary_candidates) >= 3:
                break
            
            m = cand['matchup']
            if m not in seen_matchups:
                primary_candidates.append(cand)
                seen_matchups.add(m)
        
        # Select Secondary (Next 3 Unique Matchups)
        secondary_candidates = []
        # Continue scanning from where we left off? 
        # Easier to just rescan list and skip 'seen' from primary phase?
        # Actually, let's just grab the next valid ones.
        
        for cand in candidate_sgps:
            if cand in primary_candidates: continue # Already picked
            if len(secondary_candidates) >= 3: break
            
            m = cand['matchup']
            if m not in seen_matchups:
                 secondary_candidates.append(cand)
                 seen_matchups.add(m)
                 
        # Alternates (Just the next best, regardless of matchup? Or strict?)
        # Let's keep strict unique for alternates too if possible to maximize options
        alternates = []
        for cand in candidate_sgps:
            if cand in primary_candidates or cand in secondary_candidates: continue
            if len(alternates) >= 2: break
             
            # Allow repeats for alternates? Maybe. But let's try unique first.
            m = cand['matchup']
            if m not in seen_matchups:
                alternates.append(cand)
                seen_matchups.add(m)
                
        # Fallback: If we don't have enough unique alternates, fill with whatever is left (duplicates allowed)
        if len(alternates) < 2:
            for cand in candidate_sgps:
                if cand in primary_candidates or cand in secondary_candidates or cand in alternates: continue
                if len(alternates) >= 2: break
                alternates.append(cand)

        trixie = None
        
        # Helper to format SGP and calc Units
        def format_sgp_legs(candidates):
            formatted_list = []
            for item in candidates:
                p = item['player']
                sgp = item['sgp']
                prob = sgp['joint_prob']
                odds = sgp['total_odds']
                
                # Dynamic Sizing for Single Leg (Player SGP)
                rec_leg_units = 0.1
                if prob > 0.50: rec_leg_units = 0.5
                elif prob > 0.35: rec_leg_units = 0.25
                elif prob > 0.25: rec_leg_units = 0.15
                
                formatted_bets = []
                for b in sgp['legs']:
                    formatted_bets.append({
                        'type': 'LEG', 
                        'prop': b['prop'],
                        'direction': b['direction'],
                        'target': b['line'],
                        'prediction': b['prediction'],
                        'desc': b['desc'],
                        'odds': b['odds_amer'],
                        'prob': b['prob'] * 100
                    })
                
                formatted_list.append({
                    **p,
                    'player_name': p['PLAYER_NAME'],
                    'matchup': p['MATCHUP'],
                    'bets': formatted_bets,
                    'sgp_odds': odds,
                    'win_prob': prob * 100,
                    'rec_units': rec_leg_units
                })
            return formatted_list

        if len(primary_candidates) == 3:
            primary_legs = format_sgp_legs(primary_candidates)
            primary_total_odds = 1.0
            primary_joint_prob = 1.0
            
            for leg in primary_legs: 
                primary_total_odds *= leg['sgp_odds']
                # primary_joint_prob *= (leg['win_prob'] / 100.0) # Careful with multiplying probs of dependent events?
                # Actually, unique games = independent events. So simple multiplication is valid.
                primary_joint_prob *= (leg['win_prob'] / 100.0)
                
            # Dynamic Trixie Units
            trixie_units = 0.1
            if primary_joint_prob * primary_total_odds > 1.2: # EV > +20%
                trixie_units = 0.25
            
            secondary_legs = []
            secondary_total_odds = 0
            if len(secondary_candidates) == 3:
                secondary_legs = format_sgp_legs(secondary_candidates)
                secondary_total_odds = 1.0
                for leg in secondary_legs: secondary_total_odds *= leg['sgp_odds']

            formatted_alternates = format_sgp_legs(alternates)
            
            trixie = {
                'sgp_legs': primary_legs,
                'secondary_legs': secondary_legs if secondary_legs else None,
                'alternates': formatted_alternates,
                'total_odds': round(primary_total_odds, 2),
                'secondary_total_odds': round(secondary_total_odds, 2) if secondary_legs else 0,
                'rec_units': trixie_units,
                'potential_return': round(primary_total_odds * trixie_units, 2)
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

