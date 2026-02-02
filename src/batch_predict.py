import pandas as pd
import numpy as np
import os
import torch
import joblib
from datetime import datetime
from src.data_fetch import fetch_daily_scoreboard
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
from src.odds_service import fetch_nba_odds, get_ev_for_prediction, get_game_total_for_matchup, american_to_decimal
from nba_api.stats.static import teams
import asyncio
from src.calibration_tracker import CalibrationTracker, load_calibration_factors
from src.minutes_predictor import MinutesPredictor
from src.model_metadata_cache import get_metadata_cache
from itertools import combinations

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BatchPredictor:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.resources = {} # Cache resources
        self.models_cache = {} # Cache loaded PyTorch models
        self.calibration_factors = load_calibration_factors() # Load Calibration Factors
        self.tracker = CalibrationTracker()
        self.minutes_predictor = MinutesPredictor()
        self.metadata_cache = get_metadata_cache()

    def load_common_resources(self):
        print("Loading common resources...")
        self.resources['p_enc'] = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        self.resources['t_enc'] = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        self.resources['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
        self.resources['feature_cols'] = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
        
        # Initialize Minutes Predictor
        if not self.minutes_predictor.model:
            self.minutes_predictor._load_model()

    def _get_model(self, player_id: int):
        # Determine model path
        specific_path = os.path.join(MODELS_DIR, f'pytorch_player_{player_id}.pth')
        generic_path = os.path.join(MODELS_DIR, 'pytorch_nba_global_vNext.pth')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
        # Check cache
        if path_to_load in self.models_cache:
            return self.models_cache[path_to_load], path_to_load
            
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
        except Exception as e:
            print(f"Error loading model {path_to_load}: {e}")
            if path_to_load != generic_path and os.path.exists(generic_path):
                 try:
                    model.load_state_dict(torch.load(generic_path, map_location=device))
                    path_to_load = generic_path 
                 except: pass
            
        model.to(device)
        model.eval()
        
        self.models_cache[path_to_load] = model
        return model, path_to_load

    def _get_metrics(self, player_id: int):
        """Loads MAE metrics for confidence intervals."""
        specific_path = os.path.join(MODELS_DIR, f'metrics_player_{player_id}.json')
        generic_path = os.path.join(MODELS_DIR, 'metrics_global.json')
        
        path_to_load = specific_path if os.path.exists(specific_path) else generic_path
        
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
        Main Batch Function with Two-Stage Prediction.
        1. Loads History & Resources.
        2. Creates Phantom Rows.
        3. Feature Engineering.
        4. Predicts: Minutes -> Per-Minute Stats -> Totals.
        5. Generates Trixie/SGP recommendations.
        """
        target_date = date_input if (isinstance(date_input, str) and date_input) else datetime.now().strftime('%Y-%m-%d')
        
        print(f"ANTIGRAVITY DEBUG: analyze_today_batch STARTED. Date: {target_date}")
        
        if log_manager: await log_manager.broadcast("Loading historical data from disk...")
        
        # 1. Load All Data (Heavy IO - done once)
        df_history = await asyncio.to_thread(self.fe.load_all_data)
        
        if log_manager: await log_manager.broadcast("Generating Prediction Rows...")
        
        # 2. Create Phantom Rows
        phantom_rows = []
        team_abbr_cache = {}
        
        # execution_list might be a DataFrame, convert to dict records for safe iteration
        items = execution_list.to_dict('records') if isinstance(execution_list, pd.DataFrame) else (execution_list or [])
        
        for item in items:
            pid = item['pid']
            pname = item['pname']
            tid = item['team_id']
            
            if 'opp_id' not in item:
                 continue
                 
            opp_id = item['opp_id']
            is_home = item['is_home']
            
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
                'SEASON_YEAR': 2026,
                'SEASON_ID': 22025,
                'TEAM_ABBREVIATION': own_abbr,
                'OPP_TEAM_ABBREVIATION': opp_abbr
            }
            phantom_rows.append(new_row)
            
        if not phantom_rows:
            return {'predictions': [], 'trixie': None}
            
        df_batch = pd.concat([df_history, pd.DataFrame(phantom_rows)], ignore_index=True)
        
        # 3. Process Features
        from src.injury_service import get_injury_report, load_injury_cache
        injury_map = get_injury_report(force_refresh=True)
        
        cache_info = load_injury_cache()
        # if cache_info['is_stale'] and log_manager:
        #     await log_manager.broadcast(f"⚠️ Injury data age: {cache_info['age_hours']:.1f}h (may be stale)")
            
        current_season = df_history['SEASON_YEAR'].max()
        roster_df = df_history[df_history['SEASON_YEAR'] == current_season][['TEAM_ID', 'PLAYER_NAME', 'PLAYER_ID']].drop_duplicates()
        
        team_rosters = {}
        for tid, group in roster_df.groupby('TEAM_ID'):
            team_rosters[tid] = dict(zip(group['PLAYER_NAME'], group['PLAYER_ID']))
            
        overrides = {}
        for item in execution_list:
            pid = item['pid']
            tid = item.get('team_id')
            
            derived_missing_ids = []
            if tid in team_rosters and injury_map:
                name_to_id = team_rosters[tid]
                for r_name, r_id in name_to_id.items():
                   if r_id == pid: continue 
                   status = injury_map.get(r_name)
                   if status in ['Out', 'Doubtful']:
                       derived_missing_ids.append(r_id)
            
            missing_ids_str = "_".join(map(str, derived_missing_ids)) if derived_missing_ids else "NONE"
            stars_cnt = len(derived_missing_ids)
            
            overrides[(pid, target_date)] = {
                'INJURY_REPORT': injury_map,
                'MISSING_PLAYER_IDS': missing_ids_str,
                'STARS_OUT': stars_cnt
            }
            
        df_processed = self.fe.process(df_batch, is_training=False, overrides=overrides)
        
        # if log_manager: await log_manager.broadcast("✅ Injuries checked. Analyzing Context...")

        self.load_common_resources()
        
        # 4. Predict Loops
        target_dt = pd.to_datetime(target_date)
        today_data = df_processed[df_processed['GAME_DATE'] == target_dt].copy()
        
        if not today_data.empty and 'OPP_TEAM_ABBREVIATION' in today_data.columns:
            opp_stats = today_data.groupby('OPP_TEAM_ABBREVIATION')[['OPP_ROLL_PACE', 'OPP_ROLL_DEF_RTG']].first()
            pace_percentiles = opp_stats['OPP_ROLL_PACE'].rank(pct=True).to_dict()
            def_rating_percentiles = opp_stats['OPP_ROLL_DEF_RTG'].rank(pct=True).to_dict()
            elite_defenses = set(opp_stats.nsmallest(5, 'OPP_ROLL_DEF_RTG').index.tolist())
        else:
            pace_percentiles = {}
            def_rating_percentiles = {}
            elite_defenses = set()
            
        df_history['GAME_DATE'] = pd.to_datetime(df_history['GAME_DATE'])
        last_played_map = df_history.groupby('PLAYER_ID')['GAME_DATE'].max().to_dict()
        
        # --- STAGE 1: PREDICT MINUTES ---
        if log_manager:
            active_count = len(today_data['PLAYER_ID'].unique())
            await log_manager.broadcast(f"⏳ Predicting minutes for {active_count} active players...")
        
        minutes_map = {} # pid -> (pred_minutes, min_context)
        
        for pid in today_data['PLAYER_ID'].unique():
            player_row = today_data[today_data['PLAYER_ID'] == pid]
            if player_row.empty: continue
            
            pname = player_row['PLAYER_NAME'].iloc[0]
            
            injury_status = "Active"
            if pname in injury_map:
                injury_status = injury_map[pname]
            else:
                for k, v in injury_map.items():
                    if pname in k:
                        injury_status = v
                        break
            
            if injury_status == "Out":
                continue
                
            last_date = last_played_map.get(pid)
            if last_date:
                days_inactive = (target_dt - last_date).days
                if days_inactive > 30:
                    continue
            
            min_context = {
                'recent_min_avg': float(player_row['recent_min_avg_RAW'].iloc[0]),
                'recent_min_std': float(player_row['recent_min_std_RAW'].iloc[0]),
                'INJURY_SEVERITY_TOTAL': float(player_row['INJURY_SEVERITY_TOTAL_RAW'].iloc[0]),
                'STARS_OUT': float(player_row['STARS_OUT_RAW'].iloc[0]),
                'OPP_ROLL_PACE': float(player_row['OPP_ROLL_PACE_RAW'].iloc[0]),
                'IS_HOME': float(player_row['IS_HOME_RAW'].iloc[0]),
                'DAYS_REST': float(player_row['DAYS_REST_RAW'].iloc[0]),
                'season_min_avg': float(player_row['season_min_avg_RAW'].iloc[0])
            }
            
            pred_minutes = self.minutes_predictor.predict(min_context)
            pred_minutes = max(5.0, min(48.0, pred_minutes))
            
            minutes_map[pid] = (pred_minutes, min_context, injury_status)


        # --- STAGE 2: PREDICT STATS & COMBINE ---
        if log_manager: await log_manager.broadcast("⏳ Calculating final predictions...")
        
        odds_data = fetch_nba_odds()
        results_list = []
        
        for pid in today_data['PLAYER_ID'].unique():
            if pid not in minutes_map: continue
            
            player_row = today_data[today_data['PLAYER_ID'] == pid]
            pname = player_row['PLAYER_NAME'].iloc[0]
            
            pred_minutes, min_context, injury_status = minutes_map[pid]
            
            model, model_path = self._get_model(pid)
            
            p_idx = torch.LongTensor(player_row['PLAYER_IDX'].values).to(device)
            t_idx = torch.LongTensor(player_row['TEAM_IDX'].values).to(device)
            x_cont = torch.FloatTensor(player_row[self.resources['feature_cols']].values).to(device)
            
            m_indices = [len(self.resources['p_enc'].classes_)] * 3
            m_idx_tensor = torch.LongTensor([m_indices] * len(player_row)).to(device)
            
            with torch.no_grad():
                preds = model(p_idx, t_idx, x_cont, m_idx_tensor)
                
            preds_np = preds.cpu().numpy()
            
            if preds_np.shape[1] == 12:
                pm_pts = float(preds_np[0, 0])
                pm_reb = float(preds_np[0, 1])
                pm_ast = float(preds_np[0, 2])
                pm_3pm = float(preds_np[0, 3])
                pm_blk = float(preds_np[0, 4])
                pm_stl = float(preds_np[0, 5])
                
                std_pts = float(np.exp(0.5 * preds_np[0, 6]))
                std_reb = float(np.exp(0.5 * preds_np[0, 7]))
                std_ast = float(np.exp(0.5 * preds_np[0, 8]))
                std_3pm = float(np.exp(0.5 * preds_np[0, 9]))
                std_blk = float(np.exp(0.5 * preds_np[0, 10]))
                std_stl = float(np.exp(0.5 * preds_np[0, 11]))
            else:
                pm_pts, pm_reb, pm_ast = 0,0,0
                pm_3pm, pm_blk, pm_stl = 0,0,0
                std_pts = 1.0 
            
            pred_pts = pm_pts * pred_minutes
            pred_reb = pm_reb * pred_minutes
            pred_ast = pm_ast * pred_minutes
            pred_3pm = pm_3pm * pred_minutes
            pred_blk = pm_blk * pred_minutes
            pred_stl = pm_stl * pred_minutes
            
            mae_pts = std_pts * pred_minutes
            mae_reb = std_reb * pred_minutes
            mae_ast = std_ast * pred_minutes
            mae_3pm = std_3pm * pred_minutes
            mae_blk = std_blk * pred_minutes
            mae_stl = std_stl * pred_minutes
            mae_pra = mae_pts + mae_reb + mae_ast

            # Calibration
            stat_cals = {
                'PTS': self.calibration_factors.get('global', {}).get('PTS', 1.0),
                'REB': self.calibration_factors.get('global', {}).get('REB', 1.0),
                'AST': self.calibration_factors.get('global', {}).get('AST', 1.0),
                '3PM': self.calibration_factors.get('global', {}).get('3PM', 1.0),
                'BLK': self.calibration_factors.get('global', {}).get('BLK', 1.0),
                'STL': self.calibration_factors.get('global', {}).get('STL', 1.0)
            }
            p_cals = self.calibration_factors.get('players', {}).get(str(pid), {})
            for k in stat_cals:
                if k in p_cals: stat_cals[k] *= p_cals[k]
            
            pred_pts *= stat_cals['PTS']
            pred_reb *= stat_cals['REB']
            pred_ast *= stat_cals['AST']
            pred_3pm *= stat_cals['3PM']
            pred_blk *= stat_cals['BLK']
            pred_stl *= stat_cals['STL']

            # Vegas Adjustments
            game_total = 218
            final_adjustment = 1.0
            env_score_normalized = 5.0
            is_elite_def = False
            
            team_abbrev = player_row['TEAM_ABBREVIATION'].iloc[0] if 'TEAM_ABBREVIATION' in player_row.columns else None
            opp_abbr = player_row['OPP_TEAM_ABBREVIATION'].iloc[0] if 'OPP_TEAM_ABBREVIATION' in player_row.columns else None
            
            if team_abbrev:
                game_total_info = get_game_total_for_matchup(team_abbrev, odds_data)
                
                if game_total_info.get('found'):
                    game_total = game_total_info['total']
                    deviation = (game_total - 218) / 10
                    max_cap = 0.25 if (game_total <= 205 or game_total >= 235) else 0.15
                    base_adjustment = 1.0 + (deviation * 0.05)
                    base_adjustment = max(1.0 - max_cap, min(1.0 + max_cap, base_adjustment))
                    
                    pace_pct = pace_percentiles.get(opp_abbr, 0.5)
                    pace_multiplier = 0.85 + (pace_pct * 0.30)
                    pace_adjusted = max(0.75, min(1.35, base_adjustment * pace_multiplier))
                    
                    injury_severity = float(player_row['INJURY_SEVERITY_TOTAL'].iloc[0])
                    injury_dampen = 1.0 - min(injury_severity * 0.08, 0.25)
                    injury_adjusted = 1.0 + ((pace_adjusted - 1.0) * injury_dampen)
                    
                    is_elite_def = opp_abbr in elite_defenses
                    def_pct = def_rating_percentiles.get(opp_abbr, 0.5)
                    def_penalty = 0.92 + (def_pct * 0.08) if is_elite_def else 1.0
                    
                    final_adjustment = injury_adjusted * def_penalty
                    final_adjustment = max(0.75, min(1.35, final_adjustment))
                    
                    raw_env = ((game_total - 218) / 10) + (pace_pct - 0.5) * 2 + (def_pct - 0.5) * 1.5 + (injury_severity * -0.5)
                    env_score_normalized = max(0.0, min(10.0, 5.0 + raw_env))
            
            pred_pts *= final_adjustment
            pred_reb *= final_adjustment
            pred_ast *= final_adjustment
            pred_3pm *= final_adjustment
            pred_blk *= final_adjustment
            pred_stl *= final_adjustment
            
            pred_pra = pred_pts + pred_reb + pred_ast
            pred_pr = pred_pts + pred_reb
            pred_pa = pred_pts + pred_ast
            pred_ra = pred_reb + pred_ast
            
            if pred_pra < 1.0: continue

            last_5 = self._get_last_5(pid, df_history, target_dt)
            season_stats = self._get_season_stats(pid, df_history)
            
            model_metadata = self.metadata_cache.get_metadata(pid)
            
            # Convert internal status to UI-friendly status
            if model_metadata:
                if model_metadata.get('status') == 'failed':
                    model_status = "FAILED"
                else:
                    is_fresh = self.metadata_cache.is_fresh(pid, max_age_hours=24.0)
                    model_status = "FRESH" if is_fresh else "CACHED"
                
                model_reason = model_metadata.get('trigger_reason', 'none')
                model_last_trained = model_metadata.get('last_updated', 'never')
            else:
                model_status = "UNKNOWN"
                model_reason = "none"
                model_last_trained = "never"

            from src.bet_sizing import calculate_bet_quality
            props = [
                ('PTS', pred_pts, mae_pts, pred_pts),
                ('REB', pred_reb, mae_reb, pred_reb),
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
            
            for p_name, val, mae, _ in props:
                calc = calculate_bet_quality(p_name, val, mae, val, last_5)
                if calc['units'] > highest_score:
                    highest_score = calc['units']
                    best_prop = calc
                    
            if best_prop is None:
                 best_prop = {'prop': 'PTS', 'val': pred_pts, 'line_low': pred_pts-mae_pts, 'line_high': pred_pts+mae_pts, 'units': 0.0, 'badge': "NO BET", 'consistency': 0.5, 'mae': mae_pts}

            res = {
                'PLAYER_ID': int(float(pid)),
                'PLAYER_NAME': str(pname),
                'GAME_DATE': target_date,
                'MATCHUP': str(player_row['MATCHUP'].iloc[0]),
                'OPPONENT': str(player_row['OPP_TEAM_ABBREVIATION'].iloc[0]),
                'IS_HOME': bool(self._get_home_status(player_row)),
                
                'PRED_MIN': float(pred_minutes),
                'SEASON_AVG_MIN': float(min_context['season_min_avg']),
                'MIN_DELTA': float(pred_minutes - min_context['season_min_avg']),
                'MIN_DELTA_PCT': float((pred_minutes - min_context['season_min_avg']) / max(min_context['season_min_avg'], 1e-3)),
                
                'PRED_PTS_PER_MIN': float(pm_pts),
                'PRED_REB_PER_MIN': float(pm_reb),
                'PRED_AST_PER_MIN': float(pm_ast),
                'PRED_PRA_PER_MIN': float(pm_pts + pm_reb + pm_ast),
                'PRED_3PM_PER_MIN': float(pm_3pm),
                'PRED_BLK_PER_MIN': float(pm_blk),
                'PRED_STL_PER_MIN': float(pm_stl),
                
                'INJURY_STATUS': injury_status,
                'MISSING_TEAMMATES': str(player_row['MISSING_PLAYER_IDS'].iloc[0]),
                'INJURY_IMPACT': bool(min_context['INJURY_SEVERITY_TOTAL'] > 0),
                'OPPORTUNITY_SCORE': float(player_row['OPP_SCORE'].iloc[0]),
                'USAGE_BOOST': float(player_row['OPP_USG_BOOST'].iloc[0]) if 'OPP_USG_BOOST' in player_row.columns else 0.0,
                
                'MODEL_STATUS': model_status,
                'MODEL_LAST_TRAINED': model_last_trained,
                'MODEL_TRIGGER_REASON': model_reason,
                'MODEL_TYPE': 'Specific' if 'global' not in os.path.basename(model_path) else 'Global',

                'PRED_PTS': float(pred_pts), 'PRED_REB': float(pred_reb), 'PRED_AST': float(pred_ast), 'PRED_PRA': float(pred_pra),
                'PRED_PR': float(pred_pr), 'PRED_PA': float(pred_pa), 'PRED_RA': float(pred_ra),
                'PRED_3PM': float(pred_3pm), 'PRED_BLK': float(pred_blk), 'PRED_STL': float(pred_stl),
                
                'SEASON_PTS': float(season_stats.get('pts', 0)),
                'SEASON_REB': float(season_stats.get('reb', 0)),
                'SEASON_AST': float(season_stats.get('ast', 0)),
                'SEASON_3PM': float(season_stats.get('3pm', 0)),
                'SEASON_PRA': float(season_stats.get('pra', 0)),
                
                'MAE_PTS': float(mae_pts), 'MAE_REB': float(mae_reb), 'MAE_AST': float(mae_ast), 'MAE_PRA': float(mae_pra),
                'MAE_3PM': float(mae_3pm), 'MAE_BLK': float(mae_blk), 'MAE_STL': float(mae_stl),
                
                'LINE_PTS_LOW': float(pred_pts - mae_pts), 'LINE_PTS_HIGH': float(pred_pts + mae_pts),
                'LINE_REB_LOW': float(pred_reb - mae_reb), 'LINE_REB_HIGH': float(pred_reb + mae_reb),
                'LINE_AST_LOW': float(pred_ast - mae_ast), 'LINE_AST_HIGH': float(pred_ast + mae_ast),
                'LINE_PRA_LOW': float(pred_pra - mae_pra), 'LINE_PRA_HIGH': float(pred_pra + mae_pra),
                
                'BEST_PROP': best_prop['prop'],
                'BEST_VAL': best_prop['val'],
                'LINE_LOW': best_prop['line_low'],
                'LINE_HIGH': best_prop['line_high'],
                'UNITS': best_prop['units'],
                'BADGE': best_prop['badge'],
                'CONSISTENCY': best_prop['consistency'],
                
                'LAST_5': last_5,
                
                'VEGAS_TOTAL': game_total,
                'VEGAS_ADJUSTMENT_FINAL': round(final_adjustment, 3) if final_adjustment != 1.0 else None,
                'VEGAS_ENV_SCORE': round(env_score_normalized, 1),
                'PACE_PERCENTILE': round(pace_percentiles.get(opp_abbr, 0.5), 2),
                'IS_ELITE_DEFENSE': is_elite_def,
                
                'MODEL_STATUS': model_status,
                'MODEL_TRIGGER_REASON': model_reason,
                'MODEL_LAST_TRAINED': model_last_trained,
                
                'CALIBRATION_APPLIED': True,
                'CALIBRATION_factors': stat_cals
            }
            results_list.append(res)
            
            self.tracker.save_prediction(pid, pname, 'PTS', pred_pts, game_date=target_date, calibration_applied=True)
            self.tracker.save_prediction(pid, pname, 'REB', pred_reb, game_date=target_date, calibration_applied=True)
            self.tracker.save_prediction(pid, pname, 'AST', pred_ast, game_date=target_date, calibration_applied=True)
            self.tracker.save_prediction(pid, pname, '3PM', pred_3pm, game_date=target_date, calibration_applied=True)
            self.tracker.save_prediction(pid, pname, 'BLK', pred_blk, game_date=target_date, calibration_applied=True)
            self.tracker.save_prediction(pid, pname, 'STL', pred_stl, game_date=target_date, calibration_applied=True)
        
        # if log_manager: await log_manager.broadcast("Fetching market odds for final analysis...")
        
        for res in results_list:
            res['PROPS_ANALYSIS'] = {}
            stats_to_analyze = [
                ('PTS', 'PRED_PTS', 'MAE_PTS'),
                ('REB', 'PRED_REB', 'MAE_REB'),
                ('AST', 'PRED_AST', 'MAE_AST'),
                ('3PM', 'PRED_3PM', 'MAE_3PM'),
                ('BLK', 'PRED_BLK', 'MAE_BLK'),
                ('STL', 'PRED_STL', 'MAE_STL'),
                ('RA', 'PRED_RA', 'MAE_REB'),
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
                        std_dev=mae_val, 
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
                        
                        if stat_code == res['BEST_PROP']:
                            res['MARKET_LINE'] = ev_result.get('market_line')
                            recommendation = ev_result.get('recommendation')
                            if recommendation == 'UNDER':
                                final_ev = ev_result.get('ev_under', 0)
                                res['MARKET_ODDS'] = ev_result.get('odds_under')
                            else:
                                final_ev = ev_result.get('ev_over', 0)
                                res['MARKET_ODDS'] = ev_result.get('odds_over')
                            res['EV'] = round(final_ev * 100, 1)
                            res['EV_RECOMMENDATION'] = recommendation
                except:
                    pass

        # Smart Trixie Logic
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
        candidate_sgps = [] 
        
        for p in sorted_res:
            good_bets = []
            stats_list = ['PTS', 'REB', 'AST', '3PM', 'STL', 'RA', 'PR', 'PA', 'PRA']
            
            for stat in stats_list:
                analysis = p.get('PROPS_ANALYSIS', {}).get(stat)
                if not analysis: continue
                
                rec = analysis.get('recommendation')
                if rec == 'PASS' or not rec: continue
                
                prob = analysis.get('p_over', 0)
                if rec == 'UNDER':
                    prob = 100 - prob 
                    odds = analysis.get('odds_under')
                else:
                    odds = analysis.get('odds_over')
                
                if prob < 55.0: continue
                if american_to_decimal(odds) < 1.50: continue
                
                market_line = analysis.get('market_line')
                
                good_bets.append({
                    'prop': stat,
                    'direction': rec,
                    'line': market_line,
                    'prob': prob / 100.0,
                    'odds_amer': odds,
                    'odds_dec': american_to_decimal(odds),
                    'desc': f"{stat} {rec[0]} {market_line}",
                    'prediction': p.get(f'PRED_{stat}'),
                    'matchup': p.get('MATCHUP')
                })
                
            if len(good_bets) < 2: continue
                
            best_sgp = None
            best_sgp_score = -1
            
            for r in [2, 3]:
                for combo in combinations(good_bets, r):
                    joint_prob = 1.0
                    combined_odds = 1.0
                    safe_legs = 0
                    
                    for leg in combo:
                        joint_prob *= leg['prob']
                        combined_odds *= leg['odds_dec']
                        if leg['prob'] > 0.60: safe_legs += 1
                        
                    joint_prob *= 0.85
                    combined_odds *= 0.85 
                    
                    if combined_odds < 2.0: continue
                        
                    ev = (joint_prob * combined_odds) - 1
                    score = ev * (joint_prob * 10) 
                    
                    directions = [leg['direction'] for leg in combo]
                    if all(d == 'UNDER' for d in directions): score *= 1.30 
                    elif all(d == 'OVER' for d in directions): score *= 1.15

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
                    'matchup': p.get('MATCHUP')
                })
        
        candidate_sgps.sort(key=lambda x: x['score'], reverse=True)
        
        seen_matchups = set()
        primary_candidates = []
        for cand in candidate_sgps:
            if len(primary_candidates) >= 3: break
            m = cand['matchup']
            if m not in seen_matchups:
                primary_candidates.append(cand)
                seen_matchups.add(m)
        
        secondary_candidates = []
        for cand in candidate_sgps:
            if cand in primary_candidates: continue
            if len(secondary_candidates) >= 3: break
            m = cand['matchup']
            if m not in seen_matchups:
                 secondary_candidates.append(cand)
                 seen_matchups.add(m)
                 
        alternates = []
        for cand in candidate_sgps:
            if cand in primary_candidates or cand in secondary_candidates: continue
            if len(alternates) >= 2: break
            m = cand['matchup']
            if m not in seen_matchups:
                alternates.append(cand)
                seen_matchups.add(m)
        
        if len(alternates) < 2:
            for cand in candidate_sgps:
                if cand in primary_candidates or cand in secondary_candidates or cand in alternates: continue
                if len(alternates) >= 2: break
                alternates.append(cand)

        def format_sgp_legs(candidates):
            formatted_list = []
            for item in candidates:
                p = item['player']
                sgp = item['sgp']
                prob = sgp['joint_prob']
                odds = sgp['total_odds']
                
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
                primary_joint_prob *= (leg['win_prob'] / 100.0)
                
            trixie_units = 0.1
            if primary_joint_prob * primary_total_odds > 1.2: 
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
        
        if log_manager: 
            final_count = len(sorted_res)
            await log_manager.broadcast(f"✅ Analysis complete. {final_count} predictions generated.")
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

    def _get_home_status(self, row):
        if 'IS_HOME' in row.columns:
             return row['IS_HOME'].iloc[0]
        matchup = row['MATCHUP'].iloc[0]
        return 'vs.' in matchup


if __name__ == "__main__":
    import asyncio
    from src.data_fetch import fetch_daily_scoreboard
    from nba_api.stats.endpoints import commonteamroster

    async def run_manual_batch():
        print("🚀 Starting Batch Prediction from CLI...")
        bp = BatchPredictor()
        bp.load_common_resources()
        
        target_date = datetime.now().strftime('%Y-%m-%d')
        print(f"🏀 Fetching daily scoreboard for {target_date}...")
        scoreboard = fetch_daily_scoreboard(target_date)
        
        if scoreboard.empty:
            print("⚠️ No games found for today.")
            return

        print(f"📋 Building execution list for {len(scoreboard)} games...")
        team_schedule = {} 
        for _, row in scoreboard.iterrows():
            h = row['HOME_TEAM_ID']
            v = row['VISITOR_TEAM_ID']
            team_schedule[h] = {'opp_id': v, 'is_home': True}
            team_schedule[v] = {'opp_id': h, 'is_home': False}
            
        team_ids = list(team_schedule.keys())
        execution_list = []
        current_season = "2025-26"
        
        for tid in team_ids:
            try:
                print(f"  - Fetching roster for team {tid}...")
                # nba_api is synchronous here, we'll wrap if needed but for CLI it's fine
                roster = commonteamroster.CommonTeamRoster(team_id=tid, season=current_season).get_data_frames()[0]
                ctx = team_schedule.get(tid, {})
                for _, player in roster.iterrows():
                    execution_list.append({
                        'pid': player['PLAYER_ID'],
                        'pname': player['PLAYER'],
                        'team_id': tid,
                        'opp_id': ctx.get('opp_id'),
                        'is_home': ctx.get('is_home')
                    })
            except Exception as e:
                print(f"  ⚠️ Failed for team {tid}: {e}")

        if not execution_list:
            print("⚠️ No players identified for processing.")
            return

        print(f"📊 Analyzing {len(execution_list)} players...")
        results = await bp.analyze_today_batch(execution_list=execution_list)
        
        print(f"✅ Analysis complete. Generated {len(results.get('predictions', []))} predictions.")
        if results.get('trixie'):
             print(f"💎 Trixie/SGP Recommendations generated.")

    asyncio.run(run_manual_batch())
