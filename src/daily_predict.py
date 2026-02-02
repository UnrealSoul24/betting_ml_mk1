import pandas as pd
import numpy as np
import os
import torch
import joblib
from datetime import datetime
from src.data_fetch import fetch_daily_scoreboard
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor # Import model architecture
from src.injury_service import get_injury_report, load_injury_cache # Step 1: Import

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

def load_resources(target_player_id: int = None):
    # Load Generic Resources
    player_encoder = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
    team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    
    # Init Model
    num_players = len(player_encoder.classes_) + 1
    num_teams = len(team_encoder.classes_)
    num_cont = len(feature_cols)
    model = NBAPredictor(num_players, num_teams, num_cont)
    
    # Load Model Weights
    # 1. Try Specific Model
    if target_player_id:
        specific_path = os.path.join(MODELS_DIR, f'pytorch_player_{target_player_id}.pth')
        if os.path.exists(specific_path):
            print(f"Loading specific model for player {target_player_id}...")
            model.load_state_dict(torch.load(specific_path, map_location=device)) # weights_only=True removed as it's not a standard arg for torch.load
        else:
            print(f"Specific model not found. Loading generic model...")
            generic_path = os.path.join(MODELS_DIR, 'pytorch_nba_global.pth')
            model.load_state_dict(torch.load(generic_path, map_location=device)) # weights_only=True removed
    else:
        print("Loading generic model...")
        generic_path = os.path.join(MODELS_DIR, 'pytorch_nba_global.pth')
        model.load_state_dict(torch.load(generic_path, map_location=device)) # weights_only=True removed
        
    model.to(device)
    model.eval()
    
    return model, player_encoder, team_encoder, scaler, feature_cols

def predict_daily(target_player_id: int = None, date_input: str = None, stars_out: int = None, missing_ids: str = None):
    results_list = []
    try:
        model, p_enc, t_enc, scaler, feature_cols = load_resources(target_player_id)
        fe = FeatureEngineer()
        
        # 1. Get History Data
        # print("Fetching historical data...") # Silence for API
        df_history = fe.load_all_data()
        
        # Step 1: Auto-Fetch Injury Report
        injury_report = get_injury_report(force_refresh=True)
        
        # Step 6: Validation
        from src.injury_service import load_injury_cache # Ensure import if not at top, but added at top
        cache_info = load_injury_cache()
      
        if cache_info['is_stale']:
            print(f"⚠️ WARNING: Injury data is {cache_info['age_hours']:.1f} hours old (stale threshold: 2h)")
            print("   Predictions may not reflect latest injury updates.")
      
        if not injury_report:
            print("❌ CRITICAL: No injury data available. Predictions will lack injury context.")
        else:
            print(f"✓ Loaded {len(injury_report)} injuries for context")

        
        # 2. Determine Target Date & Game Context
        target_date = date_input if date_input else datetime.now().strftime('%Y-%m-%d')
        # print(f"Target Date: {target_date}")
        
        # Fetch Scoreboard
        scoreboard = fetch_daily_scoreboard(target_date)
        
        if scoreboard.empty:
            print(f"No games found for date {target_date}")
            return []

        # Logic to handle "All Players" vs "Single Player"
        # If target_player_id is set, we process just that one.
        # If NOT set, checking ALL players is hard because `load_resources` might load a specific model or generic.
        # For "Today's Bets", we likely want to iterate over ALL players in the scoreboard?
        
        # For now, let's keep the existing logic: If target_player_id is provided, predict for him.
        # The API will handle the iteration and calls.
        
        if target_player_id:
            # Filter history to just this player to find their Team ID
            player_history = df_history[df_history['PLAYER_ID'] == target_player_id].sort_values('GAME_DATE')
            if player_history.empty:
                print(f"Player {target_player_id} not found in history.")
                return []
                
            last_game = player_history.iloc[-1]
            team_id = last_game['TEAM_ID']
            player_name = last_game['PLAYER_NAME']
            
            # Find game in scoreboard
            game = scoreboard[(scoreboard['HOME_TEAM_ID'] == team_id) | (scoreboard['VISITOR_TEAM_ID'] == team_id)]
            
            if game.empty:
                # print(f"No game found for {player_name} (Team {team_id}) on {target_date}.")
                return []
                
            game = game.iloc[0]
            is_home = game['HOME_TEAM_ID'] == team_id
            opp_id = game['VISITOR_TEAM_ID'] if is_home else game['HOME_TEAM_ID']
            
            # Get Team Abbreviations
            from nba_api.stats.static import teams
            try:
                opp_info = teams.find_team_name_by_id(opp_id)
                opp_abbr = opp_info['abbreviation']
                
                own_info = teams.find_team_name_by_id(team_id)
                own_abbr = own_info['abbreviation']
            except:
                opp_abbr = "UNK"
                own_abbr = "UNK"
                
            # print(f"Matchup: {player_name} vs {opp_abbr} ({'Home' if is_home else 'Away'})")
            
            # Create Phantom Row
            new_row = {
                'PLAYER_ID': target_player_id,
                'GAME_DATE': target_date,
                'MATCHUP': f"{own_abbr} vs. {opp_abbr}",
                'TEAM_ID': team_id,
                'PTS': np.nan, 'REB': np.nan, 'AST': np.nan,
                'MIN': 0, 'FGA': 0,
                'SEASON_YEAR': 2026
            }
            
            df_history = pd.concat([df_history, pd.DataFrame([new_row])], ignore_index=True)
            
            # 3. Process
            # print(f"Processing features...")
            
            # Step 2: Pass Injury Report in Overrides
            
            # --- DERIVE MISSING IDs for Target Row ---
            # 1. Get Roster for Team ID (Name -> ID)
            # Use recent history (this season) to ensure roster is fresh
            current_season = df_history['SEASON_YEAR'].max()
            team_roster = df_history[
                (df_history['TEAM_ID'] == team_id) & 
                (df_history['SEASON_YEAR'] == current_season)
            ][['PLAYER_NAME', 'PLAYER_ID']].drop_duplicates()
            
            name_to_id = dict(zip(team_roster['PLAYER_NAME'], team_roster['PLAYER_ID']))
            
            derived_missing_ids = []
            
            # 2. Check Injury Report
            if injury_report:
                for p_name, status in injury_report.items():
                    if status in ['Out', 'Doubtful']: # Filter severity
                        # Try exact match
                        mid = name_to_id.get(p_name)
                        if not mid:
                            # Try fuzzy/partial match if needed?
                            # For safety, simplistic check:
                            for r_name, r_id in name_to_id.items():
                                if p_name in r_name or r_name in p_name: # "Giannis" in "Giannis Antetokounmpo"
                                    mid = r_id
                                    break
                                    
                        if mid and mid != target_player_id:
                            derived_missing_ids.append(mid)
                            
            missing_ids_str = "_".join(map(str, derived_missing_ids)) if derived_missing_ids else "NONE"
            stars_out_count = len(derived_missing_ids) # Simplistic "Stars" count (User asked for derived param)
            # In reality "Stars" filter usually applies to Top Usage, but here we count all Out Rotation players?
            # Let's count all derived missing as "Stars Out" for the override metric to be safe/consistent.
            
            print(f"Derived Missing Teammates: {missing_ids_str} (Count: {stars_out_count})")
            # ----------------------------------------
            
            overrides = {}
            if target_player_id and target_date:
                overrides[(target_player_id, target_date)] = {
                    'INJURY_REPORT': injury_report,
                    'MISSING_PLAYER_IDS': missing_ids_str,
                    'STARS_OUT': stars_out_count
                }
                # Preserve manual overrides if explicitly provided (CLI high priority)
                if missing_ids is not None:
                     overrides[(target_player_id, target_date)]['MISSING_PLAYER_IDS'] = missing_ids
                if stars_out is not None:
                    overrides[(target_player_id, target_date)]['STARS_OUT'] = stars_out
            
            df_processed = fe.process(df_history, is_training=False, overrides=overrides)
            
            # Step 3: Log Injury Context
            if target_player_id and target_date:
                target_dt = pd.to_datetime(target_date)
                mask = (df_processed['PLAYER_ID'] == target_player_id) & (df_processed['GAME_DATE'] == target_dt)
                if mask.any():
                    latest_stats_check = df_processed[mask]
                    if not latest_stats_check.empty:
                        missing_ids_str = latest_stats_check['MISSING_PLAYER_IDS'].iloc[0]
                        opp_score = latest_stats_check.get('OPP_SCORE', pd.Series([0.0])).iloc[0]
                        injury_severity = latest_stats_check.get('INJURY_SEVERITY_TOTAL', pd.Series([0.0])).iloc[0]
                        usg_boost = latest_stats_check.get('OPP_USG_BOOST', pd.Series([0.0])).iloc[0]
                        
                        if missing_ids_str and missing_ids_str != "NONE":
                            print(f"\n🏥 Injury Context for {player_name}:")
                            print(f"   Missing Teammates: {missing_ids_str.replace('_', ', ')}")
                            print(f"   Injury Severity Weight: {injury_severity:.2f}")
                            print(f"   Opportunity Score: {opp_score:+.2f}")
                            print(f"   Expected Usage Boost: {usg_boost:+.1f}%")

            
            # 4. Extract Target Row
            target_dt = pd.to_datetime(target_date)
            mask = (df_processed['PLAYER_ID'] == target_player_id) & (df_processed['GAME_DATE'] == target_dt)
            latest_stats = df_processed[mask].copy()
            
            if latest_stats.empty:
                return []
    
            # 5. Predict
            p_idx = torch.LongTensor(latest_stats['PLAYER_IDX'].values).to(device)
            t_idx = torch.LongTensor(latest_stats['TEAM_IDX'].values).to(device)
            x_cont = torch.FloatTensor(latest_stats[feature_cols].values).to(device)
            
            pad_idx = len(p_enc.classes_)
            max_missing = 3
            m_indices = []
            
            if missing_ids:
                ids = missing_ids.split('_')
                for pid_str in ids:
                    if pid_str in p_enc.classes_:
                        encoded = p_enc.transform([pid_str])[0]
                        m_indices.append(encoded)
                        
            if len(m_indices) > max_missing:
                m_indices = m_indices[:max_missing]
            else:
                m_indices += [pad_idx] * (max_missing - len(m_indices))
                
            batch_size = len(latest_stats)
            m_idx_tensor = torch.LongTensor([m_indices] * batch_size).to(device)
            
            with torch.no_grad():
                preds = model(p_idx, t_idx, x_cont, m_idx_tensor)
                
            preds_np = preds.cpu().numpy()
            
            # Build Result Dict
            # Load Metrics
            mae_pts = 6.0
            mae_reb = 2.5
            mae_ast = 2.0
            
            try:
                import json
                # Try specific metrics
                m_path = os.path.join(MODELS_DIR, f'metrics_player_{target_player_id}.json')
                if not os.path.exists(m_path):
                     m_path = os.path.join(MODELS_DIR, 'metrics_global.json')
                     
                if os.path.exists(m_path):
                    with open(m_path, 'r') as f:
                        metrics = json.load(f)
                        mae_pts = metrics.get('mae_pts', 6.0)
                        mae_reb = metrics.get('mae_reb', 2.5)
                        mae_ast = metrics.get('mae_ast', 2.0)
            except:
                pass

            res = latest_stats[['PLAYER_ID', 'GAME_DATE', 'MATCHUP']].iloc[0].to_dict()
            res['PLAYER_NAME'] = player_name
            
            # Base
            p_pts = float(preds_np[0, 0])
            p_reb = float(preds_np[0, 1])
            p_ast = float(preds_np[0, 2])
            
            res['PRED_PTS'] = p_pts
            res['PRED_REB'] = p_reb
            res['PRED_AST'] = p_ast
            
            # Combos
            res['PRED_PRA'] = p_pts + p_reb + p_ast
            res['PRED_PR'] = p_pts + p_reb
            res['PRED_PA'] = p_pts + p_ast
            res['PRED_RA'] = p_reb + p_ast
            
            # MAEs
            mae_pra = mae_pts + mae_reb + mae_ast
            mae_pr = mae_pts + mae_reb
            mae_pa = mae_pts + mae_ast
            mae_ra = mae_reb + mae_ast
            
            # Lines
            res['LINE_PTS_LOW'] = p_pts - mae_pts
            res['LINE_PTS_HIGH'] = p_pts + mae_pts
            
            res['LINE_REB_LOW'] = p_reb - mae_reb
            res['LINE_REB_HIGH'] = p_reb + mae_reb
            
            res['LINE_AST_LOW'] = p_ast - mae_ast
            res['LINE_AST_HIGH'] = p_ast + mae_ast
            
            res['LINE_PRA_LOW'] = res['PRED_PRA'] - mae_pra
            res['LINE_PRA_HIGH'] = res['PRED_PRA'] + mae_pra
            
            res['OPPONENT'] = opp_abbr
            res['IS_HOME'] = bool(is_home)
            res['GAME_DATE'] = str(res['GAME_DATE']) # Serialization
            
            print("Prediction Results:")
            print(f" PLAYER_ID  GAME_DATE     MATCHUP  PRED_PTS  PRED_REB  PRED_AST")
            print(f"{target_player_id:>10} {res['GAME_DATE']} {res['MATCHUP']} {p_pts:<6.2f}    {p_reb:<6.2f}    {p_ast:<6.2f}")
            print(f"Combos -> PRA: {res['PRED_PRA']:.2f} | PR: {res['PRED_PR']:.2f} | PA: {res['PRED_PA']:.2f}")
            print("-" * 60)
            print(f"Betting Lines (Safe Zones):")
            print(f"PTS: Under {res['LINE_PTS_HIGH']:.1f} / Over {res['LINE_PTS_LOW']:.1f} (MAE: {mae_pts:.1f})")
            print(f"REB: Under {res['LINE_REB_HIGH']:.1f} / Over {res['LINE_REB_LOW']:.1f} (MAE: {mae_reb:.1f})")
            print(f"AST: Under {res['LINE_AST_HIGH']:.1f} / Over {res['LINE_AST_LOW']:.1f} (MAE: {mae_ast:.1f})")
            print(f"PRA: Under {res['LINE_PRA_HIGH']:.1f} / Over {res['LINE_PRA_LOW']:.1f}")
            
            results_list.append(res)
            
            # Legacy CSV Save (Optional, can remove if interfering)
            # out_path = os.path.join(DATA_DIR, f'predictions_{target_player_id}_{datetime.now().strftime("%Y%m%d")}.csv')
            # pd.DataFrame([res]).to_csv(out_path, index=False)
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        
    return results_list

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--player_id', type=int, help='ID of player to predict', default=None)
    parser.add_argument('--date', type=str, help='Date to predict (YYYY-MM-DD)', default=None)
    parser.add_argument('--stars_out', type=int, 
                        help='[DEPRECATED] Manual override for missing stars. Injury report now auto-detected.', 
                        default=None)
    parser.add_argument('--missing_ids', type=str, help='Underscore separated list of missing player IDs', default=None)
    args = parser.parse_args()
    
    if args.player_id:
        predict_daily(args.player_id, args.date, args.stars_out, args.missing_ids)
    else:
        # Interactive
        try:
            pid = input("Enter Player ID to predict (or Press Enter for all): ")
            date_in = input("Enter Date (YYYY-MM-DD) [Default: Today]: ")
            print("Note: Injury report is now automatically fetched. Manual 'stars_out' is optional override.")
            stars = input("Stars Out Override (0, 1, 2) [Default: Auto-detect]: ")
            missing = input("Missing Player IDs (e.g., 123_456) [Default: None]: ")
            
            p_val = int(pid) if pid.strip() else None
            d_val = date_in.strip() if date_in.strip() else None
            s_val = int(stars) if stars.strip() else None
            
            if p_val:
                predict_daily(p_val, d_val, s_val)
        except:
            pass
