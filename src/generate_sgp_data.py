import pandas as pd
import numpy as np
import torch
import os
import sys
import joblib
from datetime import datetime

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor, gaussian_nll_loss
from tqdm import tqdm

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
LOGS_DIR = os.path.join(DATA_DIR, 'raw')
OUTPUT_FILE = os.path.join(DATA_DIR, 'processed', 'sgp_training_data.csv')

def load_global_model():
    """Loads the Global Pytorch Model"""
    # Load resources
    try:
        p_enc = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib'))
        t_enc = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
        feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None, None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    num_players = len(p_enc.classes_) + 1
    num_teams = len(t_enc.classes_)
    num_cont = len(feature_cols)
    
    model = NBAPredictor(num_players, num_teams, num_cont)
    
    # Load Weights (Global vNext)
    model_path = os.path.join(MODELS_DIR, 'pytorch_nba_global_vNext.pth')
    if not os.path.exists(model_path):
        print("Global model not found. Please train it first.")
        return None, None, None, None
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, p_enc, t_enc, feature_cols

def generate_data():
    print("Initializing Data Generation...")
    
    # 1. Load Data
    fe = FeatureEngineer()
    # Load all data (FeatureEngineer filters for 2025+ by default now)
    df = fe.load_all_data() 
    
    # Filter for just current season (2025-26 -> SEASON_YEAR 2026) for Meta Training
    # We want to simulate "Active Season" behavior.
    df = df[df['SEASON_YEAR'] == 2026].copy()
    print(f"Loaded {len(df)} rows for simulation (2025-26 Season).")
    
    # 2. Process Features
    print("Processing features...")
    df_processed = fe.process(df)
    
    # 3. Load Model
    model, p_enc, t_enc, f_cols = load_global_model()
    if not model: return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = []
    
    # 4. Simulation Loop
    print("Simulating predictions...")
    
    # Group by Game/Date to simulate "Batch" processing? 
    # Actually we just need row-by-row prediction.
    
    dataset_tensor = torch.FloatTensor(df_processed[f_cols].values).to(device)
    p_idx_tensor = torch.LongTensor(df_processed['PLAYER_IDX'].values).to(device)
    t_idx_tensor = torch.LongTensor(df_processed['TEAM_IDX'].values).to(device)
    
    # Missing ID embedding (Standard)
    pad_idx = len(p_enc.classes_)
    m_indices = [pad_idx] * 3 
    m_idx_tensor = torch.LongTensor([m_indices] * len(df_processed)).to(device)

    batch_size = 1024
    with torch.no_grad():
        for i in tqdm(range(0, len(df_processed), batch_size)):
            batch_p = p_idx_tensor[i:i+batch_size]
            batch_t = t_idx_tensor[i:i+batch_size]
            batch_x = dataset_tensor[i:i+batch_size]
            batch_m = m_idx_tensor[i:i+batch_size]
            
            preds = model(batch_p, batch_t, batch_x, batch_m)
            preds_np = preds.cpu().numpy()
            
            # Process Batch
            start_idx = i
            for j in range(len(preds_np)):
                idx = start_idx + j
                row = df_processed.iloc[idx]
                
                # Extract Predictions
                p_out = preds_np[j]
                
                # 6 Targets
                means = p_out[:6]
                logvars = p_out[6:]
                stds = np.exp(0.5 * logvars)
                
                # Stats: PTS, REB, AST, 3PM, BLK, STL
                stats = ['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL']
                
                # Actuals
                actuals = [
                    row['PTS'], row['REB'], row['AST'], 
                    row['FG3M'], row['BLK'], row['STL']
                ]
                
                # Build Row for Meta-Training
                # We want to record: {Stat}_Mean, {Stat}_Std, Is_Hit
                
                meta_row = {}
                
                # Context Features
                meta_row['PLAYER_ID'] = row['PLAYER_ID']
                meta_row['GAME_DATE'] = row['GAME_DATE']
                meta_row['MIN'] = row['MIN'] # Did they play?
                
                # If they didn't play enough, skip? No, model should learn risk.
                
                # For each stat, calculate "Hit" on Safe Line
                # Safe Line = Pred - Margin
                # Margin: 2 for PTS, 1 for others
                
                for k, stat_name in enumerate(stats):
                    pred = max(0, means[k])
                    std = stds[k]
                    actual = actuals[k]
                    
                    margin = 2.0 if stat_name == 'PTS' else 1.0
                    safe_line = max(0, pred - margin)
                    
                    # Target: Did actual exceed safe line?
                    hit = 1 if actual >= safe_line else 0
                    
                    # Store Features
                    meta_row[f'{stat_name}_PRED'] = pred
                    meta_row[f'{stat_name}_STD'] = std
                    meta_row[f'{stat_name}_SAFE'] = safe_line
                    meta_row[f'{stat_name}_ACTUAL'] = actual # For debug/verification
                    meta_row[f'{stat_name}_HIT'] = hit

                results.append(meta_row)
                
    # 5. Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(out_df)} simulation rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_data()
