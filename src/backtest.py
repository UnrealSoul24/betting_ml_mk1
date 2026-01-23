
import pandas as pd
import numpy as np
import joblib
import os
import torch
from src.feature_engineer import FeatureEngineer
from src.train_models import NBAPredictor
from datetime import timedelta

class Backtester:
    def __init__(self):
        self.fe = FeatureEngineer()
        self.models_dir = os.path.join(os.path.dirname(__file__), '../data/models')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.resources = {}
        self._load_resources()
        
    def _load_resources(self):
        print("Loading resources...")
        self.resources['p_enc'] = joblib.load(os.path.join(self.models_dir, 'player_encoder.joblib'))
        self.resources['t_enc'] = joblib.load(os.path.join(self.models_dir, 'team_encoder.joblib'))
        self.resources['scaler'] = joblib.load(os.path.join(self.models_dir, 'scaler.joblib'))
        self.resources['feature_cols'] = joblib.load(os.path.join(self.models_dir, 'feature_cols.joblib'))

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
        test_df['ERR_PTS'] = test_df['PRED_PTS'] - test_df['PTS']
        mae_pts = test_df['ERR_PTS'].abs().mean()
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

if __name__ == "__main__":
    bt = Backtester()
    bt.run_backtest(days=30)

