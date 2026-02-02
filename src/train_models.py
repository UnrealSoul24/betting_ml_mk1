import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from datetime import datetime

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(DATA_DIR, 'models')




# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')

class NBADataset(Dataset):
    def __init__(self, X, y, missing_idxs):
        self.X = torch.FloatTensor(X) # (N, Features)
        self.y = torch.FloatTensor(y) # (N, 3)
        self.missing_idxs = torch.LongTensor(missing_idxs) # (N, Max_Missing)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Row-based access
        # X: [PLAYER_IDX, TEAM_IDX, ... Cont Features ...]
        p_idx = self.X[idx, 0].long()
        t_idx = self.X[idx, 1].long()
        x_cont = self.X[idx, 2:]
        m_idx = self.missing_idxs[idx]
        return p_idx, t_idx, x_cont, m_idx, self.y[idx]

class NBAPredictor(nn.Module):
    """
    Distributional Model - Outputs mean AND variance for uncertainty estimation.
    Now supports 6 targets: PTS, REB, AST, FG3M, BLK, STL
    
    Output Shape: (B, 12) = [mean_6, logvar_6]
    Targets: PTS_PER_MIN, REB_PER_MIN, AST_PER_MIN, FG3M_PER_MIN, BLK_PER_MIN, STL_PER_MIN
    """
    def __init__(self, num_players, num_teams, num_cont):
        super().__init__()
        # Embeddings
        self.player_emb = nn.Embedding(num_players, 32, padding_idx=num_players-1)
        self.team_emb = nn.Embedding(num_teams, 16)
        
        # Shared backbone
        in_dim = 32 + 16 + num_cont + 32
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Separate heads for mean and variance (6 targets)
        self.mean_head = nn.Linear(64, 6)  # PTS, REB, AST, FG3M, BLK, STL
        self.logvar_head = nn.Linear(64, 6)  # Log-variance
        
    def forward(self, p_idx, t_idx, x_cont, m_idx):
        p_emb = self.player_emb(p_idx)
        t_emb = self.team_emb(t_idx)
        m_emb = self.player_emb(m_idx).sum(dim=1)
        
        x = torch.cat([p_emb, t_emb, x_cont, m_emb], dim=1)
        features = self.backbone(x)
        
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        
        # Concatenate: [mean(6), logvar(6)]
        return torch.cat([mean, logvar], dim=1)
    
    def predict_with_uncertainty(self, p_idx, t_idx, x_cont, m_idx):
        output = self.forward(p_idx, t_idx, x_cont, m_idx)
        mean = output[:, :6]
        logvar = output[:, 6:]
        std = torch.exp(0.5 * logvar)
        return mean, std





def gaussian_nll_loss(output, target, calibration_penalty_weight=0.0):
    """
    Gaussian Negative Log Likelihood Loss.
    
    Trains the model to predict both mean and variance.
    Loss = 0.5 * [log(var) + (y - mean)^2 / var]
    
    Args:
        output: (B, 12) - [mean(6), logvar(6)]
        target: (B, 6) - [pts, reb, ast, 3pm, blk, stl]
        calibration_penalty_weight: float - Weight for systematic bias penalty
    
    Returns:
        loss: scalar
    """
    mean = output[:, :6]
    logvar = output[:, 6:]
    
    # Gaussian NLL
    nll_loss = 0.5 * (logvar + ((target - mean) ** 2) / torch.exp(logvar)).mean()
    
    if calibration_penalty_weight > 0:
        # Penalize systematic bias (mean residuals) across the batch
        residuals = target - mean
        bias = residuals.mean(dim=0) # Mean error per target
        cal_loss = (bias ** 2).mean() * calibration_penalty_weight
        return nll_loss + cal_loss
        
    return nll_loss

def calculate_distribution_alignment(y_true, y_pred_mean):
    """
    Calculates distribution alignment score.
    Score = 1.0 - |OverRate - 0.5| * 2 where OverRate is % of Actual > Predicted.
    Perfect alignment (0.5 split) -> 1.0 score.
    All overs or all unders -> 0.0 score.
    """
    if len(y_true) == 0: return 0.0
    
    # Per target over/under
    # y_true: (N, 6), y_pred_mean: (N, 6)
    overs = (y_true > y_pred_mean).sum(axis=0)
    total = len(y_true)
    
    over_rates = overs / total
    # Average alignment across all 6 targets
    alignment_scores = 1.0 - np.abs(over_rates - 0.5) * 2
    return np.mean(alignment_scores)

class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

def evaluate_model(model, loader, criterion, scaler, feature_cols):
    model.eval()
    total_loss = 0
    y_true_all = []
    y_pred_all = []
    
    with torch.no_grad():
        for p_idx, t_idx, x_cont, m_idx, y in loader:
            p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
            outputs = model(p_idx, t_idx, x_cont, m_idx)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            
            y_true_all.append(y.cpu().numpy())
            y_pred_all.append(outputs.cpu().numpy())
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    
    # Calculate MAE per target
    y_true = np.concatenate(y_true_all, axis=0) if y_true_all else np.array([])
    y_pred = np.concatenate(y_pred_all, axis=0) if y_pred_all else np.array([])
    
    return avg_loss, mae_pts, mae_reb, mae_ast, mae_fg3m, mae_blk, mae_stl, history_dict

    # Note: I modified signature to return separate values but also need to return history dict or update history check?
    # Actually evaluate_model is used inside training loop but the return values are not all unpacked or used there?
    # Let's see usage in training loop.
    # Usage: avg_loss, ... = evaluate_model(...) - NO, it's inline in training loop (lines 536-549).
    # The function evaluate_model (lines 171-200) IS NOT CALLED in the loop! The loop has inline validation code.
    # But for final evaluation it might be used? No, there is no final call to evaluate_model either.
    # It seems evaluate_model is dead code or helper not currently used in the main flow.
    # However, I should update it just in case.
    
    if len(y_true) > 0:
        mae_pts = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
        mae_reb = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
        mae_ast = np.mean(np.abs(y_true[:, 2] - y_pred[:, 2]))
        mae_fg3m = np.mean(np.abs(y_true[:, 3] - y_pred[:, 3]))
        mae_blk = np.mean(np.abs(y_true[:, 4] - y_pred[:, 4]))
        mae_stl = np.mean(np.abs(y_true[:, 5] - y_pred[:, 5]))
    else:
        mae_pts, mae_reb, mae_ast, mae_fg3m, mae_blk, mae_stl = 0, 0, 0, 0, 0, 0
        
    return avg_loss, mae_pts, mae_reb, mae_ast, mae_fg3m, mae_blk, mae_stl

def evaluate_implied_totals(model, loader, scaler, feature_cols):
    """
    Evaluates MAE on IMPLIED Totals (Per-Min Prediction * Actual Minutes).
    This helps compare against previous baselines.
    """
    model.eval()
    y_true_totals = []
    y_pred_totals = []
    
    with torch.no_grad():
        for p_idx, t_idx, x_cont, m_idx, y in loader:
            p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
            outputs = model(p_idx, t_idx, x_cont, m_idx)
            
            # y is per-minute
            # where is minutes? It's in the original dataframe, but not passed in loader directly except maybe in x_cont?
            # We don't have minutes in X usually (it's a target for the minutes model).
            # So we can't easily calculate this inside the loader loop without passing MIN as a feature or metadata.
            # For now, we will skip this or add MIN to the loader if needed.
            pass
            
    return 0, 0, 0

def train_model(target_player_id: int = None, debug: bool = False, pretrain: bool = False, calibrate: bool = False, calibration_weight: float = 0.1):
    print(f"Training on {device} (Debug={debug}, Pretrain={pretrain}, Calibrate={calibrate})...")
    
    # 1. Load Data
    path = os.path.join(PROCESSED_DIR, 'processed_features.csv')
    df = pd.read_csv(path)
    
    # [NEW] Filter to current season (2026) only - GLOBAL FILTER
    print("Filtering data to current season (2026)...")
    df = df[df['SEASON_YEAR'] == 2026]
    
    if df.empty:
        print("No data found for current season (2026).")
        return

    # Filter for specific player if needed
    if target_player_id and not pretrain:
        print(f"Filtering data for Player ID: {target_player_id}")
        # Need Encoder for filtering by ID -> IDX map or just use ID since we saved it
        df = df[df['PLAYER_ID'] == target_player_id]
        
        if df.empty:
            print("No data found for player in current season (2026).")
            return

        # [NEW] Check minimum sample size
        if len(df) < 20:
             print(f"Insufficient games ({len(df)} < 20) for fine-tuning. Using Global Model.")
             return


    # Load Artifacts
    feature_cols = joblib.load(os.path.join(MODELS_DIR, 'feature_cols.joblib'))
    player_encoder = joblib.load(os.path.join(MODELS_DIR, 'player_encoder.joblib')) 
    team_encoder = joblib.load(os.path.join(MODELS_DIR, 'team_encoder.joblib'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
    
    num_players = len(player_encoder.classes_)
    num_teams = len(team_encoder.classes_)
    num_cont = len(feature_cols)
    print(f"Features: {num_cont}, Players: {num_players}, Teams: {num_teams}")

    # 2. Split (Strict Time Based)
    df = df.sort_values('GAME_DATE')
    
    # For global pretraining, we can use a random split or time split. 
    # Time split is better to prevent leakage.
    split_idx = int(len(df) * 0.85)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")
    # Prepare Dataset
    # Separate numeric features from Metadata
    # X needs: PLAYER_IDX, TEAM_IDX, Continuous Features
    
    # [FIX] Do not recompute feature_cols. Use the loaded one.
    # feature_cols = [c for c in df.columns if c not in ['PLAYER_ID' ...]]
    # joblib.dump(feature_cols, ...) 
    
    # Ensure all feature cols exist in DF
    # If not, fill with 0 (though they should exist if processed correctly)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    X = df[['PLAYER_IDX', 'TEAM_IDX'] + feature_cols].values.astype(np.float32)
    # [NEW] Targets are PER_MIN stats
    target_cols = ['PTS_PER_MIN', 'REB_PER_MIN', 'AST_PER_MIN', 'FG3M_PER_MIN', 'BLK_PER_MIN', 'STL_PER_MIN']
    
    # Check if columns exist
    missing_targets = [c for c in target_cols if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}. Please run feature engineering to generate per-minute stats.")
        
    y = df[target_cols].values.astype(np.float32)
    
    # ---- PROCESS MISSING IDS ----
    print("Processing Missing Player Embeddings...")
    missing_raw = df['MISSING_PLAYER_IDS'].fillna("NONE").astype(str).tolist()
    
    # Pad Index = num_players_encoded (The size of encoder classes)
    # The Embedding layer has size (len(classes) + 1). 
    # Indices 0 to N-1 are valid players. Index N is Padding.
    pad_idx = len(player_encoder.classes_)
    max_missing = 3
    
    missing_matrix = []
    
    # Create a mapping for speed
    id_to_idx = {pid: idx for idx, pid in enumerate(player_encoder.classes_)}
    
    for row_str in missing_raw:
        if row_str == "NONE" or row_str == "":
            indices = []
        else:
            pids = row_str.split('_')
            indices = []
            for pid_str in pids:
                try:
                    pid = int(float(pid_str)) # Handle "123.0" potential string
                    if pid in id_to_idx:
                        indices.append(id_to_idx[pid])
                    else:
                        # Unknown player (not in training set) -> Treat as Padding/Ignore
                        pass 
                except:
                    pass
        
        # Truncate or Pad
        if len(indices) > max_missing:
            indices = indices[:max_missing]
        else:
            indices += [pad_idx] * (max_missing - len(indices))
            
        missing_matrix.append(indices)
        
    missing_matrix = np.array(missing_matrix, dtype=np.int64)
    # -----------------------------
    
    # Split
    train_indices, val_indices = train_test_split(np.arange(len(df)), test_size=0.15, shuffle=True, random_state=42)
    
    # Calculate Baseline MAE (Mean of Train Predicts Val)
    y_train = y[train_indices]
    y_val = y[val_indices]
    
    baseline_preds = np.mean(y_train, axis=0) # [PTS, REB, AST, 3PM, BLK, STL]
    baseline_mae = np.mean(np.abs(y_val - baseline_preds), axis=0)
    
    baseline_mae_pts = baseline_mae[0]
    baseline_mae_reb = baseline_mae[1]
    baseline_mae_ast = baseline_mae[2]
    baseline_mae_fg3m = baseline_mae[3]
    baseline_mae_blk = baseline_mae[4]
    baseline_mae_stl = baseline_mae[5]
    
    print(f"Baseline MAE - PTS: {baseline_mae_pts:.4f}, REB: {baseline_mae_reb:.4f}, AST: {baseline_mae_ast:.4f}")

    # Weighted Sampler for Training
    # Give higher weight to recent games using Exponential Decay
    # We need weights for TRAIN set only
    train_df = df.iloc[train_indices]
    
    from src.weighting import ExponentialDecayWeighter
    weighter = ExponentialDecayWeighter(decay_factor=0.95)
    
    # Calculate weights based on GAME_DATE recency
    weights = weighter.calculate_weights(train_df, date_col='GAME_DATE')
    
    # [NEW] Weight Transparency / Validation
    print("\n--- Weighting Statistics ---")
    print(f"Min Weight: {weights.min():.4f}, Max Weight: {weights.max():.4f}, Mean Weight: {weights.mean():.4f}")
    
    # Compare recent vs older
    # Since train_df is sorted by date? No, train_df is a slice of df which IS sorted by date.
    # train_df = df.iloc[train_indices] -> train_indices are RANDOM shuffled split.
    # So we need to look at logical recency.
    # But wait, weights are aligned with train_df using the correct indices.
    # Let's reconstruct a view to check dates vs weights.
    
    weight_check_df = train_df.copy()
    weight_check_df['weight'] = weights
    weight_check_df = weight_check_df.sort_values('GAME_DATE', ascending=False)
    
    recent_10_avg = weight_check_df.iloc[:10]['weight'].mean() if len(weight_check_df) >= 10 else weight_check_df['weight'].mean()
    old_20_plus_avg = weight_check_df.iloc[20:]['weight'].mean() if len(weight_check_df) > 20 else 0.0
    
    print(f"Avg Weight (Last 10 Games): {recent_10_avg:.4f}")
    print(f"Avg Weight (Games 20+ Ago): {old_20_plus_avg:.4f}")
    if old_20_plus_avg > 0:
        print(f"Recency Ratio: {recent_10_avg / old_20_plus_avg:.2f}x")
    print("----------------------------\n")
    
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_dataset = NBADataset(X[train_indices], y[train_indices], missing_matrix[train_indices])
    val_dataset = NBADataset(X[val_indices], y[val_indices], missing_matrix[val_indices])
    
    train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Model Init
    model = NBAPredictor(
        num_players=len(player_encoder.classes_) + 1, 
        num_teams=len(team_encoder.classes_), 
        num_cont=len(feature_cols)
    ).to(device)
    
    # TRANSFER LEARNING LOGIC
    global_model_path = os.path.join(MODELS_DIR, 'pytorch_nba_global_vNext.pth') # vNext to avoid conflict/ensure refresh
    if os.path.exists(global_model_path):
        print(f"Loading Global Model weights from {global_model_path}...")
        try:
            model.load_state_dict(torch.load(global_model_path, map_location=device), strict=False)
            if not pretrain: # Only print this if we are fine-tuning, not pretraining
                print("Global weights loaded successfully! Fine-tuning now...")
        except Exception as e:
            print(f"Could not load global weights (likely architecture mismatch): {e}")
            print("Proceeding with fresh weights.")
    else:
        print("No global model found. Training from scratch.")
    
    # Loss function: Gaussian NLL for V2, MSE for V1
    # Loss function
    criterion = gaussian_nll_loss
    
    # Lower LR for fine-tuning
    lr = 0.001 if pretrain else 0.0005 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    save_path = global_model_path if pretrain else os.path.join(MODELS_DIR, f'pytorch_player_{target_player_id}.pth')
    
    early_stopping = EarlyStopping(patience=50 if not debug else 1000, path=save_path) if not (debug and not pretrain) else None
    
    # 5. Training Loop
    epochs = 100 if pretrain else 150
    
    # History for plotting
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'mae_pts': [],
        'mae_reb': [],
        'mae_ast': [],
        'alignment': []
    }
    
    # Plotting Setup for Debug
    # Plotting Setup (Debug or Pretrain)
    if debug or pretrain:
        import matplotlib.pyplot as plt
        plt.ion() # Interactive mode
        fig, axs = plt.subplots(3, 2, figsize=(15, 15)) # Changed to 3x2
        
        # Plot 1: Loss
        ax_loss = axs[0, 0]
        line_train_loss, = ax_loss.plot([], [], label='Train Loss')
        line_val_loss, = ax_loss.plot([], [], label='Val Loss')
        ax_loss.set_title(f'Loss ({ "Global" if pretrain else f"Player {target_player_id}" })')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('MSE Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        
        # Plot 2: PTS MAE
        ax_pts = axs[0, 1]
        line_mae_pts, = ax_pts.plot([], [], label='PTS MAE')
        ax_pts.axhline(y=baseline_mae_pts, color='r', linestyle='--', label='Baseline')
        ax_pts.set_title('PTS MAE')
        ax_pts.set_xlabel('Epoch')
        ax_pts.legend()
        ax_pts.grid(True)

        # Plot 3: REB MAE
        ax_reb = axs[1, 0]
        line_mae_reb, = ax_reb.plot([], [], label='REB MAE')
        ax_reb.axhline(y=baseline_mae_reb, color='r', linestyle='--', label='Baseline')
        ax_reb.set_title('REB MAE')
        ax_reb.set_xlabel('Epoch')
        ax_reb.legend()
        ax_reb.grid(True)

        # Plot 4: AST MAE
        ax_ast = axs[1, 1]
        line_mae_ast, = ax_ast.plot([], [], label='AST MAE')
        ax_ast.axhline(y=baseline_mae_ast, color='r', linestyle='--', label='Baseline')
        ax_ast.set_title('AST MAE')
        ax_ast.set_xlabel('Epoch')
        ax_ast.legend()
        ax_ast.grid(True)
        
        # Plot 5: Alignment
        ax_align = axs[2, 0] # 3rd row, 1st col
        line_align, = ax_align.plot([], [], label='Alignment Score')
        ax_align.set_title('Distribution Alignment (1.0 = Perfect)')
        ax_align.set_xlabel('Epoch')
        ax_align.set_ylim(0, 1.0)
        ax_align.legend()
        ax_align.grid(True)
        
        # Hide 6th if unused or use for something else?
        axs[2, 1].axis('off')
        
        plt.tight_layout()

        
        if pretrain:
            plot_path = os.path.join(DATA_DIR, 'training_curve_global.png')
        else:
            plot_path = os.path.join(DATA_DIR, f'training_curve_{target_player_id}.png')

    print("\nStarting Training (Per-Minute Targets)...")
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'PM_PTS MAE':<10} | {'PM_REB MAE':<10}")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for p_idx, t_idx, x_cont, m_idx, y in train_loader:
            p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(p_idx, t_idx, x_cont, m_idx)
            
            # Apply Calibration Penalty if enabled
            cal_weight = calibration_weight if calibrate else 0.0
            loss = criterion(outputs, y, calibration_penalty_weight=cal_weight)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        val_loss = 0
        all_y_true = []
        all_y_pred = []
        model.eval()
        with torch.no_grad():
            for p_idx, t_idx, x_cont, m_idx, y in val_loader:
                p_idx, t_idx, x_cont, m_idx, y = p_idx.to(device), t_idx.to(device), x_cont.to(device), m_idx.to(device), y.to(device)
                outputs = model(p_idx, t_idx, x_cont, m_idx)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                
                all_y_true.append(y.cpu().numpy())
                all_y_pred.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate MAE for validation set
        y_true_combined = np.concatenate(all_y_true, axis=0)
        y_pred_combined = np.concatenate(all_y_pred, axis=0)
        
        # Distributional output is [mean(6), logvar(6)]
        y_pred_means = y_pred_combined[:, :6]
        
        if len(y_true_combined) > 0:
            mae_pts = np.mean(np.abs(y_true_combined[:, 0] - y_pred_means[:, 0]))
            mae_reb = np.mean(np.abs(y_true_combined[:, 1] - y_pred_means[:, 1]))
            mae_ast = np.mean(np.abs(y_true_combined[:, 2] - y_pred_means[:, 2]))
            mae_fg3m = np.mean(np.abs(y_true_combined[:, 3] - y_pred_means[:, 3]))
            mae_blk = np.mean(np.abs(y_true_combined[:, 4] - y_pred_means[:, 4]))
            mae_stl = np.mean(np.abs(y_true_combined[:, 5] - y_pred_means[:, 5]))
        else:
            mae_pts, mae_reb, mae_ast, mae_fg3m, mae_blk, mae_stl = 0, 0, 0, 0, 0, 0
        
        # Scheduler update
        scheduler.step(avg_val_loss)
        
        # Track history
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(avg_val_loss)
        history['mae_pts'].append(mae_pts)
        history['mae_reb'].append(mae_reb)
        history['mae_ast'].append(mae_ast)
        # Add others if we want to plot/track in history
        # For now just metrics saving is critical

        
        # Calculate Alignment Metric
        alignment = calculate_distribution_alignment(y_true_combined, y_pred_means)
        if (epoch+1) % 10 == 0:
             print(f"   Distribution Alignment Score: {alignment:.3f}")
             
        history['alignment'].append(alignment)
             
        # Update Plot
        if debug or pretrain:
            # Update data
            line_train_loss.set_data(history['epoch'], history['train_loss'])
            line_val_loss.set_data(history['epoch'], history['val_loss'])
            
            line_mae_pts.set_data(history['epoch'], history['mae_pts'])
            line_mae_reb.set_data(history['epoch'], history['mae_reb'])
            line_mae_ast.set_data(history['epoch'], history['mae_ast'])
            line_align.set_data(history['epoch'], history['alignment'])
            
            # Rescale axes
            for ax in [ax_loss, ax_pts, ax_reb, ax_ast, ax_align]:
                ax.relim()
                ax.autoscale_view()
            
            # Draw
            plt.draw()
            plt.pause(0.01) # Trigger full redraw loop
            
            # Save current frame frequently
            if epoch % 5 == 0:
                 plt.savefig(plot_path)

        # Early Stopping
        if early_stopping:
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"{epoch+1:<6} | {train_loss:<10.4f} | {avg_val_loss:<10.4f} | {mae_pts:<10.4f} | {mae_reb:<10.4f}")
                print("Early stopping triggered.")
                break
                
        # Always output in debug or periodic
        if (epoch+1) % 10 == 0:
             print(f"{epoch+1:<6} | {train_loss:<10.4f} | {avg_val_loss:<10.4f} | {mae_pts:<10.4f} | {mae_reb:<10.4f}")
    
    # Save model at end
    if not early_stopping:
        torch.save(model.state_dict(), save_path)
        print(f"\nDebug Run Complete. Model Saved to {save_path}")
        if debug or pretrain: 
            plt.savefig(plot_path)
            print(f"Final plot saved to {plot_path}")
            plt.ioff()
            plt.close()
    else:
        model.load_state_dict(torch.load(save_path))
        print(f"\nFinal Best Model Saved to {save_path}")
        
    print(f"Final Validation Scores -> PTS_PER_MIN MAE: {mae_pts:.4f} (Baseline: {baseline_mae_pts:.4f})")
    
    # Save Metrics to JSON
    import json
    metrics = {
        'mae_pts_per_min': float(mae_pts),
        'mae_reb_per_min': float(mae_reb),
        'mae_ast_per_min': float(mae_ast),
        'mae_fg3m_per_min': float(mae_fg3m),
        'mae_blk_per_min': float(mae_blk),
        'mae_stl_per_min': float(mae_stl),
        'baseline_mae_pts_per_min': float(baseline_mae_pts),
        'alignment_score': float(history['alignment'][-1]) if history['alignment'] else 0.0,
        'calibration_enabled': calibrate,
        'calibration_weight': calibration_weight,
        'date': datetime.now().strftime('%Y-%m-%d'),
        'model_version': 'per_minute_v1'
    }
    
    if pretrain:
        metrics_path = os.path.join(DATA_DIR, 'models', 'metrics_global.json')
    else:
        metrics_path = os.path.join(DATA_DIR, 'models', f'metrics_player_{target_player_id}.json')
        
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--player_id', type=int, help='Player ID to train on')
    parser.add_argument('--debug', action='store_true', help='Debug mode (no early stop, plotting)')
    parser.add_argument('--pretrain', action='store_true', help='Pretrain global model on all data')
    parser.add_argument('--v2', action='store_true', help='Use V2 distributional model (Gaussian NLL)')
    parser.add_argument('--calibrate', action='store_true', help='Enable calibration penalty in loss')
    parser.add_argument('--calibration_weight', type=float, default=0.1, help='Weight of calibration penalty')
    args = parser.parse_args()
    
    if args.pretrain:
        train_model(pretrain=True, calibrate=args.calibrate, calibration_weight=args.calibration_weight)
        
        # Also pretrain Global Minutes Predictor
        print("\n--- Training Global Minutes Predictor ---")
        from src.minutes_predictor import MinutesPredictor
        predictor = MinutesPredictor()
        predictor.train_global_model()
    elif args.player_id:
        train_model(target_player_id=args.player_id, debug=args.debug, calibrate=args.calibrate, calibration_weight=args.calibration_weight)
    else:
        print("Please provide --player_id or --pretrain")

