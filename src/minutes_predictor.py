import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import joblib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
PROCESSED_FEATURES_PATH = os.path.join(DATA_DIR, 'processed', 'processed_features.csv')
MINUTES_MODEL_PATH = os.path.join(MODELS_DIR, 'minutes_model_global.pth')
MINUTES_SCALER_PATH = os.path.join(MODELS_DIR, 'minutes_scaler.joblib')

class MinutesMLP(nn.Module):
    def __init__(self, input_dim):
        super(MinutesMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1) # Output: Predicted Minutes
        )
        
    def forward(self, x):
        return self.network(x)

class MinutesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MinutesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = [
            'recent_min_avg', 
            'recent_min_std', 
            'INJURY_SEVERITY_TOTAL', 
            'STARS_OUT', # Ensure this exists in processed data (created by calculate_teammate_absence)
            'OPP_ROLL_PACE', # 'opponent_pace'
            'IS_HOME', 
            'DAYS_REST', 
            'season_min_avg'
        ]
        
    def _load_model(self):
        if os.path.exists(MINUTES_MODEL_PATH) and os.path.exists(MINUTES_SCALER_PATH):
            self.scaler = joblib.load(MINUTES_SCALER_PATH)
            self.model = MinutesMLP(len(self.features))
            self.model.load_state_dict(torch.load(MINUTES_MODEL_PATH))
            self.model.eval()
            return True
        return False

    def train_global_model(self, data_path: str = PROCESSED_FEATURES_PATH):
        print("Training Global Minutes Predictor...")
        df = pd.read_csv(data_path)
        
        # Check required columns
        missing_cols = [c for c in self.features if c not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}. Filling with 0.")
            for c in missing_cols:
                df[c] = 0.0
                
        # Drop rows with NaN targets or features?
        # Fill features with reasonable defaults for robustness
        df['recent_min_avg'] = df['recent_min_avg'].fillna(20.0)
        df['recent_min_std'] = df['recent_min_std'].fillna(0.0)
        df['INJURY_SEVERITY_TOTAL'] = df['INJURY_SEVERITY_TOTAL'].fillna(0.0)
        df['STARS_OUT'] = df['STARS_OUT'].fillna(0)
        df['OPP_ROLL_PACE'] = df['OPP_ROLL_PACE'].fillna(100.0)
        df['IS_HOME'] = df['IS_HOME'].fillna(0.5)
        df['DAYS_REST'] = df['DAYS_REST'].fillna(2.0)
        df['season_min_avg'] = df['season_min_avg'].fillna(20.0)
        
        df = df.dropna(subset=['MIN']) # Target must exist
        
        X = df[self.features].values
        y = df['MIN'].values
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.15, random_state=42)
        
        train_dataset = MinutesDataset(X_train, y_train)
        val_dataset = MinutesDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64)
        
        # Model
        self.model = MinutesMLP(len(self.features))
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        epochs = 50
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    outputs = self.model(X_batch)
                    val_loss += criterion(outputs, y_batch).item()
            
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
                
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(self.model.state_dict(), MINUTES_MODEL_PATH)
                
        # Save Scaler
        joblib.dump(self.scaler, MINUTES_SCALER_PATH)
        print(f"Minutes Prediction Model Trained & Saved. Best Val Loss: {best_val_loss:.4f} (MAE ~ {np.sqrt(best_val_loss):.2f})")

    def predict(self, context_dict: dict) -> float:
        """
        Predicts minutes for a single player-game context.
        Args:
            context_dict: Dictionary containing feature values.
        """
        if self.model is None:
            if not self._load_model():
                print("Error: Minutes model not found. Please train first.")
                return 30.0 # Fallback
                
        # Extract features in order
        feature_vals = []
        for f in self.features:
            val = context_dict.get(f, 0.0) # Default 0 if missing
            feature_vals.append(val)
            
        X = np.array([feature_vals])
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(X_tensor).item()
            
        return max(0.0, pred) # Minutes cannot be negative

if __name__ == "__main__":
    predictor = MinutesPredictor()
    predictor.train_global_model()
