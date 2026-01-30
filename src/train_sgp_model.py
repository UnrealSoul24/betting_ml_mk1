import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
PROCESSED_DATA = os.path.join(DATA_DIR, 'processed', 'sgp_training_data.csv')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SGPMetaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Add src to path if needed or relative import
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sgp_model import SGPMetaModel

def train_meta_model():
    print("Loading SGP Training Data...")
    if not os.path.exists(PROCESSED_DATA):
        print("Data not found. Please run generate_sgp_data.py first.")
        return
        
    df = pd.read_csv(PROCESSED_DATA)
    print(f"Loaded {len(df)} samples.")
    
    # Features
    # We want Base Preds, Stds, and Context
    # generate_sgp_data.py saves: {STAT}_PRED, {STAT}_STD
    # We should add context features if possible?
    # Ideally generate_sgp_data.py should have saved them.
    # Currently generate_sgp_data saves: PRED, STD, SAFE, ACTUAL, HIT
    # It does NOT save L5 context or Vegas.
    # We should rely mainly on the MODEL's uncertainty (STD).
    
    feature_cols = []
    stats = ['PTS', 'REB', 'AST', '3PM', 'BLK', 'STL']
    for s in stats:
        feature_cols.append(f'{s}_PRED')
        feature_cols.append(f'{s}_STD')
        
    # Targets
    target_cols = [f'{s}_HIT' for s in stats]
    
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Scale Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'sgp_scaler.joblib'))
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Loaders
    train_loader = DataLoader(SGPMetaDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(SGPMetaDataset(X_val, y_val), batch_size=32)
    
    # Model
    model = SGPMetaModel(input_dim=len(feature_cols)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    print("Training SGP Meta Model...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Val
        model.eval()
        val_acc = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                preds = torch.sigmoid(out) > 0.5
                val_acc += (preds == y_b).float().mean().item()
        
        val_acc /= len(val_loader)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}, Val Acc {val_acc:.4f}")
            
    # Save
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'sgp_meta_model.pth'))
    print("SGP Meta Model Saved -> data/models/sgp_meta_model.pth")

if __name__ == "__main__":
    train_meta_model()
