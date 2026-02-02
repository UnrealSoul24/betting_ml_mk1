"""
Batch Training Script - Trains V2 models for all rotation players.

Usage:
    python scripts/train_all_players.py

This script:
1. Loads the processed features
2. Identifies all unique players with sufficient data
3. Trains individual V2 models for each player (fine-tuning from global)
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_models import train_model

DATA_DIR = Path(__file__).parent.parent / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'

def get_rotation_players(min_games: int = 20, min_ppg: float = 5.0):
    """
    Identifies rotation players worth training individual models for.
    
    Args:
        min_games: Minimum games played
        min_ppg: Minimum PPG average
    
    Returns:
        List of player_ids
    """
    path = PROCESSED_DIR / 'processed_features.csv'
    df = pd.read_csv(path)
    
    # Filter to 2024-2026 seasons only (recent data)
    df = df[df['SEASON_YEAR'] >= 2024]
    
    # Group by player
    player_stats = df.groupby('PLAYER_ID').agg({
        'PTS': 'mean',
        'GAME_DATE': 'count'
    }).rename(columns={'GAME_DATE': 'games'})
    
    # Filter
    rotation = player_stats[
        (player_stats['games'] >= min_games) & 
        (player_stats['PTS'] >= min_ppg)
    ]
    
    player_ids = rotation.index.tolist()
    print(f"Found {len(player_ids)} rotation players (>= {min_games} games, >= {min_ppg} PPG)")
    
    return player_ids

def main():
    print("=" * 60)
    print("BATCH V2 MODEL TRAINING")
    print("=" * 60)
    
    # Get players to train
    player_ids = get_rotation_players(min_games=20, min_ppg=5.0)
    
    total = len(player_ids)
    failed = []
    
    start_time = time.time()
    
    for i, pid in enumerate(player_ids):
        print(f"\n[{i+1}/{total}] Training Player ID: {pid}")
        print("-" * 40)
        
        try:
            # Train with V2 architecture
            train_model(target_player_id=pid)
            print(f"✓ Player {pid} complete")
        except Exception as e:
            print(f"✗ Player {pid} failed: {e}")
            failed.append(pid)
        
        # Progress estimate
        elapsed = time.time() - start_time
        avg_per_player = elapsed / (i + 1)
        remaining = avg_per_player * (total - i - 1)
        print(f"  [Progress: {i+1}/{total} | ETA: {remaining/60:.1f} min]")
    
    # Summary
    elapsed_total = (time.time() - start_time) / 60
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE")
    print(f"  Total players: {total}")
    print(f"  Successful: {total - len(failed)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed_total:.1f} minutes")
    print("=" * 60)
    
    if failed:
        print(f"\nFailed player IDs: {failed}")

if __name__ == "__main__":
    main()
