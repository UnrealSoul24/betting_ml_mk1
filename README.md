# NBA Player Prop Prediction System (v3.0)

A professional-grade Machine Learning system for predicting NBA player stats (Points, Rebounds, Assists, 3-Pointers, Blocks, Steals) using PyTorch neural networks and advanced feature engineering (Defense vs. Position, Head-to-Head history, Injury Impact).

![Distributional Model](https://img.shields.io/badge/Model-Distributional%20%28Gaussian%20NLL%29-blue)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

## 🚀 Key Features

- **6 Prediction Targets**: Points, Rebounds, Assists, 3PM, Blocks, Steals.
- **Distributional Modeling**: Outputs both *predicted value* and *uncertainty* (confidence interval) for every stat.
- **Advanced Context**:
    - **Defense vs. Position (DvP)**: How well the opponent defends specific positions.
    - **Teammate Impact**: Adjusts predictions when star teammates (e.g., Embiid, Giannis) are injured.
    - **Vegas Adjustment**: Calibrates predictions based on the game's total (Over/Under).
- **On-Demand Training**: Automatically trains detailed player-specific models on-the-fly when you request a prediction.

## 🛠️ Installation

1.  **Clone & Environment**
    ```bash
    git clone <repo_url>
    cd betting_ml_mk1
    python -m venv venv
    
    # Windows
    .\venv\Scripts\activate
    # Mac/Linux
    source venv/bin/activate
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```
    *(Ensure you have PyTorch installed for your specific CUDA version if using GPU)*

## 🚦 Quick Start (The "Daily" Flow)

### 1. Start the Backend API
The easiest way to use the system is via the Web Dashboard.

```bash
uvicorn src.api:app --reload
```
- API runs at: `http://localhost:8000`
- API Logs: `http://localhost:8000/logs` (WebSocket)

### 2. Start the Frontend (Optional)
If you have the React frontend set up (in `/frontend`):

```bash
cd frontend
npm run dev
```
- Dashboard: `http://localhost:5173`

### 3. Generate Predictions
1. Open the Dashboard.
2. Click **"Find Today's Bets"**.
3. The system will:
    - active Fetch today's NBA schedule.
    - 🏥 Check the latest injury reports.
    - 🧠 **Automatically Train** models for any active players who don't have one yet.
    - 🔮 Generate predictions and find value bets.

---

## 💻 CLI Commands (Manual Usage)

You can also run individual components manually from the terminal.

### 1. Data Setup (One-Time / Periodic)
Download historical game logs and generate features.

```bash
# Download raw game data (2021-2026)
python src/data_fetch.py

# Process features (DvP, Rolling Stats, etc.)
python src/feature_engineer.py

# Pretrain Global Model (Required for Transfer Learning)
python -m src.train_models --pretrain
```

### 2. Train a Player Model Manually
Training is handled automatically by the API, but you can force-train a specific player.

```bash
# Train LeBron James (ID: 2544)
python -m src.train_models --player_id 2544
```

### 3. Run Daily Predictions for a Player
Get a quick CLI prediction for a specific player for *today*.

```bash
python -m src.daily_predict --player_id 2544
```

---

## 📂 Project Structure

```text
├── data/
│   ├── raw/               # Downloaded NBA game logs (.csv)
│   ├── processed/         # Engineered features ready for training
│   └── models/            # Saved PyTorch models (.pth) and scalers
├── src/
│   ├── api.py             # FastAPI backend implementation
│   ├── batch_predict.py   # High-performance batch prediction engine
│   ├── feature_engineer.py# Feature generation logic (DvP, H2H)
│   ├── train_models.py    # PyTorch model definition & training loop
│   ├── daily_predict.py   # CLI inference script
│   └── odds_service.py    # Fetches market odds / Calculates EV
└── frontend/              # (Optional) React Dashboard
```

## 🧠 Model Architecture

The system uses a **Distributional Regression Network**:
- **Inputs**: 
    - Player Embedding (ID)
    - Team/Opponent Embeddings
    - 50+ Continuous Features (Rolling Avgs, DvP, H2H, Rest Days)
    - "Missing Teammates" Embedding (Contextual Impact)
- **Outputs** (12 dimensions):
    - **Means (6)**: Expected value for Pts, Reb, Ast, 3PM, Blk, Stl.
    - **Log-Variances (6)**: Learned uncertainty for each stat (allows calculating confidence intervals).

## ⚠️ Troubleshooting

**"Model not found" / "Size Mismatch"**
- **Cause**: Feature definitions changed (e.g., added new stats).
- **Fix**: The system is designed to auto-fix this. Just running the API prediction will cause it to notice the mismatch/missing file and trigger a retrain.
- **Manual Fix**: Delete `data/models/*.pth` and run prediction again.

**"No games found"**
- **Cause**: No NBA games scheduled for the current date.
- **Fix**: Wait for a game day or test with a specific date using `python -m src.daily_predict --date YYYY-MM-DD`.
