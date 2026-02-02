# Antigravity NBA Predictor (v4.0)

A professional-grade Machine Learning system for NBA player prop betting, featuring a dual-stage prediction architecture, real-time odds integration, and automated strategy generation (SGPs/Trixies).

![Distributional Model](https://img.shields.io/badge/Model-Distributional%20%28Gaussian%20NLL%29-blue)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-orange)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-blueviolet)

## 🚀 Key Features

### 🧠 Dual-Stage Prediction Engine
Unlike simple regression models, this system separates **Opportunity** from **Ability**:
1.  **Minutes Predictor**: A specialized model estimates playing time based on rotation changes, "Stars Out" impact, injury severity, blowout potential, and rest days.
2.  **Efficiency Model**: A distributional neural network predicts per-minute production (PTS/min, REB/min, etc.) and their associated uncertainties.
**Final Projection** = `Predicted Minutes × Predicted Efficiency`

### 🔄 Smart Retraining & Lifecycle
- **Just-in-Time Training**: The system detects when a player's model is missing, stale, or drifting in accuracy.
- **Trigger-Based Updates**: Retraining is automatically triggered by:
    - 🏥 **Injury Events**: Major rotation changes (e.g., "Embiid Out" triggers Maxey retrain).
    - 📉 **Accuracy Drift**: If recent predictions deviate beyond thresholds.
    - ⏱ **Staleness**: Periodic refreshes.
- **Exponential Decay**: Training data is weighted by recency to capture current form.

### 💰 Advanced Betting Logic
- **EV Calculation**: Compares model confidence intervals against real-time market odds to find positive Expected Value.
- **SGP & Trixie Generation**: Automatically constructs:
    - **Same Game Parlays (SGP)**: Correlated props from the same match.
    - **Trixies**: System bets combining high-confidence edges from different games (3 doubles + 1 treble).

### 📊 Modern Dashboard
A React+Vite frontend allows you to:
- View daily "Best Bets" and Trixie recommendations.
- Inspect detailed player cards with "Matchup", "Injury Impact", and "Last 5" analysis.
- Monitor background retraining jobs via WebSocket logs.

---

## 🛠️ Architecture

### Tech Stack
- **Backend**: Python 3.10+, FastAPI, PyTorch, Pandas, Numpy.
- **Frontend**: React, Vite, TailwindCSS.
- **Data**: Live NBA Stats API, Odds API integration.

### Core Modules
- `src/minutes_predictor.py`: Manages the playing time model.
- `src/batch_predict.py`: Orchestrates the daily prediction interaction.
- `src/retraining_queue.py`: Manages background model updates.
- `src/feature_engineer.py`: Generates 50+ contextual features (DvP, Pace, Rest, etc.).
- `src/bet_sizing.py`: Calculates Kelly Criterion and EV.

---

## 🚦 Usage

### 1. Start the Backend API
The API handles data fetching, model training, and prediction generation.

```bash
uvicorn src.api:app --reload
```
- API runs at: `http://localhost:8000`
- API Logs: `http://localhost:8000/logs` (WebSocket)

### 2. Start the Frontend
The React dashboard provides the visual interface.

```bash
cd frontend
npm run dev
```
- Dashboard: `http://localhost:5173`

### 3. Generate Predictions
1. Open the Dashboard.
2. Click **"Find Today's Bets"**.
3. The system will:
    - Fetch today's schedule and odds.
    - Check the latest injury reports.
    - **Automatically Train** models for any active players who don't have one yet.
    - Generate predictions and find value bets.

---

## � CLI Commands (Manual Usage)

You can also run individual components manually from the terminal for debugging or specific tasks.

### 1. Data Setup (One-Time / Periodic)
Download historical game logs and generate features.

```bash
# Download raw game data (2021-2026)
python src/data_fetch.py

# Process features (DvP, Rolling Stats, etc.)
python src/feature_engineer.py

# Pretrain Global Model & Minutes Predictor
python -m src.train_models --pretrain
```

### 2. Train a Player Model Manually
The API handles this automatically, but you can force-train a specific player.

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

## �📂 Project Structure

```text
├── data/
│   ├── raw/               # Downloaded NBA game logs
│   ├── processed/         # Engineered features ready for training
│   ├── models/            # Saved PyTorch models (.pth) and scalers
│   └── cache/             # Cached injury reports and metadata
├── src/
│   ├── api.py             # FastAPI backend implementation
│   ├── batch_predict.py   # High-performance batch prediction engine
│   ├── minutes_predictor.py # Minutes prediction logic
│   ├── feature_engineer.py# Feature generation (DvP, H2H, Rest)
│   ├── train_models.py    # PyTorch model definition & training loop
│   └── odds_service.py    # Fetches market odds / Calculates EV
├── frontend/              # React Dashboard
└── tests/                 # Unit and integration tests
```

## 🧠 Theory: Distributional Modeling
The system outputs a probability distribution for every stat, not just a single number. This allows us to calculate the probability of hitting a specific line (e.g., "Over 24.5 Points"). The model learns the **Aleatoric Uncertainty** (inherent noise) of each player.

## ⚠️ Troubleshooting

**"Model not found"**
The system is designed to auto-heal. If a model is missing, simply running a prediction for that player (via API or Dashboard) will trigger a background training job.

**"No games found"**
Ensure that today is actually a game day. You can verify the schedule in `data/cache` or check NBA.com.

**Environment Setup**
Ensure you have installed dependencies:
```bash
pip install -r requirements.txt
```
(GPU support recommended for faster training, but CPU works fine for inference).
