from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os
import pandas as pd
from datetime import datetime
from typing import List, Optional
import io
import contextlib

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetch import fetch_daily_scoreboard
from src.daily_predict import predict_daily
from src.train_models import train_model
from src.train_models import train_model
from nba_api.stats.endpoints import commonteamroster

# Constants
DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Logger for WebSocket
class LogManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

log_manager = LogManager()

# Capture stdout/stderr logic
class StreamToLogger(object):
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, buf):
        self.original_stream.write(buf)
        # Verify loop exists before creating task to avoid "RuntimeError: no running event loop"
        try:
             loop = asyncio.get_running_loop()
             if loop.is_running():
                 asyncio.create_task(log_manager.broadcast(buf))
        except:
             pass 

    def flush(self):
        self.original_stream.flush()

# Redirect stdout/stderr (Careful with concurrency)
# sys.stdout = StreamToLogger(sys.stdout) # Global override might be risky but effective for "Console Log"

@app.websocket("/logs")
async def websocket_endpoint(websocket: WebSocket):
    await log_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except:
        log_manager.disconnect(websocket)

class AnalysisRequest(BaseModel):
    date: Optional[str] = None
    force_train: bool = False
    check_injuries: bool = True  # New: Enable injury-triggered retraining

@app.post("/analyze-today")
async def analyze_today(req: AnalysisRequest):
    """
    Main workflow:
    1. Check for injury changes (trigger retraining if needed)
    2. Get games for date
    3. Get active active players in those games
    4. Check if trained
    5. Train if needed
    6. Predict
    """
    target_date = req.date if req.date else datetime.now().strftime('%Y-%m-%d')
    current_season = "2025-26" # Hardcoded for now, logical
    
    await log_manager.broadcast(f"Starting Analysis for {target_date}...")
    
    # ==== INJURY CHANGE DETECTION ====
    if req.check_injuries:
        try:
            from src.retrain_trigger import detect_injury_changes
            from src.retraining_queue import get_queue
            
            await log_manager.broadcast("🏥 Checking for injury report changes...")
            
            injury_result = detect_injury_changes()
            
            if injury_result['affected_player_ids']:
                await log_manager.broadcast(
                    f"🚨 {len(injury_result['affected_player_ids'])} players affected by injury changes"
                )
                
                # Queue retraining in background (non-blocking)
                queue = get_queue()
                await queue.queue_retraining(injury_result['affected_player_ids'], use_v2=True)
                
                await log_manager.broadcast(
                    f"⏳ Retraining queued for affected players (running in background)"
                )
            else:
                await log_manager.broadcast("✅ No injury changes detected")
        except Exception as e:
            await log_manager.broadcast(f"⚠️ Injury check failed: {str(e)[:100]}")
            # Continue with predictions even if injury check fails
    
    # ==== CONTINUE WITH NORMAL PREDICTION FLOW ====
    
    # 1. Fetch Scoreboard
    scoreboard = fetch_daily_scoreboard(target_date)
    if scoreboard.empty:
        await log_manager.broadcast("No games found today.")
        return {"message": "No games found", "results": []}
    
    team_ids = set(scoreboard['HOME_TEAM_ID'].unique().tolist() + scoreboard['VISITOR_TEAM_ID'].unique().tolist())
    await log_manager.broadcast(f"Found {len(team_ids)} teams playing today.")
    
    all_predictions = []
    
    # 2. Iterate Teams
    
    # 2. Build Execution Plan (Fetch Rosters)
    await log_manager.broadcast("Building Execution Plan (Fetching Rosters)...")
    
    execution_list = []
    
    current_team_idx = 0
    total_teams = len(team_ids)
    
    # We need to know who is playing whom to set opp_id
    # Reset scoreboard iteration to map team -> opp
    team_schedule = {} # {team_id: {'opp_id': id, 'is_home': bool}}
    for _, row in scoreboard.iterrows():
        h = row['HOME_TEAM_ID']
        v = row['VISITOR_TEAM_ID']
        team_schedule[h] = {'opp_id': v, 'is_home': True}
        team_schedule[v] = {'opp_id': h, 'is_home': False}
        
    for team_id in team_ids:
        current_team_idx += 1
        await log_manager.broadcast(f"Scanning Team {current_team_idx}/{total_teams} (ID: {team_id})...")
        try:
             roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=current_season).get_data_frames()[0]
             
             # Context
             ctx = team_schedule.get(team_id, {})
             
             for _, player in roster.iterrows():
                 execution_list.append({
                     'pid': player['PLAYER_ID'],
                     'pname': player['PLAYER'],
                     'team_id': team_id,
                     'opp_id': ctx.get('opp_id'),
                     'is_home': ctx.get('is_home')
                 })
        except Exception as e:
             await log_manager.broadcast(f"[WARN] Failed to fetch roster for team {team_id}: {e}")

    total_players = len(execution_list)
    await log_manager.broadcast(f"Plan Built: {total_players} Players identified across {total_teams} Teams.")

    # ---------------------------------------------------------
    # 3. ON-DEMAND TRAINING (Missing Models)
    # ---------------------------------------------------------
    await log_manager.broadcast("Checking for missing player models...")
    missing_pids = []

    for item in execution_list:
        pid = item['pid']
        # Check if pytorch_player_{pid}.pth exists
        # Note: train_models.py now saves as pytorch_player_{pid}.pth (v2 suffix removed)
        model_path = os.path.join(MODELS_DIR, f'pytorch_player_{pid}.pth')
        
        # Also check if we have a global model fallback? 
        # User requested specific models to be trained.
        if not os.path.exists(model_path):
            missing_pids.append(item)
            
    if missing_pids:
        count = len(missing_pids)
        await log_manager.broadcast(f"⚠️ Found {count} players with missing models. Training them now...")
        
        for i, item in enumerate(missing_pids):
            pid = item['pid']
            pname = item['pname']
            
            # Skip if recently processed in this loop (duplicates?)
            # execution_list is unique by nature of loop? No, roster is unique.
            
            msg = f"Training [{i+1}/{count}]: {pname}..."
            await log_manager.broadcast(msg)
            
            try:
                # Run training in thread to avoid blocking WebSocket heartbeats
                # using asyncio.to_thread (requires Python 3.9+)
                await asyncio.to_thread(train_model, target_player_id=pid)
            except Exception as e:
                await log_manager.broadcast(f"❌ Failed to train {pname}: {e}")
                
        await log_manager.broadcast("✅ On-demand training complete.")
    else:
        await log_manager.broadcast("✅ All player models present.")

    # ---------------------------------------------------------
    # 4. Execute BATCH
    # ---------------------------------------------------------
    from src.batch_predict import BatchPredictor
    batch_predictor = BatchPredictor()
    
    # Train missing models first?
    # Batch Predictor assumes trained models.
    # We can do a quick pass for training if we want, OR just let Batch Predictor fallback to generic.
    # The user values speed. Generic fallback is fine for new players.
    
    # However, if we need to TRAIN, we should check missing models.
    # Let's do a quick check? 
    # For now, let's SKIP training automation in favor of speed, unless explicitly requested.
    
    # If req.force_train is True, we can't do batch effectively unless we train first.
    if req.force_train:
         await log_manager.broadcast("Force Train requested. Retraining ALL identified players...")
         for i, item in enumerate(execution_list):
            pid = item['pid']
            pname = item['pname']
            await log_manager.broadcast(f"Force Retrain [{i+1}/{total_players}]: {pname}...")
            try:
                await asyncio.to_thread(train_model, target_player_id=pid)
            except Exception as e:
                 await log_manager.broadcast(f"Failed to train {pname}: {e}")
    
    # Running Batch Analysis
    try:
        results = await batch_predictor.analyze_today_batch(
            date_input=target_date, 
            execution_list=execution_list, 
            log_manager=log_manager
        )
    except Exception as e:
        await log_manager.broadcast(f"Batch Analysis Failed: {e}")
        import traceback
        traceback.print_exc()
        return {"results": []}

    # Handle new dictionary return format (predictions + trixie)
    if isinstance(results, dict):
        preds_list = results.get('predictions', [])
        trixie = results.get('trixie', None)
    else:
        preds_list = results
        trixie = None

    # sorted_preds = sorted(preds_list, key=lambda x: x['PRED_PTS'] if x['PRED_PTS'] else 0, reverse=True)
    sorted_preds = preds_list # Already sorted by BatchPredictor (Tier -> Units -> Consistency)
    
    await log_manager.broadcast(f"Analysis Complete. Generated {len(sorted_preds)} predictions.")
    return {"predictions": sorted_preds, "trixie": trixie, "results": sorted_preds} # Keep 'results' for legacy compat

@app.get("/health")
def health():
    return {"status": "ok"}

class CustomBetRequest(BaseModel):
    player_id: int
    prop_type: str # 'PTS', 'REB', 'RA', etc.
    custom_line: float
    prediction: float
    mae: float
    recent_stats: Optional[dict] = None

@app.post("/analyze-custom-bet")
async def analyze_custom_bet(req: CustomBetRequest):
    """
    Recalculates units and badge for a user-supplied custom line.
    """
    from src.bet_sizing import calculate_bet_quality
    
    # Calculate Quality
    result = calculate_bet_quality(
        prop_name=req.prop_type,
        val=req.prediction,
        mae=req.mae,
        line=req.custom_line,
        recent_stats=req.recent_stats if req.recent_stats else {}
    )
    
    return result
