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
    
    # ==== INJURY CHANGE DETECTION ====
    if req.check_injuries:
        try:
            from src.retrain_trigger import detect_injury_changes
            from src.retraining_queue import get_queue
            
            # await log_manager.broadcast("🏥 Checking for injury report changes...")
            
            injury_result = detect_injury_changes()
            
            affected_ids = injury_result.get('affected_player_ids', [])
            stats = injury_result.get('stats', {})
            
            if affected_ids:
                count = len(affected_ids)
                # Get first affected player name for context
                all_affected = injury_result.get('changes', {}).get('all_affected', [])
                first_pname = all_affected[0] if all_affected else "unknown"
                
                # Queue retraining in background (non-blocking)
                queue = get_queue()
                await queue.queue_retraining(affected_ids, use_v2=True, trigger_reason=f"injury_change: {first_pname}")
                
                await log_manager.broadcast(f"⏳ Retraining queued for {count} players...")
            # else:
            #     await log_manager.broadcast("✅ No injury changes detected")
        except Exception as e:
            await log_manager.broadcast(f"⚠️ Injury check failed: {str(e)[:100]}")
            
    # ==== ACCURACY TRIGGER CHECK ====
    if req.check_injuries: 
        try:
             from src.retrain_trigger import check_accuracy_triggers
             # await log_manager.broadcast("🎯 Checking post-injury prediction accuracy...")
             
             acc_triggers = check_accuracy_triggers()
             
             if acc_triggers:
                 count = len(acc_triggers)
                 from src.retraining_queue import get_queue
                 queue = get_queue()
                 await queue.queue_retraining(acc_triggers, use_v2=True, trigger_reason="accuracy_degradation")
                 await log_manager.broadcast(f"⏳ Accuracy retraining queued for {count} players...")
                 
        except Exception as e:
             pass # Silent fail for accuracy check
    
    affected_count = len(affected_ids) if 'affected_ids' in locals() else 0
    await log_manager.broadcast(f"✅ Injuries checked ({affected_count} players affected)")
    
    # ==== CONTINUE WITH NORMAL PREDICTION FLOW ====
    
    # 1. Fetch Scoreboard
    scoreboard = fetch_daily_scoreboard(target_date)
    if scoreboard.empty:
        await log_manager.broadcast("No games found today.")
        return {"message": "No games found", "results": []}
    
    team_ids = set(scoreboard['HOME_TEAM_ID'].unique().tolist() + scoreboard['VISITOR_TEAM_ID'].unique().tolist())
    # await log_manager.broadcast(f"Found {len(team_ids)} teams playing today.")
    
    all_predictions = []
    
    # 2. Iterate Teams
    
    # 2. Build Execution Plan (Fetch Rosters)
    # await log_manager.broadcast("Building Execution Plan (Fetching Rosters)...")
    
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
        # await log_manager.broadcast(f"Scanning Team {current_team_idx}/{total_teams} (ID: {team_id})...")
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
             # await log_manager.broadcast(f"[WARN] Failed to fetch roster for team {team_id}: {e}")
             pass

    total_players = len(execution_list)
    # await log_manager.broadcast(f"Plan Built: {total_players} Players identified across {total_teams} Teams.")

    # ---------------------------------------------------------
    # 3. ON-DEMAND TRAINING (Missing Models)
    # ---------------------------------------------------------
    # await log_manager.broadcast("Checking for missing player models...")
    missing_pids = []

    for item in execution_list:
        pid = item['pid']
        model_path = os.path.join(MODELS_DIR, f'pytorch_player_{pid}.pth')
        
        if not os.path.exists(model_path):
            missing_pids.append(item)
            
    if missing_pids:
        count = len(missing_pids)
        await log_manager.broadcast(f"⏳ Training {count} missing models...")
        
        for i, item in enumerate(missing_pids):
            pid = item['pid']
            pname = item['pname']
            
            try:
                await asyncio.to_thread(train_model, target_player_id=pid)
            except Exception as e:
                pass
                
        # await log_manager.broadcast("✅ On-demand training complete.")

    # ---------------------------------------------------------
    # 4. Execute BATCH
    # ---------------------------------------------------------
    from src.batch_predict import BatchPredictor
    batch_predictor = BatchPredictor()
    
    if req.force_train:
         await log_manager.broadcast("⏳ Force Retraining ALL players...")
         for i, item in enumerate(execution_list):
            pid = item['pid']
            try:
                await asyncio.to_thread(train_model, target_player_id=pid)
            except Exception as e:
                 pass
    
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

    sorted_preds = preds_list 
    
    # await log_manager.broadcast(f"Analysis Complete. Generated {len(sorted_preds)} predictions.")
    return {"predictions": sorted_preds, "trixie": trixie, "results": sorted_preds}

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

# --- Retraining & Injury Endpoints ---

class ManualRetrainingRequest(BaseModel):
    player_ids: List[int]
    reason: str = "manual_trigger"
    force_priority: Optional[str] = None  # "CRITICAL", "HIGH", etc.

@app.post("/trigger-retraining")
async def trigger_manual_retraining(req: ManualRetrainingRequest):
    """
    Manually trigger retraining for specific players with injury context.
    """
    from src.retrain_trigger import calculate_retraining_priority, queue_injury_retraining, identify_star_players, get_player_team
    from src.retraining_queue import get_queue
    
    # Calculate priorities for requested players
    priorities = {}
    stars_map = identify_star_players()
    
    for pid in req.player_ids:
        if req.force_priority:
            priorities[pid] = {'priority_score': 100, 'tier': req.force_priority, 'reason': req.reason}
        else:
            tid = get_player_team(pid)
            if tid:
                priorities[pid] = calculate_retraining_priority(pid, "Manual", stars_map, tid)
            else:
                priorities[pid] = {'priority_score': 50, 'tier': "MANUAL", 'reason': req.reason}
    
    # Queue jobs
    from src.retraining_queue import get_queue
    queue = get_queue()
    await queue.queue_retraining(req.player_ids, use_v2=True, trigger_reason=req.reason)
    
    # Log manual trigger
    await log_manager.broadcast(f"Manual retraining triggered for {len(req.player_ids)} players: {req.reason}")
    
    return {
        "status": "queued",
        "player_count": len(req.player_ids),
        "priorities": priorities,
        "queue_status": queue.get_status()
    }

@app.get("/retraining-status")
async def get_retraining_status():
    """
    Get current retraining queue status and recent history.
    """
    from src.retraining_queue import get_queue
    
    queue = get_queue()
    status = queue.get_status()
    
    # Load recent history
    from src.retrain_trigger import RETRAINING_HISTORY
    if RETRAINING_HISTORY.exists():
        df = pd.read_csv(RETRAINING_HISTORY)
        recent = df.tail(20).to_dict('records')
    else:
        recent = []
    
    return {
        "queue": status,
        "recent_history": recent
    }

@app.get("/injury-impact-report")
async def get_injury_impact_report():
    from src.retrain_trigger import generate_injury_impact_report
    return generate_injury_impact_report()
