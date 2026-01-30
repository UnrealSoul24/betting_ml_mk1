"""
Retraining Queue - Background task manager for model retraining.

Handles asynchronous model retraining when injury changes are detected.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import List
from datetime import datetime


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_EXE = sys.executable


class RetrainingQueue:
    """Manages background retraining jobs."""
    
    def __init__(self):
        self.active_jobs = {}  # player_id -> asyncio.Task
        self.completed_jobs = []
        self.failed_jobs = []
        self.sem = asyncio.Semaphore(1) # Limit to 1 (or 2) concurrent trainings to prevent OOM
    
    async def queue_retraining(self, player_ids: List[int], use_v2: bool = True):
        """
        Queue retraining jobs for multiple players.
        
        Args:
            player_ids: List of player IDs to retrain
            use_v2: Whether to use V2 distributional model
        """
        print(f"\n[RetrainingQueue] Queuing {len(player_ids)} players for retraining...")
        
        for player_id in player_ids:
            if player_id in self.active_jobs:
                print(f"  ⏭️ Skipping {player_id} (already queued)")
                continue
            
            # Create background task
            task = asyncio.create_task(
                self._retrain_player_model(player_id, use_v2)
            )
            self.active_jobs[player_id] = task
            print(f"  ✓ Queued player {player_id}")
    
    async def _retrain_player_model(self, player_id: int, use_v2: bool = True):
        """
        Retrain a single player model in the background.
        
        Args:
            player_id: NBA player ID
            use_v2: Use distributional model (Arg kept for compatibility, ignored logic)
        """
        async with self.sem: # Critical: Wait for slot
            start_time = datetime.now()
            print(f"\n[Retrain] Starting training for player {player_id}...")
            
            try:
                # Build command
                cmd = [
                    PYTHON_EXE,
                    str(PROJECT_ROOT / "src" / "train_models.py"),
                    "--player_id", str(player_id)
                ]
                

                
                # Run training subprocess (Blocking wrapped in Thread)
                # Fix for Windows NotImplementedError (SelectorEventLoop)
                def run_sync():
                    return subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(PROJECT_ROOT)
                    )
                
                # Run in thread
                process = await asyncio.to_thread(run_sync)
                
                stdout = process.stdout
                stderr = process.stderr
                
                if process.returncode == 0:
                    duration = (datetime.now() - start_time).total_seconds()
                    print(f"[Retrain] ✅ Player {player_id} completed in {duration:.1f}s")
                    self.completed_jobs.append({
                        'player_id': player_id,
                        'timestamp': datetime.now().isoformat(),
                        'duration_sec': duration,
                        'success': True
                    })
                else:
                    print(f"[Retrain] ❌ Player {player_id} FAILED")
                    print(f"  STDOUT: {stdout.decode()[:500]}")
                    print(f"  STDERR: {stderr.decode()[:500]}")
                    self.failed_jobs.append({
                        'player_id': player_id,
                        'timestamp': datetime.now().isoformat(),
                        'error': (stderr.decode() + "\nSTDOUT: " + stdout.decode())[:1000],
                        'success': False
                    })
            
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"[Retrain] ❌ Player {player_id} EXCEPTION!")
                print(f"  Type: {type(e)}")
                print(f"  Repr: {repr(e)}")
                print(f"  Trace:\n{tb}")
                
                self.failed_jobs.append({
                    'player_id': player_id,
                    'timestamp': datetime.now().isoformat(),
                    'error': f"{repr(e)}\n{tb}",
                    'success': False
                })
            
            finally:
                # Remove from active jobs
                if player_id in self.active_jobs:
                    del self.active_jobs[player_id]
    
    def get_status(self):
        """Returns current queue status."""
        return {
            'active': len(self.active_jobs),
            'completed': len(self.completed_jobs),
            'failed': len(self.failed_jobs),
            'active_ids': list(self.active_jobs.keys()),
            'recent_completed': self.completed_jobs[-5:],
            'recent_failed': self.failed_jobs[-5:]
        }


# Global queue instance
_global_queue = None

def get_queue() -> RetrainingQueue:
    """Returns the global retraining queue singleton."""
    global _global_queue
    if _global_queue is None:
        _global_queue = RetrainingQueue()
    return _global_queue
