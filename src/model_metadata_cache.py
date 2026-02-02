import json
import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional

# Constants
CACHE_FILE = os.path.join(os.path.dirname(__file__), '../data/models/model_metadata.json')

class ModelMetadataCache:
    """
    Manages metadata for player models (timestamps, training status, metrics).
    Thread-safe and persistent.
    """
    def __init__(self, filepath: str = CACHE_FILE):
        self.filepath = filepath
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        self._load()
        
    def _load(self):
        """Loads metadata from JSON file."""
        with self.lock:
            if os.path.exists(self.filepath):
                try:
                    with open(self.filepath, 'r') as f:
                        self.metadata = json.load(f)
                except Exception as e:
                    print(f"[ModelMetadataCache] Error loading cache: {e}")
                    self.metadata = {}
            else:
                self.metadata = {}

    def _save(self):
        """Saves metadata to JSON file."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.metadata, f, indent=4)
        except Exception as e:
            print(f"[ModelMetadataCache] Error saving cache: {e}")

    def update_metadata(self, player_id: int, updates: Dict[str, Any]):
        """
        Updates metadata for a specific player.
        Args:
            player_id: The Player ID.
            updates: Dictionary of fields to update (e.g. {'status': 'active', 'accuracy': 0.8})
        """
        pid_str = str(player_id)
        with self.lock:
            if pid_str not in self.metadata:
                self.metadata[pid_str] = {
                    'created_at': datetime.now().isoformat(),
                    'update_count': 0
                }
            
            # Update fields
            target = self.metadata[pid_str]
            for k, v in updates.items():
                target[k] = v
                
            # Auto-update timestamp
            target['last_updated'] = datetime.now().isoformat()
            target['update_count'] = target.get('update_count', 0) + 1
            
            self._save()
            
    def get_metadata(self, player_id: int) -> Optional[Dict[str, Any]]:
        """Returns metadata for a player or None if not found."""
        with self.lock:
            return self.metadata.get(str(player_id))

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Returns a copy of the entire metadata cache."""
        with self.lock:
            return self.metadata.copy()

    def set_training_status(self, player_id: int, status: str):
        """Helper to set just the status (training, active, failed)."""
        self.update_metadata(player_id, {'status': status})

    def get_status(self, player_id: int) -> str:
        """Returns the current status of the player's model (e.g., 'active', 'training', 'failed')."""
        meta = self.get_metadata(player_id)
        return meta.get('status', 'unknown') if meta else 'unknown'

    def is_fresh(self, player_id: int, max_age_hours: float = 24.0) -> bool:
        """
        Checks if the model metadata is fresh (updated within max_age_hours).
        """
        meta = self.get_metadata(player_id)
        if not meta or 'last_updated' not in meta:
            return False
            
        try:
            last_updated = datetime.fromisoformat(meta['last_updated'])
            age = (datetime.now() - last_updated).total_seconds() / 3600.0
            return age <= max_age_hours
        except:
            return False

# Global Instance
_cache_instance = None

def get_metadata_cache() -> ModelMetadataCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = ModelMetadataCache()
    return _cache_instance
