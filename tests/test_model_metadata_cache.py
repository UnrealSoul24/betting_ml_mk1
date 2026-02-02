import unittest
import os
import json
import time
import threading
from datetime import datetime, timedelta
from src.model_metadata_cache import ModelMetadataCache

class TestModelMetadataCache(unittest.TestCase):
    def setUp(self):
        self.test_cache_path = os.path.join(os.path.dirname(__file__), 'temp_metadata.json')
        if os.path.exists(self.test_cache_path):
            os.remove(self.test_cache_path)
        self.cache = ModelMetadataCache(filepath=self.test_cache_path)

    def tearDown(self):
        if os.path.exists(self.test_cache_path):
            os.remove(self.test_cache_path)

    def test_update_and_get(self):
        """Update metadata, retrieve correctly."""
        player_id = 12345
        updates = {'status': 'active', 'accuracy': 0.85}
        self.cache.update_metadata(player_id, updates)
        
        meta = self.cache.get_metadata(player_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta['status'], 'active')
        self.assertEqual(meta['accuracy'], 0.85)
        self.assertIn('last_updated', meta)
        self.assertIn('created_at', meta)
        self.assertEqual(meta['update_count'], 1)

    def test_persistence(self):
        """Save to JSON, reload, verify data intact."""
        player_id = 67890
        self.cache.update_metadata(player_id, {'status': 'cached'})
        
        # Create new cache instance pointing to same file
        new_cache = ModelMetadataCache(filepath=self.test_cache_path)
        meta = new_cache.get_metadata(player_id)
        self.assertIsNotNone(meta)
        self.assertEqual(meta['status'], 'cached')

    def test_is_fresh(self):
        """Correctly identifies fresh (<24h) vs stale (>24h)."""
        player_id = 111
        # Fresh
        self.cache.update_metadata(player_id, {'status': 'active'})
        self.assertTrue(self.cache.is_fresh(player_id))
        
        # Stale (manually override for testing)
        with self.cache.lock:
            stale_time = (datetime.now() - timedelta(hours=25)).isoformat()
            self.cache.metadata[str(player_id)]['last_updated'] = stale_time
            self.cache._save()
            
        self.assertFalse(self.cache.is_fresh(player_id))

    def test_get_status(self):
        """Returns correct status strings."""
        self.cache.set_training_status(222, 'training')
        self.assertEqual(self.cache.get_status(222), 'training')
        
        self.cache.set_training_status(333, 'failed')
        self.assertEqual(self.cache.get_status(333), 'failed')
        
        self.assertEqual(self.cache.get_status(999), 'unknown')

    def test_thread_safety(self):
        """Concurrent updates don't corrupt cache."""
        def worker(pid):
            for i in range(10):
                self.cache.update_metadata(pid, {f'val_{i}': i})

        threads = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
            
        for t in threads:
            t.join()
            
        all_meta = self.cache.get_all_metadata()
        self.assertEqual(len(all_meta), 10)

    def test_corrupted_json(self):
        """Handles corrupted file gracefully."""
        with open(self.test_cache_path, 'w') as f:
            f.write("invalid json content")
            
        # Re-init cache
        self.cache = ModelMetadataCache(filepath=self.test_cache_path)
        self.assertEqual(self.cache.metadata, {})

    def test_missing_file(self):
        """Initializes empty cache when file missing."""
        if os.path.exists(self.test_cache_path):
            os.remove(self.test_cache_path)
            
        self.cache = ModelMetadataCache(filepath=self.test_cache_path)
        self.assertEqual(self.cache.metadata, {})

if __name__ == '__main__':
    unittest.main()
