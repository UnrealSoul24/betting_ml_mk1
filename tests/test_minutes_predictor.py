import unittest
import os
import pandas as pd
import numpy as np
import torch
import joblib
from src.minutes_predictor import MinutesPredictor, MINUTES_MODEL_PATH, MINUTES_SCALER_PATH

class TestMinutesPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = MinutesPredictor()
        self.test_data_path = os.path.join(os.path.dirname(__file__), 'dummy_minutes_data.csv')
        
        # Create dummy data for training
        df = pd.DataFrame({
            'recent_min_avg': np.random.uniform(5, 40, 100),
            'recent_min_std': np.random.uniform(0, 10, 100),
            'INJURY_SEVERITY_TOTAL': np.random.randint(0, 5, 100),
            'STARS_OUT': np.random.randint(0, 3, 100),
            'OPP_ROLL_PACE': np.random.uniform(90, 110, 100),
            'IS_HOME': np.random.choice([0, 1], 100),
            'DAYS_REST': np.random.uniform(0, 7, 100),
            'season_min_avg': np.random.uniform(5, 40, 100),
            'MIN': np.random.uniform(5, 48, 100)
        })
        df.to_csv(self.test_data_path, index=False)

    def tearDown(self):
        if os.path.exists(self.test_data_path):
            os.remove(self.test_data_path)
        # We don't want to delete the actual model files if they exist, 
        # but for testing we might need to mock them or use temp paths.
        # For now, we'll assume the environment is set up for testing.

    def test_train_global_model(self):
        """Verify training completes and saves model/scaler."""
        # Backup existing model/scaler if they exist
        bak_model = MINUTES_MODEL_PATH + '.bak'
        bak_scaler = MINUTES_SCALER_PATH + '.bak'
        
        if os.path.exists(MINUTES_MODEL_PATH): os.rename(MINUTES_MODEL_PATH, bak_model)
        if os.path.exists(MINUTES_SCALER_PATH): os.rename(MINUTES_SCALER_PATH, bak_scaler)
        
        try:
            self.predictor.train_global_model(data_path=self.test_data_path)
            
            self.assertTrue(os.path.exists(MINUTES_MODEL_PATH))
            self.assertTrue(os.path.exists(MINUTES_SCALER_PATH))
            
            # Verify can load
            self.assertTrue(self.predictor._load_model())
            self.assertIsNotNone(self.predictor.model)
            self.assertIsNotNone(self.predictor.scaler)
        finally:
            # Restore backups
            if os.path.exists(bak_model):
                if os.path.exists(MINUTES_MODEL_PATH): os.remove(MINUTES_MODEL_PATH)
                os.rename(bak_model, MINUTES_MODEL_PATH)
            if os.path.exists(bak_scaler):
                if os.path.exists(MINUTES_SCALER_PATH): os.remove(MINUTES_SCALER_PATH)
                os.rename(bak_scaler, MINUTES_SCALER_PATH)

    def test_predict_with_high_injury_context(self):
        """High injury severity and stars out -> should influence minutes (usually up for remaining players)."""
        self.predictor.train_global_model(data_path=self.test_data_path)
        
        context_low = {
            'recent_min_avg': 20.0,
            'recent_min_std': 2.0,
            'INJURY_SEVERITY_TOTAL': 0.0,
            'STARS_OUT': 0.0,
            'OPP_ROLL_PACE': 100.0,
            'IS_HOME': 1.0,
            'DAYS_REST': 2.0,
            'season_min_avg': 20.0
        }
        
        context_high = context_low.copy()
        context_high['INJURY_SEVERITY_TOTAL'] = 10.0
        context_high['STARS_OUT'] = 2.0
        
        pred_low = self.predictor.predict(context_low)
        pred_high = self.predictor.predict(context_high)
        
        # Note: ML models might not always go 'up' unless trained on such patterns, 
        # but they should definitely differ.
        self.assertNotEqual(pred_low, pred_high)

    def test_fallback_when_model_missing(self):
        """Returns 30.0 when model file not found."""
        # Backup and remove model
        bak_model = MINUTES_MODEL_PATH + '.bak'
        if os.path.exists(MINUTES_MODEL_PATH): os.rename(MINUTES_MODEL_PATH, bak_model)
        
        try:
            predictor = MinutesPredictor() # Fresh instance
            pred = predictor.predict({})
            self.assertEqual(pred, 30.0)
        finally:
            if os.path.exists(bak_model):
                if os.path.exists(MINUTES_MODEL_PATH): os.remove(MINUTES_MODEL_PATH)
                os.rename(bak_model, MINUTES_MODEL_PATH)

    def test_clamping(self):
        """Output should be at least 0.0 (and we usually clamp to 5-48 in batch_predict)."""
        self.predictor.train_global_model(data_path=self.test_data_path)
        
        # Force a very low/high input if possible, or just check basic >= 0
        pred = self.predictor.predict({'recent_min_avg': 0.0, 'season_min_avg': 0.0})
        self.assertGreaterEqual(pred, 0.0)

    def test_missing_features(self):
        """Handles missing features in context dict using defaults (0.0)."""
        self.predictor.train_global_model(data_path=self.test_data_path)
        
        # Missing almost everything
        pred = self.predictor.predict({'recent_min_avg': 25.0})
        self.assertIsInstance(pred, float)
        self.assertGreater(pred, 0.0)

if __name__ == '__main__':
    unittest.main()
