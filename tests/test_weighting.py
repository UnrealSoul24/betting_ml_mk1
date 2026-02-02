
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from weighting import ExponentialDecayWeighter

class TestExponentialDecayWeighter(unittest.TestCase):

    def test_calculate_weights_simple(self):
        # Create a simple DataFrame sorted by date
        df = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2026-01-05', '2026-01-03', '2026-01-01']),
            'PTS': [20, 15, 10]
        })
        
        weighter = ExponentialDecayWeighter(decay_factor=0.5)
        weights = weighter.calculate_weights(df, date_col='GAME_DATE')
        
        # Expected weights: 0.5^0, 0.5^1, 0.5^2 -> 1.0, 0.5, 0.25
        expected_weights = np.array([1.0, 0.5, 0.25])
        
        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_calculate_weights_unsorted(self):
        # Create an unsorted DataFrame
        dates = pd.to_datetime(['2026-01-01', '2026-01-05', '2026-01-03'])
        df = pd.DataFrame({
            'GAME_DATE': dates,
            'PTS': [10, 20, 15]
        })
        
        weighter = ExponentialDecayWeighter(decay_factor=0.5)
        weights = weighter.calculate_weights(df, date_col='GAME_DATE')
        
        # Map: 
        # '2026-01-05' (index 1) -> most recent -> 1.0
        # '2026-01-03' (index 2) -> 2nd recent -> 0.5
        # '2026-01-01' (index 0) -> oldest -> 0.25
        
        expected_weights = np.array([0.25, 1.0, 0.5])
        
        np.testing.assert_array_almost_equal(weights, expected_weights)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        weighter = ExponentialDecayWeighter()
        weights = weighter.calculate_weights(df)
        self.assertEqual(len(weights), 0)

if __name__ == '__main__':
    unittest.main()
