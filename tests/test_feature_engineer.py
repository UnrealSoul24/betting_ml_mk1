
import unittest
import pandas as pd
import numpy as np
from src.feature_engineer import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def setUp(self):
        self.fe = FeatureEngineer()
        # Create dummy data with ALL required columns
        self.dummy_logs = pd.DataFrame({
            'GAME_DATE': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']),
            'GAME_ID': ['0012400001', '0012400002', '0012400003', '0012400004'],
            'PLAYER_ID': [1, 1, 1, 1],
            'PLAYER_NAME': ['Test Player', 'Test Player', 'Test Player', 'Test Player'],
            'TEAM_ID': [10, 10, 10, 10],
            'TEAM_ABBREVIATION': ['LAL', 'LAL', 'LAL', 'LAL'],
            'OPP_TEAM_ABBREVIATION': ['BOS', 'MIA', 'BOS', 'MIA'],
            'MATCHUP': ['LAL vs. BOS', 'LAL @ MIA', 'LAL @ BOS', 'LAL vs. MIA'],
            'WL': ['W', 'L', 'W', 'W'],
            'PTS': [10, 20, 15, 25],
            'REB': [5, 6, 7, 8],
            'AST': [2, 3, 4, 5],
            'MIN': [30, 32, 28, 35],
            'FGM': [4, 8, 6, 10],
            'FGA': [10, 15, 12, 18],
            'FG_PCT': [0.4, 0.53, 0.5, 0.56],
            'FG3M': [1, 2, 1, 3],
            'FG3A': [4, 5, 4, 6],
            'FG3_PCT': [0.25, 0.4, 0.25, 0.5],
            'FTM': [1, 2, 2, 2],
            'FTA': [2, 2, 3, 3],
            'FT_PCT': [0.5, 1.0, 0.67, 0.67],
            'OREB': [1, 1, 2, 2],
            'DREB': [4, 5, 5, 6],
            'STL': [1, 0, 1, 2],
            'BLK': [0, 1, 0, 1],
            'TOV': [2, 1, 2, 1],
            'PF': [2, 3, 2, 4],
            'PLUS_MINUS': [5, -3, 8, 12],
            'SEASON_YEAR': [2025, 2025, 2025, 2025]
        })
        self.dummy_pos = pd.DataFrame({
            'PLAYER_ID': [1],
            'SEASON_YEAR': [2025],
            'POSITION_SIMPLE': ['G']
        })

    def test_dvp_columns_exist(self):
        """Test that DvP calculation adds expected columns."""
        df = self.fe.calculate_dvp(self.dummy_logs.copy(), self.dummy_pos)
        
        # Check columns exist
        self.assertIn('ROLL_def_PTS', df.columns)
        self.assertIn('ROLL_def_REB', df.columns)
        self.assertIn('ROLL_def_AST', df.columns)
        self.assertIn('POSITION_SIMPLE', df.columns)

    def test_h2h_columns_exist(self):
        """Test that H2H calculation adds expected columns."""
        df = self.fe.calculate_h2h(self.dummy_logs.copy())
        
        # Check columns exist
        self.assertIn('H2H_PTS', df.columns)
        self.assertIn('H2H_REB', df.columns)
        self.assertIn('H2H_AST', df.columns)

    def test_rolling_stats_no_leakage(self):
        """Test that rolling stats use shift(1) to prevent data leakage."""
        df = self.dummy_logs.copy()
        df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
        
        # Manually compute expected ROLL_PTS with shift(1)
        # Row 0: NaN (no history)
        # Row 1: 10 (avg of row 0)
        # Row 2: 15 (avg of rows 0,1)
        # Row 3: 15 (avg of rows 0,1,2)
        
        # Apply rolling as done in feature_engineer.py
        df['ROLL_PTS'] = df.groupby('PLAYER_ID')['PTS'].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
        
        # Check that Row 1's ROLL_PTS equals Row 0's PTS (i.e. 10)
        self.assertEqual(df.iloc[1]['ROLL_PTS'], 10.0)
        
        # Check that Row 3's ROLL_PTS does NOT include Row 3's PTS (25)
        # Expected: (10 + 20 + 15) / 3 = 15.0
        self.assertEqual(df.iloc[3]['ROLL_PTS'], 15.0)

    def test_opponent_derived(self):
        """Test that opponent is correctly parsed from MATCHUP."""
        df = self.fe._derive_opponent(self.dummy_logs.copy())
        
        # Check opponent extracted correctly
        self.assertEqual(df.iloc[0]['OPP_TEAM_ABBREVIATION'], 'BOS')
        self.assertEqual(df.iloc[1]['OPP_TEAM_ABBREVIATION'], 'MIA')

if __name__ == '__main__':
    unittest.main()
