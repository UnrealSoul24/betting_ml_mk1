import pandas as pd
import numpy as np

class ExponentialDecayWeighter:
    """
    Calculates sample weights using exponential decay based on recency.
    Most recent game gets weight 1.0. Older games decay by decay_factor ^ games_ago.
    """
    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor

    def calculate_weights(self, df: pd.DataFrame, date_col: str = 'GAME_DATE') -> np.ndarray:
        """
        Calculates weights for each row in the DataFrame.
        
        Args:
            df: DataFrame containing the data. Contains a date column.
            date_col: Name of the column containing the date.
            
        Returns:
            np.ndarray: Array of weights corresponding to the rows in df.
        """
        if df.empty:
            return np.array([])
            
        # Ensure we are working with datetime objects if not already
        # We don't strictly need to convert to datetime if we just sort, 
        # but it's safer for "most recent" logic.
        # Assuming the caller might have already sorted, but we should not rely on it for the calculation logic 
        # to be robust. However, to return weights matching the input DF order, we should compute 
        # "games ago" based on the sort order, then map back to original index.
        
        # 1. Create a copy to avoid modifying original
        temp_df = df.copy()
        
        # 2. Sort by date descending (most recent first)
        temp_df = temp_df.sort_values(date_col, ascending=False)
        
        # 3. Assign "games_ago" counter (0, 1, 2, ...)
        # resetting index to get a simple range 0..N-1
        # But we need to preserve original index to map weights back
        
        # Generates 0, 1, 2, ... N-1
        games_ago = np.arange(len(temp_df))
        
        # 4. Calculate Weights
        weights = self.decay_factor ** games_ago
        
        # 5. Map weights back to original order
        # temp_df.index contains the original indices. 
        # We create a series with index=original_index, values=weights
        weight_series = pd.Series(weights, index=temp_df.index)
        
        # 6. Reindex to match input df's order exactly
        final_weights = weight_series.reindex(df.index).fillna(0.0).values
        
        return final_weights
