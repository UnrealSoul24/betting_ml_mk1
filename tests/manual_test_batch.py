import sys
import os
import asyncio
import pandas as pd
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.batch_predict import BatchPredictor

async def main():
    print("Initializing BatchPredictor...")
    bp = BatchPredictor()
    
    mock_list = [
        {'pid': 2544, 'pname': 'LeBron James', 'team_id': 1610612747, 'opp_id': 1610612744, 'is_home': True}
    ]
    
    print("Running analyze_today_batch...")
    try:
        results = await bp.analyze_today_batch(date_input='2025-02-02', execution_list=mock_list)
        
        # Unpack results if dictionary
        if isinstance(results, dict):
            res_list = results.get('predictions', [])
        else:
            res_list = results

        if not res_list:
            print("No results returned.")
            return

        print(f"\nGenerated {len(res_list)} predictions.\n")
        
        first = res_list[0]
        print("--- Keys in Result ---")
        print(first.keys())
        
        with open("keys.txt", "w", encoding="utf-8") as f:
            f.write(str(list(first.keys())))
            f.write("\n")
            f.write(f"Sample: {first}")

        print("\n--- Sample Prediction ---")
        print(f"Player: {first['PLAYER_NAME']}")
        print(f"Pred Minutes: {first['PRED_MIN']}")
        print(f"Pred PTS/Min: {first['PRED_PTS_PER_MIN']}")
        print(f"Min Delta Pct: {first.get('MIN_DELTA_PCT', 'MISSING')}")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        with open("error.log", "w") as f:
            traceback.print_exc(file=f)
        print("Traceback written to error.log")

if __name__ == "__main__":
    asyncio.run(main())
