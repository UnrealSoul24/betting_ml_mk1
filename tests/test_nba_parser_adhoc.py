import sys
import os
# Adjust path to find src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.injury_service import _parse_nba_injuries
import json

def test_json_parsing():
    # Mock NBA stats JSON response
    mock_json = {
        "resultSets": [
            {
                "name": "InjuryReport",
                "headers": ["PLAYER_NAME", "TEAM_CITY", "STATUS", "COMMENT"],
                "rowSet": [
                    ["LeBron James", "Los Angeles", "Out", "Ankle"],
                    ["Anthony Davis", "Lou", "Questionable", "Back"]
                ]
            }
        ]
    }
    
    json_str = json.dumps(mock_json)
    result = _parse_nba_injuries(json_str)
    
    print(f"Parsed Result: {result}")
    assert "LeBron James" in result
    assert result["LeBron James"] == "Out"
    assert "Anthony Davis" in result
    assert result["Anthony Davis"] == "Questionable"
    print("✅ JSON Parsing Test Passed")

if __name__ == "__main__":
    test_json_parsing()
