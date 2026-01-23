import pandas as pd
import requests
import io

def get_injury_report():
    """
    Fetches the latest NBA injury report from CBS Sports.
    Returns a dictionary: { 'PLAYER_NAME': 'STATUS' }
    Status values: 'Out', 'Questionable', 'Doubtful', 'Game Time Decision'
    """
    url = "https://www.cbssports.com/nba/injuries/"
    
    print(f"Fetching injury report from {url}...")
    
    try:
        # Use simple requests to get html, then pandas to parse
        # Adding headers to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        r = requests.get(url, headers=headers)
        
        if r.status_code != 200:
            print(f"Failed to fetch injuries. Status: {r.status_code}")
            return {}

        dfs = pd.read_html(io.StringIO(r.text))
        
        # CBS usually separates by team, so multiple tables.
        # We want to combine them all.
        injury_map = {}
        
        for df in dfs:
            if 'Player' in df.columns and 'Injury' in df.columns:
                # Iterate rows
                for _, row in df.iterrows():
                    player_name = str(row['Player']).strip()
                    status = str(row['Injury']).strip() # e.g. "Game Time Decision", "Out", "Knee"
                    
                    # Clean up player name (sometimes "Player Name POS")
                    # CBS format: "LeBron James SF"
                    # But usually just Name. Let's see.
                    # Actually CBS is usually "Player Name"
                    
                    # Normalize Status
                    status_lower = status.lower()
                    clean_status = "Active"
                    
                    if "out" in status_lower:
                        clean_status = "Out"
                    elif "doubtful" in status_lower:
                        clean_status = "Doubtful"
                    elif "questionable" in status_lower:
                        clean_status = "Questionable"
                    elif "decision" in status_lower or "gtd" in status_lower:
                        clean_status = "Questionable"
                    else:
                        # Check specific status columns
                        status_col = None
                        if 'Injury Status' in df.columns:
                            status_col = 'Injury Status'
                        elif 'Status' in df.columns:
                            status_col = 'Status'
                            
                        if status_col:
                            real_status = str(row[status_col]).lower()
                            if "out" in real_status: clean_status = "Out"
                            elif "quest" in real_status: clean_status = "Questionable"
                            elif "doubt" in real_status: clean_status = "Doubtful"
                            elif "day-to-day" in real_status: clean_status = "Questionable"
                            elif "decision" in real_status: clean_status = "Questionable"
                    
                    injury_map[player_name] = clean_status
                    
        print(f"Injury Report Loaded: {len(injury_map)} players found.")
        return injury_map

    except Exception as e:
        print(f"Error fetching injury report: {e}")
        return {}

if __name__ == "__main__":
    # Test
    report = get_injury_report()
    for p, s in report.items():
        if s == "Out":
            print(f"{p}: {s}")
