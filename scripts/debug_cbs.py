import pandas as pd
import requests
import io

url = "https://www.cbssports.com/nba/injuries/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive"
}

print(f"Fetching {url}...")
try:
    r = requests.get(url, headers=headers)
    r.raise_for_status()
    print("Response First 1000 characters:")
    print(r.text[:1000])
    
    dfs = pd.read_html(io.StringIO(r.text))
    print(f"Found {len(dfs)} tables.")
    
    for i, df in enumerate(dfs):
        print(f"\n--- Table {i} ---")
        print("Columns:", df.columns.tolist())
        if not df.empty:
            print("First 5 rows:")
            print(df.head().to_string())
            
            if 'Player' in df.columns and 'Injury' in df.columns:
                print("\nSample 'Injury' column values:")
                print(df['Injury'].head(10).tolist())
                
except Exception as e:
    print(f"Error: {e}")
