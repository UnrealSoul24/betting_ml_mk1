from nba_api.stats.endpoints import playercareerstats

# Nikola Jokić
career = playercareerstats.PlayerCareerStats(player_id='203999')

# pandas data frames (optional: pip install pandas)
career.season_totals_regular_season.get_data_frame()

# json
print(career.get_json())

# dictionary
career.get_dict()