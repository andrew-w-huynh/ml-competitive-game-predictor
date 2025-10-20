import pandas as pd
import os
import time

# Official team codes used by Basketball Reference
team_codes = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP", "New York Knicks": "NYK", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTA", "Washington Wizards": "WAS"
}

# Scrape the game log for a team & a year
def scrape_game_log(team_name, year):

    team_code = team_codes.get(team_name)
    
    if not team_code:
        print(f"Team {team_name} not found.")
        return
    
    url = f'https://www.basketball-reference.com/teams/{team_code}/{year}/gamelog/'
    df_list = pd.read_html(url, header=[0, 1])
    df = df_list[0]
    
    # Flattening the table, BR tables have two sets of headers and we only want one.
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

    # Renaming the columns manually to these to undo the flattened titles.
    col_titles = [
        "Rk", "Gtm", "Date", "", "Opp_team", "Rslt", "Tm_PTS", "Opp_PTS", "OT", "FG", "FGA", "FG%", "3P", "3PA", "3P%", 
        "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF",
        "opp_FG", "opp_FGA", "opp_FG%", "opp_3P", "opp_3PA", "opp_3P%", "opp_2P", "opp_2PA", "opp_2P%", "opp_eFG%", 
        "opp_FT", "opp_FTA", "opp_FT%", "opp_ORB", "opp_DRB", "opp_TRB", "opp_AST", "opp_STL", "opp_BLK", 
        "opp_TOV", "opp_PF"
    ]
    df.columns = col_titles
    
    df = df.drop(df.index[-1])
    df = df[df["Rk"] != "Rk"]

    # Create gamelogs folder.
    if not os.path.exists('gamelogs'):
        os.makedirs('gamelogs')

    # Save DataFrame to CSV inside the 'gamelogs' folder
    file_path = f'gamelogs/{team_code}_{year}_gamelog.csv'
    df.to_csv(file_path, index=False)

    print(f"CSV file saved as '{file_path}'")
    
    return df

# Scrape game logs for every team in the parameter year. 7 second delay.
def scrape_all_teams(year):
    for team_name in team_codes.keys():
        scrape_game_log(team_name, year)
        time.sleep(7)  # Delay to avoid bot jail
        
def generate_full_game_log(year):
    all_games = []

    code_to_team = {v: k for k, v in team_codes.items()}

    for team_name, team_code in team_codes.items():
        file_path = f'gamelogs/{team_code}_{year}_gamelog.csv'

        if not os.path.exists(file_path):
            print(f"Game log not found for {team_name}. Run scrape_all_teams({year}) first.")
            continue

        df = pd.read_csv(file_path)

        if "Unnamed: 3" not in df.columns or "Opp_team" not in df.columns:
            print(f"Unexpected columns in {file_path}, skipping...")
            continue

        df["Home_Team"] = df.apply(lambda row: team_name if pd.isna(row["Unnamed: 3"]) else code_to_team.get(row["Opp_team"], row["Opp_team"]), axis=1)
        df["Away_Team"] = df.apply(lambda row: code_to_team.get(row["Opp_team"], row["Opp_team"]) if pd.isna(row["Unnamed: 3"]) else team_name, axis=1)

        cols = df.columns.tolist()
        cols.insert(cols.index("Rslt"), cols.pop(cols.index("Home_Team")))
        cols.insert(cols.index("Rslt"), cols.pop(cols.index("Away_Team")))
        df = df[cols]
        all_games.append(df)

    if all_games:
        full_game_log = pd.concat(all_games, ignore_index=True)
        os.makedirs("gamelogs", exist_ok=True)
        full_game_log.to_csv(f"gamelogs/full_game_log_{year}.csv", index=False)
        print(f"Full game log saved as 'gamelogs/full_game_log_{year}.csv'")
    else:
        print("No games found to compile.")
        
# Load team data from gamelogs
def load_team_data(team_code, year):
    file_path = f'gamelogs/{team_code}_{year}_gamelog.csv'
    
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File for {team_code} {year} not found. Run scrape_all_teams(year) first so gamelogs is populated.")
        return None

# Calculating each team's averages for a given df for use in aggregate_team_averages(year)
def calculate_team_averages(df):
    columns_to_avg = [
        "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%",
        "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "opp_FG", "opp_FGA", "opp_FG%", "opp_3P",
        "opp_3PA", "opp_3P%", "opp_2P", "opp_2PA", "opp_2P%", "opp_eFG%", "opp_FT", "opp_FTA", "opp_FT%",
        "opp_ORB", "opp_DRB", "opp_TRB", "opp_AST", "opp_STL", "opp_BLK", "opp_TOV", "opp_PF"
    ]
    averages = df[columns_to_avg].apply(pd.to_numeric, errors='coerce').mean()
    
    averages["W"] = (df["Rslt"].str.startswith("W")).sum()
    averages["L"] = (df["Rslt"].str.startswith("L")).sum()
    averages["PTS"] = pd.to_numeric(df["Tm_PTS"], errors='coerce').mean()
    averages["opp_PTS"] = pd.to_numeric(df["Opp_PTS"], errors='coerce').mean()
    
    return averages

# Aggregate team_averages_2025.csv for the parameter year 
def aggregate_team_averages(year):
    all_team_averages = []
    
    for team_name, team_code in team_codes.items():
        df = load_team_data(team_code, year)
        
        if df is not None:
            team_averages = calculate_team_averages(df)
            team_averages["Team"] = team_name
            all_team_averages.append(team_averages)
        
    aggregated_df = pd.DataFrame(all_team_averages)
    
    cols = list(aggregated_df.columns)
    cols = [cols[-1]] + cols[:-1]
    aggregated_df = aggregated_df[cols]
    
    # The misc stats that I cannot scrape for the life of me.
    stats_df = pd.read_csv(f'misc_stats_{year}.csv', usecols=["Team", "NRtg", "SRS", "VORP"])
    stats_df.rename(columns={"Team▲": "Team"}, inplace=True)    
    aggregated_df = aggregated_df.merge(stats_df, on="Team", how="left")
    
    aggregated_df.head()
    
    aggregated_df.to_csv(f'team_averages_{year}.csv', index=False)
    print(f"Aggregated averages saved to 'team_averages_{year}.csv'")


# Example usage: Scrape game logs for all teams in 2025
scrape_all_teams(2025)

# Example usage: Aggregate team_averages_2025.csv for 2025
aggregate_team_averages(2025)

# Example usage: Generate gamelog of all games in 2025 to full_game_log_{year}.csv.
generate_full_game_log(2025)

from itertools import combinations

# Load the dataset
file_path = "team_averages_2025.csv"  # Update this path if needed
df = pd.read_csv(file_path)

# Rename relevant columns for clarity
df = df.rename(columns={"Team▲": "Team", "PTS": "PPG", "TRB": "RPG"})
df["Win_Loss_Ratio"] = df["W"] / (df["W"] + df["L"])  # Calculate Win-Loss Ratio

# Remove "League Average" row
df = df[df["Team"] != "League Average"]

# Create all possible team pair combinations
team_combinations = list(combinations(df["Team"], 2))

# Compute differences for each pair
comparison_data = []
for teamA, teamB in team_combinations:
    teamA_stats = df[df["Team"] == teamA].iloc[0]
    teamB_stats = df[df["Team"] == teamB].iloc[0]

    comparison_data.append({
        "Team A": teamA,
        "Team B": teamB,
        "FG% Diff": teamA_stats["FG%"] - teamB_stats["FG%"],
        "RPG Diff": teamA_stats["RPG"] - teamB_stats["RPG"],
        "Win-Loss Ratio Diff": teamA_stats["Win_Loss_Ratio"] - teamB_stats["Win_Loss_Ratio"],
        "PPG Diff": (teamA_stats["PPG"]) - (teamB_stats["PPG"])
    })

# Convert to DataFrame
df_comparisons = pd.DataFrame(comparison_data)

# Save to CSV
df_comparisons.to_csv("team_comparisons.csv", index=False)

#print("Comparison data saved to team_comparisons.csv")
#print(df_comparisons)