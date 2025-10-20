import pandas as pd
from itertools import combinations

# Load the dataset
file_path = "statsForDemo.csv"  # Update this path if needed
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