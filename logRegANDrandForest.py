import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import glob
import os
import pickle
import re

# Load base data with error handling
try:
    df_comparisons = pd.read_csv('data/team_comparisons.csv')
    df_team_avg = pd.read_csv('data/team_averages_2025.csv')
except FileNotFoundError as e:
    print(f"Error loading base data: {e}")
    exit()

# create a directory for images
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "money"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

# Load all game logs from the data folder with better error handling
game_logs = []
for file in glob.glob('data/gamelogs/*_gamelog.csv'):
    try:
        team_name = os.path.basename(file).split('_')[0]
        df = pd.read_csv(file)
        
        # Standardize column names (handle case sensitivity and spaces)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Check if required columns exist (using lowercase)
        required_columns = {'rslt', 'tm_pts', 'opp_pts', 'fg%', '3p%', 'date'}
        if not required_columns.issubset(df.columns):
            print(f"Warning: Missing columns in {file}. Found: {df.columns}")
            continue
            
        # Add team name column
        df['team'] = team_name
        game_logs.append(df)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if not game_logs:
    print("Warning: No game logs loaded. Using season averages only.")
    df_all_games = pd.DataFrame()
else:
    df_all_games = pd.concat(game_logs, ignore_index=True)

# Feature Engineering - Create recent performance metrics with robust error handling
def create_recent_features(team_name, n_games=10):
    try:
        team_games = df_all_games[df_all_games['team'].str.lower() == team_name.lower()]
        if len(team_games) == 0:
            return pd.Series()
            
        team_games = team_games.sort_values('date')
        recent = team_games.tail(n_games)
        
        # Calculate metrics with fallbacks
        return pd.Series({
            'last10_winpct': (recent['rslt'] == 'W').mean(),
            'last10_ppg': recent['tm_pts'].mean(),
            'last10_oppppg': recent['opp_pts'].mean(),
            'last10_fg%': recent['fg%'].mean(),
            'last10_3p%': recent['3p%'].mean(),
            'last5_streak': (recent['rslt'].tail(5) == 'W').sum(),
            'last10_orb': recent['orb'].mean() if 'orb' in recent.columns else np.nan,
            'last10_drb': recent['drb'].mean() if 'drb' in recent.columns else np.nan,
            'last10_ast': recent['ast'].mean() if 'ast' in recent.columns else np.nan,
            'last10_tov': recent['tov'].mean() if 'tov' in recent.columns else np.nan,
        })
    except Exception as e:
        print(f"Error processing {team_name}: {e}")
        return pd.Series()

# Create features for all teams with default values
team_features = {}
for team in df_team_avg['Team']:
    features = create_recent_features(team)
    if features.empty:
        # Default values when no game data available
        team_data = df_team_avg[df_team_avg['Team'].str.lower() == team.lower()].iloc[0]
        team_features[team] = {
            'last10_winpct': 0.5,
            'last10_ppg': team_data['PTS']/10,
            'last10_oppppg': team_data['opp_PTS']/10,
            'last10_fg%': team_data['FG%'],
            'last10_3p%': team_data['3P%'],
            'last5_streak': 0,
            'last10_orb': team_data['ORB']/10,
            'last10_drb': team_data['DRB']/10,
            'last10_ast': team_data['AST']/10,
            'last10_tov': team_data['TOV']/10,
        }
    else:
        team_features[team] = features.fillna(0).to_dict()

df_recent = pd.DataFrame.from_dict(team_features, orient='index').reset_index()
df_recent.columns = ['team'] + list(df_recent.columns[1:])

# Merge with team averages and fill NA values
df_enhanced = pd.merge(df_team_avg, df_recent, left_on='Team', right_on='team', how='left').drop('team', axis=1).fillna(0)

# Enhanced comparison function with error handling
def create_enhanced_comparison(row):
    try:
        team_a = row['Team A']
        team_b = row['Team B']
        
        # Get base stats with fallback
        a_stats = df_enhanced[df_enhanced['Team'].str.lower() == team_a.lower()].iloc[0]
        b_stats = df_enhanced[df_enhanced['Team'].str.lower() == team_b.lower()].iloc[0]
        
        # Create enhanced features with fallbacks
        new_features = {
            'recentwinpct_diff': a_stats.get('last10_winpct', 0) - b_stats.get('last10_winpct', 0),
            'ppg_diff': a_stats.get('last10_ppg', 0) - b_stats.get('last10_ppg', 0),
            'defense_diff': b_stats.get('last10_oppppg', 0) - a_stats.get('last10_oppppg', 0),
            'fg%_diff': a_stats.get('last10_fg%', 0) - b_stats.get('last10_fg%', 0),
            '3p%_diff': a_stats.get('last10_3p%', 0) - b_stats.get('last10_3p%', 0),
            'streak_diff': a_stats.get('last5_streak', 0) - b_stats.get('last5_streak', 0),
            'orb_diff': a_stats.get('last10_orb', 0) - b_stats.get('last10_orb', 0),
            'drb_diff': a_stats.get('last10_drb', 0) - b_stats.get('last10_drb', 0),
            'ast_diff': a_stats.get('last10_ast', 0) - b_stats.get('last10_ast', 0),
            'tov_diff': b_stats.get('last10_tov', 0) - a_stats.get('last10_tov', 0),  # More TOV is bad
        }
        
        return pd.Series(new_features)
    except Exception as e:
        print(f"Error comparing {row.get('Team A', '')} vs {row.get('Team B', '')}: {e}")
        return pd.Series()

# Apply to comparisons and fill NA values
enhanced_features = df_comparisons.apply(create_enhanced_comparison, axis=1).fillna(0)
df_enhanced_comparisons = pd.concat([df_comparisons, enhanced_features], axis=1)

# Prepare final dataset with selected features
available_features = ['FG% Diff', 'RPG Diff', 'Win-Loss Ratio Diff', 'PPG Diff']
additional_features = ['recentwinpct_diff', 'ppg_diff', 'defense_diff', 'fg%_diff', 
                      '3p%_diff', 'streak_diff', 'orb_diff', 'drb_diff', 'ast_diff', 'tov_diff']

# Only include additional features if they exist
features_to_use = available_features + [f for f in additional_features if f in df_enhanced_comparisons.columns]

# Train win prediction model
X = df_enhanced_comparisons[features_to_use]
y_win = (df_enhanced_comparisons['PPG Diff'] > 0).astype(int)

# Train point difference prediction model
y_diff = df_enhanced_comparisons['PPG Diff']

# Train models with error handling
try:
    # Split data
    X_train, X_test, y_win_train, y_win_test, y_diff_train, y_diff_test = train_test_split(
        X, y_win, y_diff, test_size=0.2, random_state=42
    )
    
    # Win prediction model
    win_model = LogisticRegression(max_iter=1000)
    win_model.fit(X_train, y_win_train)
    print(f"Win Prediction Accuracy: {win_model.score(X_test, y_win_test):.2%}")
    
    # Optimized RandomForest Regressor for point difference prediction
    diff_model = RandomForestRegressor(
        n_estimators=200,       # More trees for stability
        max_depth=8,            # Shallower trees to prevent overfitting
        min_samples_split=10,   # Require more data at each split
        min_samples_leaf=4,     # Prevent small leaves
        max_features='sqrt',    # Use sqrt(n_features), good for generalization
        random_state=42,
        n_jobs=-1               # Use all CPU cores
    )
    diff_model.fit(X_train, y_diff_train)
    y_diff_pred = diff_model.predict(X_test)
    print(f"Point Difference MAE: {mean_absolute_error(y_diff_test, y_diff_pred):.2f} points")

    # Calculate MSE and RMSE for RandomForest Regressor
    mse = mean_squared_error(y_diff_test, y_diff_pred)
    rmse = np.sqrt(mse)
    print(f"Point Difference MSE: {mse:.2f}")
    print(f"Point Difference RMSE: {rmse:.2f}")
    
    # Feature Importance Bar Chart
    feature_importance = pd.DataFrame({
        'Feature': features_to_use,
        'Importance': diff_model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Point Difference Prediction')
    plt.show()

    # Distribution Plot (Actual vs Predicted Scores)
    plt.figure(figsize=(10, 6))
    sns.histplot(y_diff_test, kde=True, color='blue', label='Actual')
    sns.histplot(y_diff_pred, kde=True, color='red', label='Predicted')
    plt.legend()
    plt.title('Distribution of Actual vs Predicted Point Differences')
    plt.xlabel('Point Difference')
    plt.ylabel('Frequency')
    plt.show()

    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save models using Pickle
    with open('models/win_model.pkl', 'wb') as f:
        pickle.dump(win_model, f)
    with open('models/diff_model.pkl', 'wb') as f:
        pickle.dump(diff_model, f)

except Exception as e:
    print(f"Error training models: {e}")
    exit()

# Enhanced prediction function
def predict_enhanced_matchup(team1, team2):
    try:
        def normalize_team_name(name):
            name = name.lower().strip()
            all_teams = df_enhanced['Team'].str.lower().unique()
            
            # First, try exact match
            if name in all_teams:
                return name
            
            # Then, try substring match
            substring_matches = [team for team in all_teams if name in team]
            if len(substring_matches) == 1:
                return substring_matches[0]
            
            # Finally, use fuzzy matching
            fuzzy_match = get_close_matches(name, all_teams, n=1, cutoff=0.4)
            if fuzzy_match:
                return fuzzy_match[0]
            
            raise ValueError(f"Team '{name}' not recognized. Please check the input.")
        
        def find_team_data(name):
            normalized_name = normalize_team_name(name)
            matches = df_enhanced[df_enhanced['Team'].str.lower() == normalized_name]
            if len(matches) > 0:
                return matches.iloc[0]
            
            matches = df_enhanced[df_enhanced['Team'].str.lower().str.contains(normalized_name)]
            if len(matches) > 0:
                return matches.iloc[0]
            
            raise ValueError(f"Team '{name}' not found in data")
        
        team_a = find_team_data(team1)
        team_b = find_team_data(team2)
        
        features = {
            'FG% Diff': team_a.get('FG%', 0) - team_b.get('FG%', 0),
            'RPG Diff': team_a.get('TRB', 0) - team_b.get('TRB', 0),
            'Win-Loss Ratio Diff': (team_a.get('W', 0)/(team_a.get('W', 0)+team_a.get('L', 1))) - 
                                 (team_b.get('W', 0)/(team_b.get('W', 0)+team_b.get('L', 1))),
            'PPG Diff': team_a.get('PTS', 0) - team_b.get('PTS', 0),
            'recentwinpct_diff': team_a.get('last10_winpct', 0) - team_b.get('last10_winpct', 0),
            'ppg_diff': team_a.get('last10_ppg', 0) - team_b.get('last10_ppg', 0),
            'defense_diff': team_b.get('last10_oppppg', 0) - team_a.get('last10_oppppg', 0),
            'fg%_diff': team_a.get('last10_fg%', 0) - team_b.get('last10_fg%', 0),
            '3p%_diff': team_a.get('last10_3p%', 0) - team_b.get('last10_3p%', 0),
            'streak_diff': team_a.get('last5_streak', 0) - team_b.get('last5_streak', 0),
            'orb_diff': team_a.get('last10_orb', 0) - team_b.get('last10_orb', 0),
            'drb_diff': team_a.get('last10_drb', 0) - team_b.get('last10_drb', 0),
            'ast_diff': team_a.get('last10_ast', 0) - team_b.get('last10_ast', 0),
            'tov_diff': team_b.get('last10_tov', 0) - team_a.get('last10_tov', 0),
        }
        
        feature_vector = pd.DataFrame([{k: features[k] for k in features_to_use}])

        # Predict win probability and point difference
        win_prob = win_model.predict_proba(feature_vector)[:, 1][0]
        win_confidence = win_prob if win_prob > 0.5 else 1 - win_prob
        point_diff = diff_model.predict(feature_vector)[0]
        
        prediction = {
            'winner': team1 if win_prob > 0.5 else team2,
            'win_confidence': win_confidence,
            'team1_score': team_a['PTS'] + point_diff/2,
            'team2_score': team_b['PTS'] - point_diff/2,
            'prediction_details': features
        }
        
        return prediction
    except Exception as e:
        print(f"Error predicting matchup: {e}")
        return None

from difflib import get_close_matches

# Enhanced normalize_team_name function
def normalize_team_name(name):
    """Normalize team names to a standard format."""
    name = name.lower().strip()
    all_teams = df_enhanced['Team'].str.lower().unique()
    
    # First, try exact match
    if name in all_teams:
        return name
    
    # Then, try substring match
    substring_matches = [team for team in all_teams if name in team]
    if len(substring_matches) == 1:
        return substring_matches[0]
    
    # Finally, use fuzzy matching
    fuzzy_match = get_close_matches(name, all_teams, n=1, cutoff=0.4)
    if fuzzy_match:
        return fuzzy_match[0]
    
    raise ValueError(f"Team '{name}' not recognized. Please check the input.")


# save_fig function for saving plots    
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=1000):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# calculate total money gained/lost
# change the csv file to calculate money gained for a specific period
# df = pd.read_csv('data/odds/3_2025_odds.csv')
df = pd.read_csv('data/odds/playoff_odds.csv')
ml_predicted = []
ml_actual = []
spread_predicted = []
spread_actual = []
ml_money = [100]
spread_money = [100]

# calculate running total for two bettors
i = 0
while i < len(df):
    team1 = df.loc[i, "team"]
    team2 = df.loc[i + 1, "team"]
    score1 = int(df.loc[i, "score"])
    score2 = int(df.loc[i + 1, "score"])
    money_line1 = int(df.loc[i, "best_ml_odds"])
    money_line2 = int(df.loc[i + 1, "best_ml_odds"])
    spread1 = df.loc[i, "best_spread_odds"]
    spread2 = df.loc[i + 1, "best_spread_odds"]

    # first bettor is placing only money line (ml) bets
    ml_actual.append(team1 if int(score1) > int(score2) else team2)
    prediction = predict_enhanced_matchup(team1, team2)
    ml_predicted.append(prediction['winner'])

    # if incorrect prediction subtract money
    if prediction['winner'] != ml_actual[-1]:
        new_money = ml_money[-1] - 10
        ml_money.append(new_money)
    # if correct prediction add money
    else:
        if team1 == ml_actual[-1]:
            if money_line1 > 0:
                new_money = ml_money[-1] + (10 * money_line1 / 100)
            else:
                new_money = ml_money[-1] + (10 * 100 / abs(money_line1))
        else:
            if money_line2 > 0:
                new_money = ml_money[-1] + (10 * money_line2 / 100)
            else:
                new_money = ml_money[-1] + (10 * 100 / abs(money_line2))
        
        ml_money.append(new_money)

    # second bettor is placing only spread bets
    # formatting string to convert to float later
    spread1 = re.sub("½", ".5", spread1)
    spread2 = re.sub("½", ".5", spread2)
    spread1_words = spread1.split()
    spread2_words = spread2.split()
    if spread1_words[0] == "PK":
        spread1_words[0] = "0"
    
    if spread2_words[0] == "PK":
        spread2_words[0] == "0"

    # calculate our predicted spread
    predicted_spread = abs(int(prediction['team1_score'] - prediction['team2_score']))
    spread_predicted.append(predicted_spread)

    # if our spread prediction is correct add money
    if float(spread1_words[0]) > 0 and predicted_spread < abs(float(spread1_words[0])):
        spread_actual.append(float(spread1_words[0]))
        if int(spread1_words[1]) > 0:
            new_money = spread_money[-1] + (10 * int(spread1_words[1]) / 100)
        else:
            new_money = spread_money[-1] + (10 * 100 / abs(int(spread1_words[1])))

        spread_money.append(new_money)
    # if our spread prediction is correct add money
    elif float(spread1_words[0]) < 0 and predicted_spread > abs(float(spread1_words[0])):
        spread_actual.append(float(spread1_words[0]))
        if int(spread1_words[1]) > 0:
            new_money = spread_money[-1] + (10 * int(spread1_words[1]) / 100)
        else:
            new_money = spread_money[-1] + (10 * 100 / abs(int(spread1_words[1])))

        spread_money.append(new_money)
    # if our spread prediction is incorrect subtract money
    else:
        spread_actual.append(float(spread1_words[0]))
        new_money = spread_money[-1] - 10
        spread_money.append(new_money)
    
    i += 2

# convert to data frame
del ml_money[0]
del spread_money[0]
data = {"ML Predicted": ml_predicted, "ML Actual": ml_actual, "ML Money": ml_money,
        "SP Predicted": spread_predicted, "SP Actual": spread_actual, "SP Money": spread_money}
money_df = pd.DataFrame(data)

start_row = pd.DataFrame({
    "ML Predicted": ["Start"],
    "ML Actual": ["Start"],
    "ML Money": [100],
    "SP Predicted": ["Start"],
    "SP Actual":["Start"],
    "SP Money": [100]
})

money_df = pd.concat([start_row, money_df], ignore_index=True)

# plotting money gained for money line predictions
plt.figure(figsize=(10, 6))
plt.plot(money_df.index, money_df['ML Money'], marker='', label='Money Over Time', color='blue')
plt.xlabel("Bet Number")
plt.ylabel("Money")
plt.title("Playoff Money Line Bets Money Over Time")
# plt.title("Regular Season Money Line Bets Money Over Time")
save_fig('playoff_ml_money_over_time')
# save_fig('rg_ml_money_over_time')

# plotting money gained for money line predictions
plt.figure(figsize=(10, 6))
plt.plot(money_df.index, money_df['SP Money'], marker='', label='Money Over Time', color='red')
plt.xlabel("Bet Number")
plt.ylabel("Money")
plt.title("Playoff Point Spread Bets Money Over Time")
# plt.title("Regular Season Point Spread Bets Money Over Time")
save_fig('playoff_sp_money_over_time')
# save_fig('rg_sp_money_over_time')

# User input loop for prediction
while True:
    print("\n" + "-"*40)
    try:
        team1_input = input("First team: ").strip()
        team2_input = input("Second team: ").strip()

        # Validate inputs
        if not team1_input or not team2_input:
            raise ValueError("Team names cannot be empty.")

        # Normalize team names
        team1 = normalize_team_name(team1_input)
        team2 = normalize_team_name(team2_input)

        # Check if the interpreted teams are the same
        if team1 == team2:
            raise ValueError(f"Teams must be different. Both inputs were interpreted as '{team1.title()}'.")

        # Predict matchup
        prediction = predict_enhanced_matchup(team1, team2)
        if prediction:
            print(f"\nPrediction: {prediction['winner'].title()} wins ({prediction['win_confidence']:.1%} confidence)")
            print(f"Predicted Final Score: {team1.title()} {round(prediction['team1_score'])} - {team2.title()} {round(prediction['team2_score'])}")
            print(f"Predicted Point Difference: {abs(round(prediction['team1_score']) - round(prediction['team2_score']))} points")

    except ValueError as e:
        print(f"Input Error: {e}")
    except Exception as e:
        print(f"Prediction Error: {e}")

    cont = input("\nPredict another matchup? (y/n): ").lower()
    if cont != 'y':
        break
