import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import glob
import os

# Load base data with error handling
try:
    df_comparisons = pd.read_csv('data/team_comparisons.csv')
    df_team_avg = pd.read_csv('data/team_averages_2025.csv')
except FileNotFoundError as e:
    print(f"Error loading base data: {e}")
    exit()

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
    
    # Ensure the models directory exists
    os.makedirs('models', exist_ok=True)

    # Save models using Pickle
    with open('models/win_model.pkl', 'wb') as f:
        pickle.dump(win_model, f)
    with open('models/diff_model.pkl', 'wb') as f:
        pickle.dump(diff_model, f)
    print("Models saved successfully.")

except Exception as e:
    print(f"Error training models: {e}")
    exit()
