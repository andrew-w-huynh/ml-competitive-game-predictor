import logging
from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import pickle
from difflib import get_close_matches

# Load models and data
with open('models/win_model.pkl', 'rb') as f:
    win_model = pickle.load(f)
with open('models/diff_model.pkl', 'rb') as f:
    diff_model = pickle.load(f)

df_enhanced = pd.read_csv('data/team_averages_2025.csv')

# Flask app setup
app = Flask(__name__)

# Serve static files from the 'logos' directory
@app.route('/logos/<path:filename>')
def serve_logo(filename):
    return send_from_directory('templates/logos', filename)

logging.getLogger('werkzeug').setLevel(logging.ERROR)  # Suppress Flask's default logging

# Normalize team names
def normalize_team_name(name):
    name = name.lower().strip()
    all_teams = df_enhanced['Team'].str.lower().unique()
    
    # Exact match
    if name in all_teams:
        return name
    
    # Substring match
    substring_matches = [team for team in all_teams if name in team]
    if len(substring_matches) == 1:
        return substring_matches[0]
    
    # Fuzzy matching
    fuzzy_match = get_close_matches(name, all_teams, n=1, cutoff=0.4)
    if fuzzy_match:
        return fuzzy_match[0]
    
    raise ValueError(f"Team '{name}' not recognized. Please check the input.")

# Predict matchup
def predict_matchup(team1, team2):
    try:
        def find_team_data(name):
            matches = df_enhanced[df_enhanced['Team'].str.lower() == name]
            if len(matches) > 0:
                return matches.iloc[0]
            raise ValueError(f"Team '{name}' not found in data")
        
        team_a = find_team_data(team1)
        team_b = find_team_data(team2)
        
        # Ensure all features used during training are included
        features = {
            'FG% Diff': team_a['FG%'] - team_b['FG%'],
            'RPG Diff': team_a['TRB'] - team_b['TRB'],
            'Win-Loss Ratio Diff': (team_a['W'] / (team_a['W'] + team_a['L'])) - 
                                   (team_b['W'] / (team_b['W'] + team_b['L'])),
            'PPG Diff': team_a['PTS'] - team_b['PTS'],
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
        
        feature_vector = pd.DataFrame([features])
        win_prob = win_model.predict_proba(feature_vector)[:, 1][0]
        point_diff = diff_model.predict(feature_vector)[0]
        
        return {
            'winner': str(team1).title() if win_prob > 0.5 else str(team2).title(),
            'win_confidence': win_prob if win_prob > 0.5 else 1 - win_prob,
            'team1_score': team_a['PTS'] + point_diff / 2,
            'team2_score': team_b['PTS'] - point_diff / 2,
        }
    except Exception as e:
        return {'error': str(e)}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        team1 = normalize_team_name(request.form['team1'])
        team2 = normalize_team_name(request.form['team2'])
        if team1 == team2:
            return jsonify({'error': 'Teams must be different.'})
        prediction = predict_matchup(team1, team2)
        return jsonify(prediction)
    except ValueError as e:
        return jsonify({'error': str(e)})
    except Exception as e:
        return jsonify({'error': 'An unexpected error occurred.'})

if __name__ == '__main__':
    print("App initialized. Located at http://localhost:5000")  # Print only once
    app.run(debug=True, use_reloader=False)  # Disable Flask's reloader