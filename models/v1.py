# models/predict_outcomes.py

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os
from tqdm import tqdm
tqdm.pandas()  # activates progress_apply

def get_team_value(club_id, match_date):
    club_players = valuations[(valuations['current_club_id'] == club_id) & (valuations['date'] <= match_date)]
    # Keep only the latest valuation for each player
    latest_valuations = (
        club_players
        .sort_values(['player_id', 'date'])
        .drop_duplicates('player_id', keep='last')
    )
    return latest_valuations['market_value_in_eur'].sum()

# Connect to database
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# Load required tables
games = pd.read_sql("SELECT * FROM games", conn)
clubs = pd.read_sql("SELECT * FROM clubs", conn)
valuations = pd.read_sql("SELECT * FROM player_valuations", conn)
valuations['date'] = pd.to_datetime(valuations['date'])

# Ensure games have proper datetime
games['date'] = pd.to_datetime(games['date'])

# Calculate formation strengths
sorted_games = games.sort_values('date').copy()

# Calculate goal differences
sorted_games['home_goal_diff'] = sorted_games['home_club_goals'] - sorted_games['away_club_goals']
sorted_games['away_goal_diff'] = sorted_games['away_club_goals'] - sorted_games['home_club_goals']

# Calculate formation strengths using expanding mean shifted by 1
sorted_games['home_formation_strength'] = sorted_games.groupby(
    ['home_club_id', 'home_club_formation']
)['home_goal_diff'].transform(lambda x: x.expanding().mean().shift(1))

sorted_games['away_formation_strength'] = sorted_games.groupby(
    ['away_club_id', 'away_club_formation']
)['away_goal_diff'].transform(lambda x: x.expanding().mean().shift(1))

# Merge formation strengths back to original games DataFrame
games = games.merge(
    sorted_games[['game_id', 'home_formation_strength', 'away_formation_strength']],
    on='game_id',
    how='left'
)

# Fill NaN values for first occurrences of formations
# Use direct assignment instead of inplace
games['home_formation_strength'] = games['home_formation_strength'].fillna(0)
games['away_formation_strength'] = games['away_formation_strength'].fillna(0)

# Create formation strength difference feature
games['formation_strength_diff'] = games['home_formation_strength'] - games['away_formation_strength']

# Merge club data for home and away teams
games = games.merge(clubs.add_prefix("home_"), left_on="home_club_id", right_on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), left_on="away_club_id", right_on="away_club_id", how="left")

# Drop rows with missing values
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Create target label: 0 = home_win, 1 = draw, 2 = away_win
def get_match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1
    else:
        return 2

# Create features
games["position_diff"] = games["home_club_position"] - games["away_club_position"]
games["age_diff"] = games["home_average_age"] - games["away_average_age"]
games["seats_diff"] = games["home_stadium_seats"] - games["away_stadium_seats"]
games["nationals_diff"] = games["home_national_team_players"] - games["away_national_team_players"]

# Calculate team values
print("⏳ Outcome Prediction Model is learning...")
games['total_value_home'] = games.progress_apply(lambda row: get_team_value(row['home_club_id'], row['date']), axis=1)
games['total_value_away'] = games.progress_apply(lambda row: get_team_value(row['away_club_id'], row['date']), axis=1)
games['value_diff'] = games['total_value_home'] - games['total_value_away']

# Create target variable
games["result"] = games.progress_apply(get_match_result, axis=1)

# Feature selection
features = [
    'home_squad_size', 'home_average_age',
    'away_squad_size', 'away_average_age', 
    'home_club_position', 'away_club_position',
    'attendance', 'position_diff',
    'age_diff','nationals_diff', 'seats_diff',
    'total_value_home', 'total_value_away', 'value_diff',
    'home_formation_strength', 'away_formation_strength',
    'formation_strength_diff'
]

# Convert positions to numeric
for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

# Drop rows with missing values
games.dropna(subset=features + ["result"], inplace=True)

X = games[features]
y = games["result"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/match_outcome_model_v1.pkl")
print("✅ Model saved to models/match_outcome_model_v1.pkl")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Home Win", "Draw", "Away Win"])
disp.plot()
plt.savefig("v1_confusion_matrix.png")
plt.close()

# Feature importances
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Plot feature importances
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("v1_feature_importance.png")
plt.close()
