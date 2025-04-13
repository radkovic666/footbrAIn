import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
tqdm.pandas()  # activates progress_apply

# Connect to DB
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# Load tables
games = pd.read_sql("SELECT * FROM games", conn)
clubs = pd.read_sql("SELECT * FROM clubs", conn)
club_games = pd.read_sql("SELECT * FROM club_games", conn)

# Merge game date into club_games for use in form and coach stats
game_dates = games[['game_id', 'date']]
club_games = club_games.merge(game_dates, on='game_id', how='left')

# Merge club info into games
games = games.merge(clubs.add_prefix("home_"), left_on="home_club_id", right_on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), left_on="away_club_id", right_on="away_club_id", how="left")

# Drop rows with missing scores
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Create target label
def get_match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0  # Home Win
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1  # Draw
    else:
        return 2  # Away Win

games["result"] = games.apply(get_match_result, axis=1)

# Convert string positions to numbers
for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

# === Feature Engineering: Coach Win Rate & Recent Form ===

# Recent form ratio (last 5 matches: win count / 5)
def get_recent_form_ratio(club_id, match_date):
    #print(f"\rCalculating...", end='', flush=True)
    recent = club_games[(club_games['club_id'] == club_id) & (club_games['date'] < match_date)]
    recent = recent.sort_values(by='date', ascending=False).head(5)
    if len(recent) == 0:
        return np.nan
    return recent['is_win'].mean()

# Coach win rate up to that point
def get_coach_winrate(club_id, coach_name, match_date):
    coach_games = club_games[
        (club_games['club_id'] == club_id) &
        (club_games['own_manager_name'] == coach_name) &
        (club_games['date'] < match_date)
    ]
    if len(coach_games) == 0:
        return np.nan
    return coach_games['is_win'].mean()

# Apply features
games['home_recent_form'] = games.progress_apply(lambda row: get_recent_form_ratio(row['home_club_id'], row['date']), axis=1)
games['away_recent_form'] = games.progress_apply(lambda row: get_recent_form_ratio(row['away_club_id'], row['date']), axis=1)

games['home_coach_winrate'] = games.progress_apply(
    lambda row: get_coach_winrate(row['home_club_id'], row.get['home_manager_name'], row['date']), axis=1)
games['away_coach_winrate'] = games.progress_apply(
    lambda row: get_coach_winrate(row['away_club_id'], row.get['away_manager_name'], row['date']), axis=1)

# === Feature selection ===
features = [
    'home_squad_size', 'away_squad_size',
    'home_average_age', 'away_average_age',
    'home_club_position', 'away_club_position',
    'home_national_team_players', 'away_national_team_players',
    'home_foreigners_percentage', 'away_foreigners_percentage',
    'attendance',
    'home_recent_form', 'away_recent_form',
    'home_coach_winrate', 'away_coach_winrate'
]

# Drop missing values for features or label
games.dropna(subset=features + ['result'], inplace=True)

# Prepare data
X = games[features]
y = games['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Feature importance
print("\nFeature Importances:")
importances = model.feature_importances_
for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{f}: {imp:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "/home/magilinux/footpredict/models/match_outcome_model.pkl")
print("✅ Model saved to models/match_outcome_model.pkl")

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]  # sort descending

# Print them in order
print("Feature importances:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Optional: plot it
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

