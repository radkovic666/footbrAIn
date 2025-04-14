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

# Connect to database
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# Load required tables
games = pd.read_sql("SELECT * FROM games", conn)
clubs = pd.read_sql("SELECT * FROM clubs", conn)

# Load player valuations and calculate total market value per club (latest entries)
player_valuations = pd.read_sql("""
    SELECT 
        current_club_id, 
        SUM(market_value_in_eur) AS total_market_value 
    FROM (
        SELECT 
            player_id, 
            current_club_id, 
            market_value_in_eur,
            ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY date DESC) AS rn
        FROM player_valuations
    ) 
    WHERE rn = 1 
    GROUP BY current_club_id
""", conn)

# Rename for merging
player_valuations = player_valuations.rename(columns={'current_club_id': 'club_id'})

# Merge club data for home and away teams
games = games.merge(clubs.add_prefix("home_"), left_on="home_club_id", right_on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), left_on="away_club_id", right_on="away_club_id", how="left")

# Merge market values for home and away clubs
games = games.merge(
    player_valuations[['club_id', 'total_market_value']].rename(columns={'total_market_value': 'home_total_market_value'}),
    left_on='home_club_id',
    right_on='club_id',
    how='left'
)
games = games.merge(
    player_valuations[['club_id', 'total_market_value']].rename(columns={'total_market_value': 'away_total_market_value'}),
    left_on='away_club_id',
    right_on='club_id',
    how='left'
)

# Fill missing values with 0 (if no valuations exist for a club)
games['home_total_market_value'] = games['home_total_market_value'].fillna(0)
games['away_total_market_value'] = games['away_total_market_value'].fillna(0)

# Drop rows with missing values in original columns
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Create target label: 0 = home_win, 1 = draw, 2 = away_win
def get_match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1
    else:
        return 2

games["position_diff"] = games["home_club_position"] - games["away_club_position"]
games["age_diff"] = games["home_average_age"] - games["away_average_age"]
games["seats_diff"] = games["home_stadium_seats"] - games["away_stadium_seats"]
games["nationals_diff"] = games["home_national_team_players"] - games["away_national_team_players"]
games["market_value_diff"] = games["home_total_market_value"] - games["away_total_market_value"]  # Optional difference feature

games["result"] = games.progress_apply(get_match_result, axis=1)

# Feature selection (updated with market value features)
features = [
    'home_squad_size', 'home_average_age',
    'away_squad_size', 'away_average_age', 
    'home_club_position', 'away_club_position',
    'attendance', 'position_diff',
    'age_diff','nationals_diff', 'seats_diff',
    'home_total_market_value', 'away_total_market_value'  # New features added
]

# Convert positions to numeric
for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

# Drop rows with missing values after conversion
games.dropna(subset=features + ["result"], inplace=True)

X = games[features]
y = games["result"]

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate and save model (remainder of code unchanged)
# ... [rest of your existing code for evaluation and saving] ...

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "match_outcome_model.pkl")
print("✅ Model saved to models/match_outcome_model.pkl")

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Home Win", "Draw", "Away Win"])
disp.plot()
plt.savefig("v1_confusion_matrix.png")
plt.close()

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
plt.savefig("v1_feature_importance.png")
plt.close()

