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
tqdm.pandas()

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
games["position_diff"] = games["home_club_position"] - games["away_club_position"]
games["age_diff"] = games["home_average_age"] - games["away_average_age"]
games["nationals_diff"] = games["home_national_team_players"] - games["away_national_team_players"]
games["seats_diff"] = games["home_stadium_seats"] - games["away_stadium_seats"]

games["result"] = games.apply(get_match_result, axis=1)

# Convert string positions to numbers
for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

# === Feature Engineering: Coach Win Rate & Recent Form ===

def get_recent_form_ratio(club_id, match_date):
    recent = club_games[(club_games['club_id'] == club_id) & (club_games['date'] < match_date)]
    recent = recent.sort_values(by='date', ascending=False).head(5)
    return recent['is_win'].mean() if len(recent) > 0 else np.nan

# === Caching ===
CACHE_FILE = "/home/magilinux/footpredict/data/games_with_features.csv"

if os.path.exists(CACHE_FILE):
    print("📥 Loading cached features...")
    cached = pd.read_csv(CACHE_FILE)
    new_games = games[~games['game_id'].isin(cached['game_id'])]
    print(f"🆕 Found {len(new_games)} new games to process.")
else:
    print("🆕 No cache found. Processing all games...")
    cached = pd.DataFrame()
    new_games = games

# Only process if we have new games
if not new_games.empty:
    def extract_features(row):
        return pd.Series({
            'home_recent_form': get_recent_form_ratio(row['home_club_id'], row['date']),
            'away_recent_form': get_recent_form_ratio(row['away_club_id'], row['date']),
        })

    print("⚙️  Extracting features...")
    new_feats = new_games.progress_apply(extract_features, axis=1)
    new_games = pd.concat([new_games.reset_index(drop=True), new_feats], axis=1)

    # Merge new with cached
    full_games = pd.concat([cached, new_games], ignore_index=True)
    full_games.to_csv(CACHE_FILE, index=False)
    print("✅ Cache updated.")
else:
    full_games = cached
    print("✅ Using full cache. No updates.")

# === Feature selection ===
features = [
    'home_squad_size', 'away_squad_size',
    'home_average_age', 'away_average_age',
    #'home_club_position', 'away_club_position',
    'home_national_team_players', 'away_national_team_players',
	'attendance',
    'home_recent_form', 'away_recent_form',
    #'home_coach_winrate', 'away_coach_winrate',
    'position_diff','age_diff','nationals_diff',
    'seats_diff'
]

# Drop missing values
full_games.dropna(subset=features + ['result'], inplace=True)

# Prepare data
X = full_games[features]
y = full_games['result']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Feature importance
print("\n🔍 Feature Importances:")
importances = model.feature_importances_
for f, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    print(f"{f}: {imp:.4f}")

# Save model
os.makedirs("/home/magilinux/footpredict/models", exist_ok=True)
joblib.dump(model, "/home/magilinux/footpredict/models/match_outcome_model.pkl")
print("✅ Model saved to match_outcome_model.pkl")

# Plot Feature Importance Graph
plt.figure(figsize=(10,6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[np.argsort(importances)[::-1]], align="center")
plt.xticks(range(len(importances)), [features[i] for i in np.argsort(importances)[::-1]], rotation=45)
plt.tight_layout()
plt.show()

#Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Home Win", "Draw", "Away Win"])
disp.plot()
plt.show()
