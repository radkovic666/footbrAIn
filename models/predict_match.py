# predict_match.py

import argparse
import sqlite3
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load trained model
model = joblib.load("match_outcome_model.pkl")

# Connect to database
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Predict football match outcome.")
parser.add_argument("home_team", type=str, help="Home club name")
parser.add_argument("away_team", type=str, help="Away club name")
parser.add_argument("--ref", type=str, help="Referee name (optional)", default="")
parser.add_argument("--venue", type=str, help="Stadium/Venue name", default="")
args = parser.parse_args()

# Load club data
clubs = pd.read_sql("SELECT * FROM clubs", conn)
games = pd.read_sql("SELECT * FROM games", conn)

# Match home and away teams
home = clubs[clubs["name"] == args.home_team]
away = clubs[clubs["name"] == args.away_team]

if home.empty or away.empty:
    print("❌ Club not found in database. Check names.")
    exit()

# Attendance and seats (fallbacks if venue not found)
attendance = home.iloc[0]["stadium_seats"] #if --venue != "Unknown" else 35000
seats_diff = home.iloc[0]["stadium_seats"] - away.iloc[0]["stadium_seats"]
# Load your existing home and away club data first (assumed done)
# e.g., home = clubs[clubs["name"] == home_name]
#       away = clubs[clubs["name"] == away_name]

# Function to infer club position
def get_latest_club_position(club_id):
    # Get recent games involving this club
    club_games = games[
        (games["home_club_id"] == club_id) | (games["away_club_id"] == club_id)
    ].sort_values(by="date", ascending=False)

    for _, row in club_games.iterrows():
        if row["home_club_id"] == club_id and pd.notna(row["home_club_position"]):
            return pd.to_numeric(row["home_club_position"], errors="coerce")
        elif row["away_club_id"] == club_id and pd.notna(row["away_club_position"]):
            return pd.to_numeric(row["away_club_position"], errors="coerce")

    return None  # fallback if not found

# Extract positions
home_position = get_latest_club_position(home.iloc[0]["club_id"])
away_position = get_latest_club_position(away.iloc[0]["club_id"])

# Compute position difference (if both available)
position_diff = home_position - away_position if pd.notna(home_position) and pd.notna(away_position) else None



# Calculate derived features
features = {
    'home_squad_size': home.iloc[0]['squad_size'],
    'home_average_age': home.iloc[0]['average_age'],
    'away_squad_size': away.iloc[0]['squad_size'],
    'away_average_age': away.iloc[0]['average_age'],
    'home_club_position': home_position,
    'away_club_position': away_position,
    'attendance': attendance,
    'position_diff': position_diff,
    'age_diff': home.iloc[0]['average_age'] - away.iloc[0]['average_age'],
    'nationals_diff': home.iloc[0]['national_team_players'] - away.iloc[0]['national_team_players'],
    'seats_diff': home.iloc[0]['stadium_seats'] - away.iloc[0]['stadium_seats']
}

# Convert to DataFrame for prediction
X_input = pd.DataFrame([features])

# Predict outcome
prediction = model.predict(X_input)[0]
probs = model.predict_proba(X_input)[0]

result_map = {
    0: "🏠 Home Win",
    1: "🤝 Draw",
    2: "🚗 Away Win"
}

# Output results
print("\n📊 Match Prediction Summary:")
print(f"🏟️  {args.home_team} vs {args.away_team}")
print(f"📌 Venue: {args.venue or 'N/A'}")
print(f"🧑‍⚖️ Referee: {args.ref or 'N/A'}")

print("\n📈 Predicted Outcome:", result_map[prediction])
print(f"🟢 Home Win: {probs[0]*100:.1f}%")
print(f"🟡 Draw: {probs[1]*100:.1f}%")
print(f"🔴 Away Win: {probs[2]*100:.1f}%")

print("\n📊 Features Used:")
for k, v in features.items():
    print(f"  {k}: {v}")
