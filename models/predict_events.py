import sqlite3
import pandas as pd
import argparse
import pickle
import joblib
import numpy as np
import json

# Load the saved model
model = joblib.load('events_outcome_model.pkl')

# Load feature names used during training
with open('feature_names.json') as f:
    feature_order = json.load(f)

# === Connect to DB ===
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

def get_avg_cards_per_referee(ref_name):
    query = """
        SELECT COUNT(*) * 1.0 / COUNT(DISTINCT game_id) AS avg_cards
        FROM game_events
        WHERE type = 'Cards' AND game_id IN (
            SELECT game_id FROM games WHERE referee = ?
        )
    """
    result = pd.read_sql_query(query, conn, params=(ref_name,))
    return result['avg_cards'].iloc[0] if not result.empty and pd.notna(result['avg_cards'].iloc[0]) else 3.5

def get_club_stats(club_name):
    query = """
        SELECT
            AVG(attendance) AS avg_attendance,
            AVG(CASE WHEN home_club_name = ? THEN home_club_position
                     WHEN away_club_name = ? THEN away_club_position ELSE NULL END) AS avg_position,
            (SELECT COUNT(*) * 1.0 / COUNT(DISTINCT game_id)
             FROM game_events
             WHERE type = 'Cards' AND club_id IN (
                 SELECT DISTINCT home_club_id FROM games WHERE home_club_name = ?
                 UNION
                 SELECT DISTINCT away_club_id FROM games WHERE away_club_name = ?
             )) AS avg_cards
        FROM games
        WHERE home_club_name = ? OR away_club_name = ?
    """
    df = pd.read_sql_query(query, conn, params=(club_name, club_name, club_name, club_name, club_name, club_name))
    if df.empty or df['avg_attendance'].isna().all():
        return {'avg_attendance': 35000, 'avg_position': 10, 'avg_cards': 2.1}
    return {
        'avg_attendance': df['avg_attendance'].iloc[0] or 35000,
        'avg_position': df['avg_position'].iloc[0] or 10,
        'avg_cards': df['avg_cards'].iloc[0] or 2.1
    }

def predict_event(home_team, away_team, referee):
    home = get_club_stats(home_team)
    away = get_club_stats(away_team)
    ref_cards = get_avg_cards_per_referee(referee)

    features_dict = {
        'attendance': (home['avg_attendance'] + away['avg_attendance']) / 2,
        'avg_cards_home': home['avg_cards'],
        'avg_cards_away': away['avg_cards'],
        'home_club_position': home['avg_position'],
        'away_club_position': away['avg_position'],
        'avg_cards_per_game_referee': ref_cards
    }

    # Ensure correct column order
    features = pd.DataFrame([features_dict])[feature_order]

    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    print("\n📊 Feature Data Used:")
    print(features.T)

    if prediction == 1:
        print(f"\n🟨 High probability of card-heavy match: {prob * 100:.2f}%")
    else:
        print(f"\n🟢 Low probability of card-heavy match: {prob * 100:.2f}%")

# === CLI Entry ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("home_team", type=str, help="Home team name")
    parser.add_argument("away_team", type=str, help="Away team name")
    parser.add_argument("--ref", type=str, default="Unknown", help="Referee name")

    args = parser.parse_args()
    predict_event(args.home_team, args.away_team, args.ref)
