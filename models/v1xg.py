# models/v1xg.py

import sqlite3
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestClassifier #v1
from xgboost import XGBClassifier #v1xg
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib
import os
import time
from tqdm import tqdm
tqdm.pandas()

def step_progress(desc):
    #print(f"\n🔧 {desc}")
    tqdm.write(f"{desc}...")
    time.sleep(0.2)  # Small delay so it's noticeable

# Load data
step_progress("Step 1: Loading from DB...")
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")
games = pd.read_sql("SELECT * FROM games", conn)
clubs = pd.read_sql("SELECT * FROM clubs", conn)
valuations = pd.read_sql("SELECT * FROM player_valuations", conn)
game_lineups = pd.read_sql("SELECT * FROM game_lineups", conn)

games['date'] = pd.to_datetime(games['date'], errors='coerce')


# Goalkeeper stats
step_progress("Step 2: Calculating goalkeeper clean sheets")
goalkeepers = game_lineups[
    (game_lineups['position'] == 'Goalkeeper') & 
    (game_lineups['type'] == 'starting_lineup')
]

merged_data = pd.merge(
    goalkeepers,
    games[['game_id', 'date', 'home_club_goals', 'away_club_goals', 'home_club_id', 'away_club_id']],
    on='game_id',
    how='left'
)

merged_data['date'] = pd.to_datetime(merged_data['date_y'], errors='coerce')
merged_data.drop(columns=['date_y'], inplace=True)

merged_data['clean_sheet'] = (
    (merged_data['home_club_goals'] == 0) & 
    (merged_data['club_id'] == merged_data['home_club_id'])
) | (
    (merged_data['away_club_goals'] == 0) & 
    (merged_data['club_id'] == merged_data['away_club_id'])
)

merged_data = merged_data.sort_values(['home_club_id', 'date'])

merged_data['home_gk_clean_sheets_last5'] = merged_data.groupby('home_club_id')['clean_sheet']\
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
merged_data['away_gk_clean_sheets_last5'] = merged_data.groupby('away_club_id')['clean_sheet']\
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

home_gk_features = merged_data[merged_data['club_id'] == merged_data['home_club_id']]\
    [['game_id', 'home_gk_clean_sheets_last5']].drop_duplicates()
away_gk_features = merged_data[merged_data['club_id'] == merged_data['away_club_id']]\
    [['game_id', 'away_gk_clean_sheets_last5']].drop_duplicates()

games = games.merge(home_gk_features, on='game_id', how='left')
games = games.merge(away_gk_features, on='game_id', how='left')

step_progress("Step 3: Preprocessing match data")

# Add form feature (last 5 matches) for home and away clubs
form_games = games[['game_id', 'date', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']].copy()

# Home team result
form_games['home_result'] = form_games.apply(
    lambda x: 1 if x['home_club_goals'] > x['away_club_goals']
    else 0.5 if x['home_club_goals'] == x['away_club_goals']
    else 0, axis=1
)
form_games['away_result'] = form_games.apply(
    lambda x: 1 if x['away_club_goals'] > x['home_club_goals']
    else 0.5 if x['home_club_goals'] == x['away_club_goals']
    else 0, axis=1
)

home_form = form_games[['game_id', 'date', 'home_club_id', 'home_result']].rename(
    columns={'home_club_id': 'club_id', 'home_result': 'result'})
away_form = form_games[['game_id', 'date', 'away_club_id', 'away_result']].rename(
    columns={'away_club_id': 'club_id', 'away_result': 'result'})

club_form = pd.concat([home_form, away_form])
club_form = club_form.sort_values(['club_id', 'date'])

club_form['form_last_5'] = club_form.groupby('club_id')['result']\
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())

home_form_feature = club_form[['game_id', 'club_id', 'form_last_5']].copy()
away_form_feature = home_form_feature.copy()

games = games.merge(
    home_form_feature.rename(columns={'club_id': 'home_club_id', 'form_last_5': 'home_form_last5'}),
    on=['game_id', 'home_club_id'],
    how='left'
)
games = games.merge(
    away_form_feature.rename(columns={'club_id': 'away_club_id', 'form_last_5': 'away_form_last5'}),
    on=['game_id', 'away_club_id'],
    how='left'
)

# Team value info
step_progress("Step 4: Merging team valuation data")
valuation_summary = (
    valuations.sort_values(['player_id', 'date'])
    .drop_duplicates(subset=['player_id', 'date'], keep='last')
    .groupby(['current_club_id', 'date'])['market_value_in_eur']
    .sum()
    .reset_index()
)
valuation_summary.rename(columns={'current_club_id': 'club_id', 'market_value_in_eur': 'total_value'}, inplace=True)

games['home_club_id'] = games['home_club_id'].astype('Int64')
games['away_club_id'] = games['away_club_id'].astype('Int64')
valuation_summary['club_id'] = valuation_summary['club_id'].astype('Int64')

games['date'] = pd.to_datetime(games['date'])
valuation_summary['date'] = pd.to_datetime(valuation_summary['date'])

games_home = games[['game_id', 'home_club_id', 'date']].rename(columns={'home_club_id': 'club_id'})
games_home = pd.merge_asof(
    games_home.sort_values('date'),
    valuation_summary.sort_values('date'),
    by='club_id', on='date', direction='backward'
)
games_home.rename(columns={'total_value': 'total_value_home'}, inplace=True)

games_away = games[['game_id', 'away_club_id', 'date']].rename(columns={'away_club_id': 'club_id'})
games_away = pd.merge_asof(
    games_away.sort_values('date'),
    valuation_summary.sort_values('date'),
    by='club_id', on='date', direction='backward'
)
games_away.rename(columns={'total_value': 'total_value_away'}, inplace=True)

games = games.merge(games_home[['game_id', 'total_value_home']], on='game_id', how='left')
games = games.merge(games_away[['game_id', 'total_value_away']], on='game_id', how='left')

# Formation strength
step_progress("Step 5: Calculating formation strength differences")
sorted_games = games.sort_values('date').copy()
sorted_games['home_goal_diff'] = sorted_games['home_club_goals'] - sorted_games['away_club_goals']
sorted_games['away_goal_diff'] = sorted_games['away_club_goals'] - sorted_games['home_club_goals']

sorted_games['home_formation_strength'] = sorted_games.groupby(
    ['home_club_id', 'home_club_formation']
)['home_goal_diff'].transform(lambda x: x.expanding().mean().shift(1))

sorted_games['away_formation_strength'] = sorted_games.groupby(
    ['away_club_id', 'away_club_formation']
)['away_goal_diff'].transform(lambda x: x.expanding().mean().shift(1))

games = games.merge(
    sorted_games[['game_id', 'home_formation_strength', 'away_formation_strength']],
    on='game_id', how='left'
)
games['home_formation_strength'] = games['home_formation_strength'].fillna(0)
games['away_formation_strength'] = games['away_formation_strength'].fillna(0)
games['formation_strength_diff'] = games['home_formation_strength'] - games['away_formation_strength']

# Club metadata
step_progress("Step 6: Final data preprocessing and feature engineering")
games = games.merge(clubs.add_prefix("home_"), left_on="home_club_id", right_on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), left_on="away_club_id", right_on="away_club_id", how="left")
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Match result target
def get_match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1
    else:
        return 2

games["result"] = games.progress_apply(get_match_result, axis=1)
games["position_diff"] = games["home_club_position"] - games["away_club_position"]
games["age_diff"] = games["home_average_age"] - games["away_average_age"]
games["seats_diff"] = games["home_stadium_seats"] - games["away_stadium_seats"]
games["nationals_diff"] = games["home_national_team_players"] - games["away_national_team_players"]
games["form_diff"] = games["home_form_last5"] - games["away_form_last5"]
games["clean_sheets_diff"] = games["home_gk_clean_sheets_last5"] - games["away_gk_clean_sheets_last5"]
games['value_diff'] = games['total_value_home'] - games['total_value_away']
games['form_x_position_home'] = games['home_form_last5'] * games['home_club_position']
games['form_x_position_away'] = games['away_form_last5'] * games['away_club_position']
games['value_per_age'] = games['value_diff'] / (games['age_diff'] + 1e-5)


# Updated feature list
features = [
    #'home_squad_size', 'home_average_age',
    #'away_squad_size', 'away_average_age', 
    'home_club_position', 'away_club_position',
    'attendance', 'position_diff',
    'age_diff', 'nationals_diff', 'seats_diff',
    'total_value_home', 'total_value_away', 
    'value_diff','form_diff','clean_sheets_diff',
    'home_formation_strength', 'away_formation_strength',
    'formation_strength_diff','form_x_position_home','form_x_position_away','value_per_age'
    #'home_gk_clean_sheets_last5', 'away_gk_clean_sheets_last5',
    #'home_form_last5', 'away_form_last5'
]

for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

games.dropna(subset=features + ["result"], inplace=True)

X = games[features]
y = games["result"]

# Train model
print("🤖 Training XGBoost Model...")
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#model = RandomForestClassifier(n_estimators=100, random_state=42)
#model.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(
    n_estimators=700,
    max_depth=10,
    scale_pos_weight=10,
    tree_method='auto',  # Use histogram-based CPU tree builder
    verbosity=1,
    n_jobs=-1  # Keep it low so it doesn't freeze your system
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "match_outcome_model_v1xg.pkl")
print("✅ Model saved to match_outcome_model_v1xg.pkl")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Home Win", "Draw", "Away Win"])
disp.plot()
plt.savefig("v1xg_confusion_matrix.png")
plt.close()

# Feature importance
importances = model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("v1xg Feature importances:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10,6))
plt.title("v1xg Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("v1xg_feature_importance.png")
plt.close()
