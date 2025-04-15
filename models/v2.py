# models/v2.py

import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import joblib
import os
from tqdm import tqdm
import warnings
import xgboost as xgb

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
xgb.set_config(verbosity=0)
tqdm.pandas()

# Load data
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")
games = pd.read_sql("SELECT * FROM games", conn)
clubs = pd.read_sql("SELECT * FROM clubs", conn)
valuations = pd.read_sql("SELECT * FROM player_valuations", conn)

# Preprocessing valuations
print(games['home_club_id'].isna().sum(), "NaNs in home_club_id")
print(games['away_club_id'].isna().sum(), "NaNs in away_club_id")
print("⏳ Preprocessing player valuations...")

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

# Prepare home & away valuations
games_home = games[['game_id', 'home_club_id', 'date']].rename(columns={'home_club_id': 'club_id'})
games_home = pd.merge_asof(
    games_home.sort_values('date'),
    valuation_summary.sort_values('date'),
    by='club_id', on='date', direction='backward'
).rename(columns={'total_value': 'total_value_home'})

games_away = games[['game_id', 'away_club_id', 'date']].rename(columns={'away_club_id': 'club_id'})
games_away = pd.merge_asof(
    games_away.sort_values('date'),
    valuation_summary.sort_values('date'),
    by='club_id', on='date', direction='backward'
).rename(columns={'total_value': 'total_value_away'})

games = games.merge(games_home[['game_id', 'total_value_home']], on='game_id', how='left')
games = games.merge(games_away[['game_id', 'total_value_away']], on='game_id', how='left')
games['value_diff'] = games['total_value_home'] - games['total_value_away']

# Goal differences & formation strength
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

# Merge club info
games = games.merge(clubs.add_prefix("home_"), on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), on="away_club_id", how="left")
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Match result
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

features = [
    'home_squad_size', 'home_average_age',
    'away_squad_size', 'away_average_age',
    'home_club_position', 'away_club_position',
    'attendance', 'position_diff',
    'age_diff', 'nationals_diff', 'seats_diff',
    'total_value_home', 'total_value_away', 'value_diff',
    'home_formation_strength', 'away_formation_strength',
    'formation_strength_diff'
]

for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

games.dropna(subset=features + ["result"], inplace=True)

X = games[features]
y = games["result"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost with RandomizedSearchCV
xgb_model = XGBClassifier(eval_metric='mlogloss', random_state=42)

param_dist = {
    'n_estimators': [300, 500, 800],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_model, param_distributions=param_dist,
    n_iter=10, scoring='accuracy', cv=3, verbose=1, n_jobs=-1, random_state=42
)
xgb_search.fit(X_train, y_train)

# Random Forest model
rf_model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)

# Soft voting classifier
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_search.best_estimator_), ('rf', rf_model)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = voting_clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(voting_clf, "match_outcome_model_v2.pkl")
print("✅ Model saved to match_outcome_model_v2.pkl")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["Home Win", "Draw", "Away Win"])
disp.plot()
plt.savefig("v2_confusion_matrix.png")
plt.close()

# Feature importances (from XGBoost only)
importances = xgb_search.best_estimator_.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

print("Feature importances (from XGBoost):")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

plt.figure(figsize=(10,6))
plt.title("XGBoost Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("v2_feature_importance.png")
plt.close()
