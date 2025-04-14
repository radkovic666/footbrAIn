# build_events_outcome_model.py

import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt


# === Connect to DB ===
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# === Load required tables ===
games = pd.read_sql_query("SELECT * FROM games", conn)
events = pd.read_sql_query("SELECT * FROM game_events", conn)

# === Filter Cards from Events ===
cards = events[events['type'] == 'Cards']

# === Count Cards per Match ===
card_counts = cards.groupby('game_id').size().reset_index(name='num_cards')

# === Binary Target: Was the match card-rich (3+)? ===
card_counts['target'] = (card_counts['num_cards'] >= 3).astype(int)

# === Base Game Features to Use ===
games_features = games[[
    'game_id', 'referee', 'attendance',
    'home_club_id', 'away_club_id',
    'home_club_position', 'away_club_position'
]]

# === Merge Labels with Game Features ===
df = pd.merge(card_counts, games_features, on='game_id', how='inner')

# === Referee Aggression (avg cards per game) ===
cards_with_ref = cards.merge(games[['game_id', 'referee']], on='game_id', how='left')

# Calculate referee aggression
ref_cards = cards_with_ref.groupby('referee').size()
ref_games = games.groupby('referee').size()
ref_aggression = (ref_cards / ref_games).reset_index()
ref_aggression.columns = ['referee', 'avg_cards_per_game_referee']

df = df.merge(ref_aggression, on='referee', how='left')

# === Club Aggression (home) ===
home_cards = cards.merge(games[['game_id', 'home_club_id']], on='game_id')
home_aggression = home_cards.groupby('home_club_id').size() / games.groupby('home_club_id').size()
home_aggression = home_aggression.reset_index()
home_aggression.columns = ['home_club_id', 'avg_cards_home']
df = df.merge(home_aggression, on='home_club_id', how='left')

# === Club Aggression (away) ===
away_cards = cards.merge(games[['game_id', 'away_club_id']], on='game_id')
away_aggression = away_cards.groupby('away_club_id').size() / games.groupby('away_club_id').size()
away_aggression = away_aggression.reset_index()
away_aggression.columns = ['away_club_id', 'avg_cards_away']
df = df.merge(away_aggression, on='away_club_id', how='left')

# === Final Features and Target ===
features = [
    'attendance', 'home_club_position', 'away_club_position',
    'avg_cards_per_game_referee', 'avg_cards_home', 'avg_cards_away'
]

df = df.dropna(subset=features + ['target'])

X = df[features]
y = df['target']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === Evaluate ===
y_pred = clf.predict(X_test)
print("🧪 Classification Report:\n")
print(classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(clf, 'events_outcome_model.pkl')
print("✅ Model saved to models/events_outcome_model.pkl")

# Get feature importances
importances = clf.feature_importances_
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
plt.savefig("events_outcome_feature_importance.png")
plt.close()

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=clf.classes_, yticklabels=clf.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("events_outcome_confusion_matrix.png")
plt.close()
