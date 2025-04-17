# models/v2.py

import sqlite3
import pandas as pd
import numpy as np
import warnings
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import time
import os
from tqdm import tqdm

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
xgb.set_config(verbosity=0)
tqdm.pandas()

def step_progress(desc):
    tqdm.write(f"{desc}...")
    time.sleep(0.2)

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

# Enhanced form calculation with exponential decay
step_progress("Step 3a: Calculating weighted form with recent bias")
def calculate_weighted_form(series):
    weights = np.array([0.5, 0.6, 0.7, 0.8, 1.0])  # Weights for last 5 matches
    def weighted_avg(window):
        available = min(len(window), len(weights))
        return np.dot(window[-available:], weights[-available:]) / weights[-available:].sum()
    return series.rolling(window=5, min_periods=1).apply(weighted_avg, raw=True)

club_form['form_last_5'] = club_form.groupby('club_id')['result'].transform(calculate_weighted_form)


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

# Create sorted_games BEFORE filtering
sorted_games = games.sort_values('date').copy()

# Filter for 2024 season
step_progress("Step 5a: Recalculating formation strength for 2024 season")
current_season_start = pd.to_datetime('2024-08-01')
sorted_games = sorted_games[sorted_games['date'] >= current_season_start]

# Now calculate formation strengths
sorted_games['home_goal_diff'] = sorted_games['home_club_goals'] - sorted_games['away_club_goals']
sorted_games['away_goal_diff'] = sorted_games['away_club_goals'] - sorted_games['home_club_goals']

# Club metadata
step_progress("Step 6: Final data preprocessing and feature engineering")

# Merge club data
games = games.merge(clubs.add_prefix("home_"), left_on="home_club_id", right_on="home_club_id", how="left")
games = games.merge(clubs.add_prefix("away_"), left_on="away_club_id", right_on="away_club_id", how="left")

# Remove matches without scores
games.dropna(subset=['home_club_goals', 'away_club_goals'], inplace=True)

# Match result target
def get_match_result(row):
    if row['home_club_goals'] > row['away_club_goals']:
        return 0  # Home Win
    elif row['home_club_goals'] == row['away_club_goals']:
        return 1  # Draw
    else:
        return 2  # Away Win

games["result"] = games.apply(get_match_result, axis=1)

# Calculate all difference features
games["position_diff"] = games["home_club_position"] - games["away_club_position"]
games["form_diff"] = games["home_form_last5"] - games["away_form_last5"]
games["clean_sheets_diff"] = games["home_gk_clean_sheets_last5"] - games["away_gk_clean_sheets_last5"]
games["nationals_diff"] = games["home_national_team_players"] - games["away_national_team_players"]
games["seats_diff"] = games["home_stadium_seats"] - games["away_stadium_seats"]
games["value_diff"] = games["total_value_home"] - games["total_value_away"]

# Create interaction terms
games["form_x_position_home"] = games['home_form_last5'] * (1/games['home_club_position'].clip(lower=1))  # Avoid division by zero
games["form_x_position_away"] = games['away_form_last5'] * (1/games['away_club_position'].clip(lower=1))

# Ensure numeric positions
for col in ['home_club_position', 'away_club_position']:
    games[col] = pd.to_numeric(games[col], errors='coerce')

# Final feature list
features = [
    'position_diff', 'form_diff', 'clean_sheets_diff',
    'home_club_position', 'away_club_position',
    'formation_strength_diff', 'attendance',
    'nationals_diff', 'seats_diff', 'value_diff',
    'form_x_position_home', 'form_x_position_away',
    'home_gk_clean_sheets_last5', 'away_gk_clean_sheets_last5',
    'total_value_home', 'total_value_away'
]

# Validate all features exist before dropping NA
missing_features = [f for f in features + ["result"] if f not in games.columns]
if missing_features:
    print("Available columns:", sorted(games.columns.tolist()))
    raise ValueError(f"Missing required features: {missing_features}")

# Remove rows with missing values
initial_count = len(games)
games.dropna(subset=features + ["result"], inplace=True)
removed_count = initial_count - len(games)
print(f"Removed {removed_count} rows ({removed_count/initial_count:.1%}) with missing values")

# Final validation
assert not games[features + ["result"]].isnull().any().any(), "NA values still present in final dataset"
assert len(games) > 0, "No valid matches remaining after preprocessing"

# After all feature engineering and NA dropping:
step_progress("Step 6a: Preparing final dataset")

# Define features and target
X = games[features]
y = games["result"]

# Verify shapes
print(f"Final dataset shape: {X.shape}, target shape: {y.shape}")

# Split data
step_progress("Step 7: Splitting data")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Now apply ADASYN only to training data
step_progress("Step 8: Applying ADASYN for class balancing")
adasyn = ADASYN(
    sampling_strategy={1: int(len(y_train[y_train==1])*1.5)},  # Boost Draws by 50%
    random_state=42, 
    n_neighbors=5
)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

print(f"Resampled dataset shapes: {X_resampled.shape}, {y_resampled.shape}")

# Calculate class weights
class_weights = compute_sample_weight(class_weight='balanced', y=y_resampled)

# Enhanced XGBoost configuration
xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    use_label_encoder=False,
    random_state=42,
    tree_method='hist',
    scale_pos_weight=len(y_resampled)/(3*np.bincount(y_resampled)),  # Force class balance
    reg_lambda=1.5,  # Increased regularization
    reg_alpha=0.5,
    max_depth=5,     # Reduced tree depth
    subsample=0.8,
    colsample_bytree=0.8
)

param_dist = {
    'n_estimators': [150, 200],
    'learning_rate': [0.05, 0.075],
    'gamma': [0.5, 1],
    'max_depth': [4, 5],
}

# Optimized Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    max_features='sqrt',
    class_weight='balanced_subsample',
    min_samples_leaf=10,
    random_state=42
)



# Enhanced RandomizedSearch
xgb_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions={'xgb__'+k: v for k,v in param_dist.items()},
    n_iter=8,
    scoring='f1_macro',
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    verbose=1,
    n_jobs=1,
    random_state=42
)

# Fit with resampled data
xgb_search.fit(X_resampled, y_resampled, sample_weight=class_weights)

# Combine in Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_search.best_estimator_),
        ('rf', rf_model)
    ],
    voting='soft'
)

# Fit using the same training data
voting_clf.fit(X_train, y_train)

# Predict & Evaluate
y_pred = voting_clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

# Define stratified cross-validation
stratified_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Evaluate using resampled training data
scores = cross_val_score(
    voting_clf, 
    X_resampled, 
    y_resampled, 
    cv=stratified_cv, 
    scoring='accuracy'
)

print(f"Stratified CV Accuracy Scores: {scores}")
print(f"Mean Accuracy: {scores.mean():.4f}")

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

# Feature importance analysis with permutation importance
from sklearn.inspection import permutation_importance

result = permutation_importance(xgb_search.best_estimator_, X_test, y_test, n_repeats=5, random_state=42)
sorted_idx = result.importances_mean.argsort()[::-1]

print("\nPermutation Importance:")
for i in sorted_idx:
    print(f"{feature_names[i]}: {result.importances_mean[i]:.4f} (±{result.importances_std[i]:.4f})")

plt.figure(figsize=(10,6))
plt.title("v2 Feature Importances")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("v2_feature_importance.png")
plt.close()


