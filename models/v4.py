import sqlite3
import pandas as pd
import numpy as np
import warnings
import gc
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

warnings.filterwarnings("ignore")

def step_progress(msg):
    print(f"{msg}...")
    import time
    time.sleep(0.2)

def load_data_in_chunks():
    """Load data more efficiently by selecting only needed columns"""
    conn = sqlite3.connect("/var/www/footbrain/football_data.db")
    
    step_progress("Step 1: Loading games (only needed columns)")
    games = pd.read_sql_query("""
        SELECT game_id, date, season, 
               home_club_id, away_club_id,
               home_club_goals, away_club_goals,
               home_club_position, away_club_position,
               home_club_formation, away_club_formation,
               home_club_name, away_club_name
        FROM games
        WHERE home_club_goals IS NOT NULL AND away_club_goals IS NOT NULL
    """, conn)
    
    step_progress("Step 2: Loading clubs")
    clubs = pd.read_sql_query("""
        SELECT club_id, name, squad_size, average_age, 
               national_team_players, stadium_name, stadium_seats,
               domestic_competition_id
        FROM clubs
    """, conn)
    
    step_progress("Step 3: Loading valuations")
    valuations = pd.read_sql_query("""
        SELECT player_id, current_club_id, date, market_value_in_eur
        FROM player_valuations
    """, conn)
    
    step_progress("Step 4: Loading game_lineups")
    game_lineups = pd.read_sql_query("""
        SELECT game_id, club_id, player_id, position, type
        FROM game_lineups
        WHERE position = 'Goalkeeper' AND type = 'starting_lineup'
    """, conn)
    
    conn.close()
    
    print(f"Loaded {len(games)} games, {len(clubs)} clubs, {len(valuations)} valuations, {len(game_lineups)} lineups")
    return games, clubs, valuations, game_lineups

def calculate_clean_sheets_efficient(games, game_lineups):
    """Calculate clean sheets without merging huge DataFrames"""
    step_progress("Step 5: Calculating clean sheets")
    
    clean_sheets_count = {}
    
    for idx, row in games.iterrows():
        if idx % 10000 == 0:
            print(f"Processing clean sheets: {idx}/{len(games)}")
        
        if row['home_club_goals'] == 0:
            club_id = row['home_club_id']
            clean_sheets_count[club_id] = clean_sheets_count.get(club_id, 0) + 1
        
        if row['away_club_goals'] == 0:
            club_id = row['away_club_id']
            clean_sheets_count[club_id] = clean_sheets_count.get(club_id, 0) + 1
    
    clean_df = pd.DataFrame(list(clean_sheets_count.items()), columns=['club_id', 'clean_sheets_last5'])
    
    games = games.merge(
        clean_df.rename(columns={'club_id': 'home_club_id', 'clean_sheets_last5': 'home_gk_clean_sheets_last5'}),
        on='home_club_id',
        how='left'
    )
    games = games.merge(
        clean_df.rename(columns={'club_id': 'away_club_id', 'clean_sheets_last5': 'away_gk_clean_sheets_last5'}),
        on='away_club_id',
        how='left'
    )
    
    games['home_gk_clean_sheets_last5'] = games['home_gk_clean_sheets_last5'].fillna(0)
    games['away_gk_clean_sheets_last5'] = games['away_gk_clean_sheets_last5'].fillna(0)
    
    return games

def calculate_form_efficient(games):
    """Calculate form using vectorized operations"""
    step_progress("Step 6: Calculating form")
    
    games['home_points'] = games.apply(
        lambda x: 3 if x['home_club_goals'] > x['away_club_goals'] 
        else 1 if x['home_club_goals'] == x['away_club_goals'] 
        else 0, axis=1
    )
    games['away_points'] = games.apply(
        lambda x: 3 if x['away_club_goals'] > x['home_club_goals']
        else 1 if x['home_club_goals'] == x['away_club_goals']
        else 0, axis=1
    )
    
    home_form = games[['game_id', 'date', 'home_club_id', 'home_points']].rename(
        columns={'home_club_id': 'club_id', 'home_points': 'points'}
    )
    away_form = games[['game_id', 'date', 'away_club_id', 'away_points']].rename(
        columns={'away_club_id': 'club_id', 'away_points': 'points'}
    )
    
    all_form = pd.concat([home_form, away_form], ignore_index=True)
    all_form = all_form.sort_values(['club_id', 'date'])
    
    all_form['form_last5'] = all_form.groupby('club_id')['points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    
    home_merge = all_form[['game_id', 'club_id', 'form_last5']].rename(
        columns={'club_id': 'home_club_id', 'form_last5': 'home_form_last5'}
    )
    away_merge = all_form[['game_id', 'club_id', 'form_last5']].rename(
        columns={'club_id': 'away_club_id', 'form_last5': 'away_form_last5'}
    )
    
    games = games.merge(home_merge, on=['game_id', 'home_club_id'], how='left')
    games = games.merge(away_merge, on=['game_id', 'away_club_id'], how='left')
    
    games['home_form_last5'] = games['home_form_last5'].fillna(1.0)
    games['away_form_last5'] = games['away_form_last5'].fillna(1.0)
    
    return games

def calculate_team_value_efficient(games, valuations):
    """Calculate team values efficiently"""
    step_progress("Step 7: Calculating team values")
    
    valuations = valuations.sort_values(['player_id', 'date'])
    latest_vals = valuations.drop_duplicates(subset=['player_id'], keep='last')
    
    club_values = latest_vals.groupby('current_club_id')['market_value_in_eur'].sum().reset_index()
    club_values.columns = ['club_id', 'total_value']
    
    games = games.merge(
        club_values.rename(columns={'club_id': 'home_club_id', 'total_value': 'total_value_home'}),
        on='home_club_id',
        how='left'
    )
    games = games.merge(
        club_values.rename(columns={'club_id': 'away_club_id', 'total_value': 'total_value_away'}),
        on='away_club_id',
        how='left'
    )
    
    games['total_value_home'] = games['total_value_home'].fillna(0)
    games['total_value_away'] = games['total_value_away'].fillna(0)
    
    return games

def main():
    print("Starting memory-efficient training pipeline...")
    
    games, clubs, valuations, game_lineups = load_data_in_chunks()
    
    games['date'] = pd.to_datetime(games['date'], errors='coerce')
    
    games = calculate_clean_sheets_efficient(games, game_lineups)
    games = calculate_form_efficient(games)
    games = calculate_team_value_efficient(games, valuations)
    
    del clubs, valuations, game_lineups
    gc.collect()
    
    step_progress("Step 8: Feature engineering")
    
    def get_result(row):
        if row['home_club_goals'] > row['away_club_goals']:
            return 0
        elif row['home_club_goals'] == row['away_club_goals']:
            return 1
        else:
            return 2
    
    games['result'] = games.apply(get_result, axis=1)
    
    games['home_club_position'] = pd.to_numeric(games['home_club_position'], errors='coerce')
    games['away_club_position'] = pd.to_numeric(games['away_club_position'], errors='coerce')
    
    games['home_club_position'] = games['home_club_position'].fillna(10)
    games['away_club_position'] = games['away_club_position'].fillna(10)
    
    games['position_diff'] = games['home_club_position'] - games['away_club_position']
    games['form_diff'] = games['home_form_last5'] - games['away_form_last5']
    games['clean_sheets_diff'] = games['home_gk_clean_sheets_last5'] - games['away_gk_clean_sheets_last5']
    games['relative_strength'] = (games['home_form_last5'] + 0.1) / (games['away_form_last5'] + 0.1)
    games['form_x_position_home'] = games['home_form_last5'] * (1.0 / games['home_club_position'].clip(lower=1))
    games['form_x_position_away'] = games['away_form_last5'] * (1.0 / games['away_club_position'].clip(lower=1))
    
    games['home_win'] = (games['home_club_goals'] > games['away_club_goals']).astype(int)
    games['away_win'] = (games['away_club_goals'] > games['home_club_goals']).astype(int)
    
    games['home_win_rate'] = games.groupby('home_club_id')['home_win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    games['away_win_rate'] = games.groupby('away_club_id')['away_win'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    games['win_rate_diff'] = games['home_win_rate'] - games['away_win_rate']
    
    games['home_conceded'] = games.groupby('home_club_id')['away_club_goals'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    games['away_conceded'] = games.groupby('away_club_id')['home_club_goals'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    
    games['home_scored'] = games.groupby('home_club_id')['home_club_goals'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    games['away_scored'] = games.groupby('away_club_id')['away_club_goals'].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    games['power_rank_diff'] = games['home_scored'] - games['away_scored']
    
    def form_trend(series):
        return series.rolling(3).mean() - series.shift(3).rolling(3).mean()
    
    games['home_trend'] = games.groupby('home_club_id')['home_form_last5'].transform(form_trend)
    games['away_trend'] = games.groupby('away_club_id')['away_form_last5'].transform(form_trend)
    games['trend_diff'] = games['home_trend'] - games['away_trend']
    
    numeric_cols = games.select_dtypes(include=[np.number]).columns
    games[numeric_cols] = games[numeric_cols].fillna(0)
    
    features = [
        'position_diff', 'form_diff', 'clean_sheets_diff',
        'home_club_position', 'away_club_position',
        'form_x_position_home', 'form_x_position_away',
        'home_gk_clean_sheets_last5', 'away_gk_clean_sheets_last5',
        'relative_strength', 'power_rank_diff', 'win_rate_diff',
        'away_conceded', 'home_conceded', 'trend_diff'
    ]
    
    initial_count = len(games)
    games = games.dropna(subset=features + ['result'])
    print(f"Removed {initial_count - len(games)} rows with missing values")
    print(f"Final dataset: {len(games)} matches")
    
    X = games[features].copy()
    y = games['result'].copy()
    
    print(f"\nClass distribution:")
    print(y.value_counts().to_string())
    
    step_progress("Step 9: Splitting data")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    step_progress("Step 10: Applying ADASYN")
    from collections import Counter
    print(f"Before ADASYN: {Counter(y_train)}")
    
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=3)
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)
        print(f"After ADASYN: {Counter(y_resampled)}")
    except Exception as e:
        print(f"ADASYN failed: {e}, using original data")
        X_resampled, y_resampled = X_train, y_train
    
    step_progress("Step 11: Feature selection")
    rfe_estimator = XGBClassifier(
        n_estimators=100,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0
    )
    
    n_features = min(10, len(features))
    selector = RFE(rfe_estimator, n_features_to_select=n_features, step=1)
    X_resampled_selected = selector.fit_transform(X_resampled, y_resampled)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.support_]
    print(f"Selected {len(selected_features)} features: {selected_features.tolist()}")
    
    X_resampled = pd.DataFrame(X_resampled_selected, columns=selected_features)
    X_test = pd.DataFrame(X_test_selected, columns=selected_features)
    
    step_progress("Step 12: Training model")
    
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective='multi:softprob',
        eval_metric='mlogloss',
        random_state=42,
        use_label_encoder=False,
        verbosity=0
    )
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        final_estimator=LogisticRegression(max_iter=500, random_state=42),
        cv=3,
        n_jobs=-1
    )
    
    stacking_clf.fit(X_resampled, y_resampled)
    
    y_pred = stacking_clf.predict(X_test)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Overall Accuracy: {accuracy*100:.2f}%")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(stacking_clf, "models/match_outcome_model_v4.pkl")
    
    feature_info = {
        'feature_names': list(selected_features),
        'model_type': 'stacking_classifier',
        'training_date': datetime.now().isoformat(),
        'training_samples': len(X_resampled),
        'accuracy': accuracy
    }
    joblib.dump(feature_info, "models/match_outcome_model_v4_features.pkl")
    
    print("\n✅ Model saved to models/match_outcome_model_v4.pkl")
    
    step_progress("Step 13: Generating visualizations")
    
    os.makedirs("static", exist_ok=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Home Win", "Draw", "Away Win"])
    disp.plot()
    plt.title(f"Confusion Matrix - Accuracy: {accuracy*100:.1f}%")
    plt.savefig("static/v4_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Confusion matrix saved")
    
    # Feature Importance
    xgb_estimator = stacking_clf.named_estimators_['xgb']
    importances = xgb_estimator.feature_importances_
    
    plt.figure(figsize=(12, 6))
    indices = np.argsort(importances)[::-1]
    bars = plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [selected_features[i] for i in indices], rotation=45, ha='right')
    plt.title("Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig("static/v4_feature_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Feature importance plot saved")
    
    # Print top features
    print("\n📊 Top 5 Most Important Features:")
    for i in range(min(5, len(indices))):
        feature_idx = indices[i]
        print(f"   {i+1}. {selected_features[feature_idx]}: {importances[feature_idx]:.4f}")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"📊 Model Accuracy: {accuracy*100:.2f}%")
    print(f"📁 Model saved to: models/match_outcome_model_v4.pkl")
    print(f"📁 Features saved to: models/match_outcome_model_v4_features.pkl")
    print(f"📈 Reports saved to: static/*.png")
    print("="*60)

if __name__ == "__main__":
    main()