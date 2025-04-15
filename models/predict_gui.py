import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import sqlite3
import numpy as np
import json
from datetime import datetime
import os
import sys

# Get today's date and calculate the range
current_date = pd.Timestamp.today().normalize()
start_date = current_date - pd.Timedelta(days=28)
end_date = current_date

def get_latest_formation(club_id):
    """Get the most recent formation used by a club in the last 14 days"""
    current_date = pd.Timestamp.today().normalize()
    
    # Check last 14 days first
    recent_games = games_df[
        (games_df['date'] >= (current_date - pd.Timedelta(days=7))) &  # Fixed parenthesis
        (games_df['date'] <= current_date) &
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)
    
    for _, game in recent_games.iterrows():
        if game['home_club_id'] == club_id and pd.notna(game['home_club_formation']):
            return game['home_club_formation']
        elif game['away_club_id'] == club_id and pd.notna(game['away_club_formation']):
            return game['away_club_formation']
    
    # Fallback to last known formation if none in last 14 days
    all_games = games_df[
        (games_df['date'] <= current_date) &
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)
    
    for _, game in all_games.iterrows():
        if game['home_club_id'] == club_id and pd.notna(game['home_club_formation']):
            return game['home_club_formation']
        elif game['away_club_id'] == club_id and pd.notna(game['away_club_formation']):
            return game['away_club_formation']
    
    return '4-4-2'  # Final fallback

def calculate_formation_strength(club_id, formation, is_home=True):
    """Calculate historical strength for a club's formation"""
    current_date = pd.Timestamp.today().normalize()
    
    if is_home:
        games = games_df[
            (games_df['home_club_id'] == club_id) &
            (games_df['home_club_formation'] == formation) &
            (games_df['date'] < current_date)
        ]
        if not games.empty:
            return (games['home_club_goals'] - games['away_club_goals']).mean()
    else:
        games = games_df[
            (games_df['away_club_id'] == club_id) &
            (games_df['away_club_formation'] == formation) &
            (games_df['date'] < current_date)
        ]
        if not games.empty:
            return (games['away_club_goals'] - games['home_club_goals']).mean()
    
    return 0.0  # Default strength if no history

# Function to get the latest position for a team
def get_latest_position(team_name):
    current_date = pd.Timestamp.today().normalize()
    
    # 1. Check games within ±7 days (excluding future games)
    team_games = games_df[
        (games_df['season'] == '2024') &
        (games_df['date'].between(current_date - pd.Timedelta(days=28), current_date)) &
        ((games_df['home_club_name'] == team_name) | (games_df['away_club_name'] == team_name))
    ].sort_values('date', ascending=False)
    
    # 2. Check all 2024 season games (excluding future games)
    if team_games.empty:
        team_games = games_df[
            (games_df['season'] == '2024') &
            (games_df['date'] <= current_date) &
            ((games_df['home_club_name'] == team_name) | (games_df['away_club_name'] == team_name))
        ].sort_values('date', ascending=False)

    # 3. Find first valid position
    for _, game in team_games.iterrows():
        position = game['home_club_position'] if game['home_club_name'] == team_name else game['away_club_position']
        if pd.notna(position):
            return pd.to_numeric(position, errors='coerce')

    # 4. Fallback to club's average position
    club_stats = get_club_stats(team_name)
    return club_stats['avg_position']

class AutocompleteCombobox(ttk.Combobox):
    def set_completion_list(self, completion_list):
        self._completion_list = sorted(completion_list, key=str.lower)
        self._hits = []
        self._hit_index = 0
        self.position = 0
        self.configure(values=self._completion_list)
        self.bind('<KeyRelease>', self.handle_keyrelease)

    def autocomplete(self, delta=0):
        if delta:
            self.delete(self.position, tk.END)
        else:
            self.position = len(self.get())

        _hits = [elem for elem in self._completion_list if elem.lower().startswith(self.get().lower())]

        if _hits != self._hits:
            self._hit_index = 0
            self._hits = _hits

        if _hits:
            self.delete(0, tk.END)
            self.insert(0, _hits[self._hit_index])
            self.select_range(self.position, tk.END)

    def handle_keyrelease(self, event):
        if event.keysym == "BackSpace":
            self.position = self.index(tk.END)
        elif event.keysym == "Left":
            if self.position < self.index(tk.END):
                self.delete(self.position, tk.END)
        elif event.keysym == "Right":
            self.position = self.index(tk.END)
        elif len(event.keysym) == 1:
            self.autocomplete()

# Load models and database connection
events_model = joblib.load('events_outcome_model.pkl')
#model = joblib.load("match_outcome_model_v2.pkl")

# Get list of available models (excluding events model)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
pkl_files = [f for f in os.listdir(script_dir) 
            if f.endswith('.pkl') and f != 'events_outcome_model.pkl']

if not pkl_files:
    messagebox.showerror("Error", "No model files (.pkl) found in directory!")
    sys.exit()

with open('feature_names.json') as f:
    feature_order = json.load(f)
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

def predict_events(home_team, away_team, referee):
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

    features = pd.DataFrame([features_dict])[feature_order]
    prob = events_model.predict_proba(features)[0][1]

    if prob < 0.6:
        estimate = "🔵 Estimated: 1 card"
    elif prob < 0.7:
        estimate = "🔵 Estimated: 2 cards"
    elif prob < 0.8:
        estimate = "🟢 Estimated: 3 cards"
    elif prob < 0.85:
        estimate = "🟠 Estimated: 4 cards"
    elif prob < 0.9:
        estimate = "🔴 Estimated: 5 cards"
    else:
        estimate = "🔴 Very likely 5+ cards"

    return (
        f"\n📊 Event Prediction Feature Summary:\n"
        + features.T.to_string()
        + f"\n\n📈 Prediction Probability: {prob * 100:.2f}%"
        + f"\n📌 Estimated Match Card Range: {estimate}"
    )

# Load data
clubs_df = pd.read_sql("SELECT * FROM clubs", conn)
games_df = pd.read_sql("SELECT * FROM games", conn)
player_valuations_df = pd.read_sql("SELECT * FROM player_valuations", conn)
competitions_df = pd.read_sql("SELECT DISTINCT country_id, country_name, competition_id, name as competition_name FROM competitions", conn)
games_df['date'] = pd.to_datetime(games_df['date'])
games_df['season'] = games_df['season'].astype(str)

# Process competitions data
competitions_df['country_name'] = np.where(competitions_df['country_id'] == -1, 'UEFA', competitions_df['country_name'])
countries = sorted(competitions_df['country_name'].unique().tolist())
country_to_id = competitions_df.set_index('country_name')['country_id'].to_dict()

# Create country-league mapping
country_league_map = {}
for country in countries:
    country_id = country_to_id[country]
    leagues = competitions_df[competitions_df['country_name'] == country][['competition_name', 'competition_id']]
    country_league_map[country] = {row['competition_name']: str(row['competition_id']) for _, row in leagues.iterrows()}

# Create competition-clubs mapping
competition_clubs_map = {}
for competition_id in clubs_df['domestic_competition_id'].unique():
    competition_id_str = str(competition_id)
    clubs = clubs_df[clubs_df['domestic_competition_id'] == competition_id]['name'].tolist()
    competition_clubs_map[competition_id_str] = clubs

# Extract dropdown values
teams = sorted(clubs_df['name'].dropna().unique().tolist())
referees = sorted(games_df['referee'].dropna().unique().tolist())
venues = sorted(clubs_df['stadium_name'].dropna().unique().tolist())

# Create main window
root = tk.Tk()
root.title("Match Outcome Predictor")
root.geometry("600x700")  # Increased height for new dropdowns
root.configure(bg="#f9f9f9")

# Model Selection
tk.Label(root, text="🤖 Select Model", bg="#f9f9f9").pack(pady=(10,5))
model_var = tk.StringVar()
model_combo = ttk.Combobox(root, textvariable=model_var, 
                          values=pkl_files, state='readonly', width=50)
model_combo.pack()

# Set default model
default_model = "match_outcome_model_v2.pkl" if "match_outcome_model_v2.pkl" in pkl_files else pkl_files[0]
model_var.set(default_model)

# Country and League Selectors
tk.Label(root, text="🌍 Country", bg="#f9f9f9").pack(pady=(20, 5))
country_var = tk.StringVar()
country_combo = ttk.Combobox(root, textvariable=country_var, values=countries, state='readonly', width=50)
country_combo.pack()

tk.Label(root, text="🏆 League", bg="#f9f9f9").pack(pady=(10, 5))
league_var = tk.StringVar()
league_combo = ttk.Combobox(root, textvariable=league_var, state='readonly', width=50)
league_combo.pack()

def on_country_selected(event):
    selected_country = country_var.get()
    if selected_country:
        leagues = list(country_league_map[selected_country].keys())
        league_combo['values'] = leagues
        league_var.set('')
        home_var.set('')
        away_var.set('')
        home_combo.set_completion_list(teams)
        away_combo.set_completion_list(teams)

def on_league_selected(event):
    selected_league = league_var.get()
    if selected_league and country_var.get():
        competition_id = country_league_map[country_var.get()][selected_league]
        clubs = competition_clubs_map.get(competition_id, [])
        home_combo.set_completion_list(clubs)
        away_combo.set_completion_list(clubs)

country_combo.bind("<<ComboboxSelected>>", on_country_selected)
league_combo.bind("<<ComboboxSelected>>", on_league_selected)

# Team Selectors
tk.Label(root, text="🏠 Home Team", bg="#f9f9f9").pack(pady=(10, 5))
home_var = tk.StringVar()
home_combo = AutocompleteCombobox(root, textvariable=home_var, width=50)
home_combo.set_completion_list(teams)
home_combo.pack()

tk.Label(root, text="🚗 Away Team", bg="#f9f9f9").pack(pady=(10, 5))
away_var = tk.StringVar()
away_combo = AutocompleteCombobox(root, textvariable=away_var, width=50)
away_combo.set_completion_list(teams)
away_combo.pack()

# Referee and Venue Selectors
tk.Label(root, text="🧑‍⚖️ Referee", bg="#f9f9f9").pack(pady=(10, 5))
ref_var = tk.StringVar()
ref_combo = AutocompleteCombobox(root, textvariable=ref_var, width=50)
ref_combo.set_completion_list(referees)
ref_combo.pack()

tk.Label(root, text="🏟️ Venue", bg="#f9f9f9").pack(pady=(10, 5))
venue_var = tk.StringVar()
venue_combo = AutocompleteCombobox(root, textvariable=venue_var, width=50)
venue_combo.set_completion_list(venues)
venue_combo.pack()

# Button Frame
button_frame = ttk.Frame(root)
button_frame.pack(pady=(10, 20))

def predict():
    try:
        # Load selected model
        selected_model = model_var.get()
        model = joblib.load(selected_model)  # Now loaded dynamically
        home_team = home_var.get()
        away_team = away_var.get()
        referee = ref_var.get()
        venue = venue_var.get()

        if home_team == away_team:
            messagebox.showerror("Error", "Home and away teams must be different.")
            return

        home = clubs_df[clubs_df["name"] == home_team]
        away = clubs_df[clubs_df["name"] == away_team]

        if home.empty or away.empty:
            messagebox.showerror("Error", "One or both teams not found.")
            return

        home_id = home.iloc[0]['club_id']
        away_id = away.iloc[0]['club_id']

        # Get formations and calculate strengths
        home_formation = get_latest_formation(home_id)
        away_formation = get_latest_formation(away_id)
        home_form_strength = calculate_formation_strength(home_id, home_formation, is_home=True)
        away_form_strength = calculate_formation_strength(away_id, away_formation, is_home=False)
        form_strength_diff = home_form_strength - away_form_strength

        game_sample = games_df[
            (games_df['home_club_id'] == home_id) &
            (games_df['away_club_id'] == away_id)
        ].sort_values('date', ascending=False)

        home_position = get_latest_position(home_team)
        away_position = get_latest_position(away_team)
        position_diff = home_position - away_position if not np.isnan(home_position) and not np.isnan(away_position) else 0
        attendance = home.iloc[0]['stadium_seats'] if venue != "Unknown" else 35000
        seats_diff = home.iloc[0]['stadium_seats'] - away.iloc[0]['stadium_seats']
        age_diff = home.iloc[0]['average_age'] - away.iloc[0]['average_age']
        nationals_diff = home.iloc[0]['national_team_players'] - away.iloc[0]['national_team_players']

        # 🧮 Calculate team values from latest valuations
        def get_team_value(club_id):
            valuations = player_valuations_df[player_valuations_df["current_club_id"] == club_id]
            latest_dates = valuations.groupby("player_id")["date"].idxmax()
            latest_valuations = valuations.loc[latest_dates]
            return latest_valuations["market_value_in_eur"].sum()

        home_team_value = get_team_value(home_id)
        away_team_value = get_team_value(away_id)

        value_diff = home_team_value - away_team_value
        
        X_input = pd.DataFrame([{
        'home_squad_size': home.iloc[0]['squad_size'],
        'home_average_age': home.iloc[0]['average_age'],
        'away_squad_size': away.iloc[0]['squad_size'],
        'away_average_age': away.iloc[0]['average_age'],
        'home_club_position': home_position,
        'away_club_position': away_position,
        'attendance': attendance,
        'position_diff': position_diff,
        'age_diff': age_diff,
        'nationals_diff': nationals_diff,
        'seats_diff': seats_diff,
        'total_value_home': home_team_value,
        'total_value_away': away_team_value,
        'value_diff': value_diff,
        'home_formation_strength': home_form_strength,
        'away_formation_strength': away_form_strength,
        'formation_strength_diff': form_strength_diff,
        }])


        probs = model.predict_proba(X_input)[0]  # Use dynamically loaded model
        prediction = model.predict(X_input)[0]
        outcome = {0: "🏠 Home Win", 1: "🤝 Draw", 2: "🚗 Away Win"}

        result_text.delete("1.0", tk.END)
        # Add formation display
        result_text.insert(tk.END, f"📊 Match Prediction Summary:\n\n")
        result_text.insert(tk.END, f"🏟️  {home_team} vs {away_team}\n\n")
        result_text.insert(tk.END, f"📌 Venue: {venue}\n")
        result_text.insert(tk.END, f"🧑‍⚖️ Referee: {referee}\n\n")
        result_text.insert(tk.END, f"\n⚽ Recent Formations:\n")
        result_text.insert(tk.END, f"🏠 {home_team}: {home_formation} (Strength: {home_form_strength:.2f})\n")
        result_text.insert(tk.END, f"🚗 {away_team}: {away_formation} (Strength: {away_form_strength:.2f})\n\n")
        result_text.insert(tk.END, f"📈 Predicted Outcome: {outcome[prediction]}\n")
        result_text.insert(tk.END, f"🟢 Home Win: {probs[0]*100:.1f}%\n")
        result_text.insert(tk.END, f"🟡 Draw: {probs[1]*100:.1f}%\n")
        result_text.insert(tk.END, f"🔴 Away Win: {probs[2]*100:.1f}%\n\n")
        result_text.insert(tk.END, f"📊 Features Used:\n")
        for k, v in X_input.iloc[0].items():
            result_text.insert(tk.END, f"  {k}: {v}\n")

        ref_avg = get_avg_cards_per_referee(referee)
        home_stats = get_club_stats(home_team)
        away_stats = get_club_stats(away_team)

        event_features = pd.DataFrame([{
            'attendance': attendance,
            'avg_cards_home': home_stats['avg_cards'],
            'avg_cards_away': away_stats['avg_cards'],
            'home_club_position': home_stats['avg_position'],
            'away_club_position': away_stats['avg_position'],
            'avg_cards_per_game_referee': ref_avg
        }])[feature_order]

        event_prob = events_model.predict_proba(event_features)[0][1]

        if event_prob < 0.4:
            card_estimate = "🔵 Estimated: 1 card"
        elif event_prob < 0.5:
            card_estimate = "🔵 Estimated: 1-2 cards"
        elif event_prob < 0.6:
            card_estimate = "🟢 Estimated: 2-3 cards"
        elif event_prob < 0.7:
            card_estimate = "🟠 Estimated: 3-4 cards"
        elif event_prob < 0.8:
            card_estimate = "🔴 Estimated: 4-5 cards"
        else:
            card_estimate = "🔴 Very likely 5+ cards"

        result_text.insert(tk.END, "\n📦 Event Outcome Prediction (Cards):\n")
        result_text.insert(tk.END, f"📈 Probability: {event_prob * 100:.2f}%\n")
        result_text.insert(tk.END, f"{card_estimate}\n")

    except Exception as e:
        messagebox.showerror("Error", str(e))

def save_to_file():
    try:
        content = result_text.get("1.0", tk.END)
        if not content.strip():
            messagebox.showwarning("Warning", "No prediction to save!")
            return
            
        with open("betslip.txt", "a") as f:
            f.write(f"\n\n{'='*40}\n")
            f.write(f"Prediction saved at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(content)
            f.write(f"{'='*40}\n")
        messagebox.showinfo("Success", "Prediction saved to betslip.txt!")
    except Exception as e:
        messagebox.showerror("Save Error", str(e))

def clear_fields():
    country_var.set('')
    league_var.set('')
    home_var.set('')
    away_var.set('')
    ref_var.set('')
    venue_var.set('')
    home_combo.set_completion_list(teams)
    away_combo.set_completion_list(teams)
    #result_text.delete("1.0", tk.END)
    home_combo.focus_set()

# Buttons
ttk.Button(button_frame, text="🔮 Predict", command=predict).grid(row=0, column=0, padx=5)
ttk.Button(button_frame, text="💾 Save to File", command=save_to_file).grid(row=0, column=1, padx=5)
ttk.Button(button_frame, text="🧹 Clear Fields", command=clear_fields).grid(row=0, column=2, padx=5)

# Result Display
result_text = tk.Text(root, height=15, width=70, bg="#fff", fg="#333")
result_text.pack(pady=(10, 20))

root.mainloop()
