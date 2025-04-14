import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import sqlite3
import numpy as np
import json
from datetime import datetime

# Get today's date and calculate the range
current_date = pd.Timestamp.today().normalize()
start_date = current_date - pd.Timedelta(days=7)
end_date = current_date + pd.Timedelta(days=7)

# Function to get the latest position for a team
def get_latest_position(team_name):
    current_date = pd.Timestamp.today().normalize()
    
    # 1. Check games within ±7 days (excluding future games)
    team_games = games_df[
        (games_df['season'] == '2024') &
        (games_df['date'].between(current_date - pd.Timedelta(days=7), current_date)) &
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
model = joblib.load("match_outcome_model.pkl")
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
games_df['date'] = pd.to_datetime(games_df['date'])
games_df['season'] = games_df['season'].astype(str)

# Extract dropdown values
teams = sorted(clubs_df['name'].dropna().unique().tolist())
referees = sorted(games_df['referee'].dropna().unique().tolist())
venues = sorted(clubs_df['stadium_name'].dropna().unique().tolist())

# Create main window
root = tk.Tk()
root.title("Match Outcome Predictor")
root.geometry("600x600")
root.configure(bg="#f9f9f9")

# UI Elements
tk.Label(root, text="🏠 Home Team", bg="#f9f9f9").pack(pady=(20, 5))
home_var = tk.StringVar()
home_combo = AutocompleteCombobox(root, textvariable=home_var, width=50)
home_combo.set_completion_list(teams)
home_combo.pack()

tk.Label(root, text="🚗 Away Team", bg="#f9f9f9").pack(pady=(10, 5))
away_var = tk.StringVar()
away_combo = AutocompleteCombobox(root, textvariable=away_var, width=50)
away_combo.set_completion_list(teams)
away_combo.pack()

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

        game_sample = games_df[
            (games_df['home_club_id'] == home.iloc[0]['club_id']) &
            (games_df['away_club_id'] == away.iloc[0]['club_id'])
        ].sort_values('date', ascending=False)

        #home_position = pd.to_numeric(game_sample.iloc[0]['home_club_position'], errors='coerce') if not game_sample.empty else np.nan
        #away_position = pd.to_numeric(game_sample.iloc[0]['away_club_position'], errors='coerce') if not game_sample.empty else np.nan
        home_position = get_latest_position(home_team)
        away_position = get_latest_position(away_team)
        position_diff = home_position - away_position if not np.isnan(home_position) and not np.isnan(away_position) else 0
        attendance = home.iloc[0]['stadium_seats'] if venue != "Unknown" else 35000
        seats_diff = home.iloc[0]['stadium_seats'] - away.iloc[0]['stadium_seats']
        #position_diff = (home_position - away_position) if not np.isnan(home_position) and not np.isnan(away_position) else 0
        age_diff = home.iloc[0]['average_age'] - away.iloc[0]['average_age']
        nationals_diff = home.iloc[0]['national_team_players'] - away.iloc[0]['national_team_players']

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
            'seats_diff': seats_diff
        }])

        probs = model.predict_proba(X_input)[0]
        prediction = model.predict(X_input)[0]
        outcome = {0: "🏠 Home Win", 1: "🤝 Draw", 2: "🚗 Away Win"}

        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"📊 Match Prediction Summary:\n")
        result_text.insert(tk.END, f"🏟️  {home_team} vs {away_team}\n")
        result_text.insert(tk.END, f"📌 Venue: {venue}\n")
        result_text.insert(tk.END, f"🧑‍⚖️ Referee: {referee}\n\n")
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
    home_var.set('')
    away_var.set('')
    ref_var.set('')
    venue_var.set('')
    result_text.delete("1.0", tk.END)
    home_combo.focus_set()

# Buttons
ttk.Button(button_frame, text="🔮 Predict", command=predict).grid(row=0, column=0, padx=5)
ttk.Button(button_frame, text="💾 Save to File", command=save_to_file).grid(row=0, column=1, padx=5)
ttk.Button(button_frame, text="🧹 Clear Fields", command=clear_fields).grid(row=0, column=2, padx=5)

# Result Display
result_text = tk.Text(root, height=15, width=70, bg="#fff", fg="#333")
result_text.pack(pady=(10, 20))

root.mainloop()
