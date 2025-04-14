import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib
import sqlite3
import numpy as np

import tkinter as tk
from tkinter import ttk

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


# Load model and DB
model = joblib.load("match_outcome_model.pkl")
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

# Load data
clubs_df = pd.read_sql("SELECT * FROM clubs", conn)
games_df = pd.read_sql("SELECT * FROM games", conn)

# Extract dropdown values
teams = sorted(clubs_df['name'].dropna().unique().tolist())
referees = sorted(games_df['referee'].dropna().unique().tolist())
venues = sorted(clubs_df['stadium_name'].dropna().unique().tolist())

# Create main window
root = tk.Tk()
root.title("Match Outcome Predictor")
root.geometry("600x500")
root.configure(bg="#f9f9f9")

# --- UI ELEMENTS ---

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

result_text = tk.Text(root, height=15, width=70, bg="#fff", fg="#333")
result_text.pack(pady=(20, 10))


def predict():
    try:
        home_team = home_var.get()
        away_team = away_var.get()
        referee = ref_var.get()
        venue = venue_var.get()

        if home_team == away_team:
            messagebox.showerror("Error", "Home and away teams must be different.")
            return

        # Fetch team data
        home = clubs_df[clubs_df["name"] == home_team]
        away = clubs_df[clubs_df["name"] == away_team]

        if home.empty or away.empty:
            messagebox.showerror("Error", "One or both teams not found.")
            return

        # Get positions if available
        game_sample = games_df[
            (games_df['home_club_id'] == home.iloc[0]['club_id']) &
            (games_df['away_club_id'] == away.iloc[0]['club_id'])
        ].sort_values('date', ascending=False)

        home_position = pd.to_numeric(game_sample.iloc[0]['home_club_position'], errors='coerce') if not game_sample.empty else np.nan
        away_position = pd.to_numeric(game_sample.iloc[0]['away_club_position'], errors='coerce') if not game_sample.empty else np.nan

        # Feature calculation
        attendance = home.iloc[0]['stadium_seats'] if venue != "Unknown" else 35000
        seats_diff = home.iloc[0]['stadium_seats'] - away.iloc[0]['stadium_seats']
        position_diff = (home_position - away_position) if not np.isnan(home_position) and not np.isnan(away_position) else 0
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

        # Display results
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

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Predict Button
ttk.Button(root, text="🔮 Predict", command=predict).pack(pady=(0, 20))

# Start GUI
root.mainloop()
