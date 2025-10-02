import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import PIL.ImageTk  # Explicit ImageTk import
import pandas as pd
import joblib
import sqlite3
import numpy as np
import json
from datetime import datetime
import os
import sys
import traceback
from sklearn.metrics import f1_score  # Added for custom_f1_score
# --- Splash Screen Setup ---
splash = tk.Tk()
splash.overrideredirect(True)  # Remove window decorations

# Load splash image
img = Image.open(os.path.join(os.path.dirname(__file__), "splash.png"))

# Calculate half size while maintaining aspect ratio
new_width = img.width // 2
new_height = img.height // 2
img = img.resize((new_width, new_height), Image.LANCZOS)  # High-quality resizing

# Create PhotoImage after resizing
#photo = ImageTk.PhotoImage(img)

# Center the splash screen
screen_width = splash.winfo_screenwidth()
screen_height = splash.winfo_screenheight()
max_width = screen_width // 2
max_height = screen_height // 2
if img.width > max_width or img.height > max_height:
    ratio = min(max_width/img.width, max_height/img.height)
    new_width = int(img.width * ratio)
    new_height = int(img.height * ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)

# Create PhotoImage after resizing
photo = PIL.ImageTk.PhotoImage(img)

# Create canvas instead of label for better control
canvas = tk.Canvas(splash, width=new_width, height=new_height, highlightthickness=0)
canvas.pack()

# Display image on canvas
canvas.create_image(0, 0, image=photo, anchor="nw")

# Add text directly on canvas
text_info = """VARIO Engine v0.54
Created by Dragomir Dimitrov
April 2025
"""
canvas.create_text(
    10, 10,  # x,y position
    text=text_info,
    font=("Arial", 10),
    fill="white",
    anchor="nw",  # northwest alignment
    width=new_width-40  # text wrapping boundary
)
text_loading = """Loading current season database...
Please wait...
"""
canvas.create_text(
    10, 370,  # x,y position
    text=text_loading,
    font=("Arial", 12),
    fill="white",
    anchor="sw",  # northwest alignment
    width=new_width-40  # text wrapping boundary
)

# Add semi-transparent background for text
#canvas.create_rectangle(
#    10, 10,  # x1,y1
#    new_width-10, 100,  # x2,y2 (adjust height as needed)
#    fill="black",
#    stipple="gray25"  # creates 25% transparency effect
#)

# Center the splash screen
screen_width = splash.winfo_screenwidth()
screen_height = splash.winfo_screenheight()
x = (screen_width - new_width) // 2
y = (screen_height - new_height) // 2
splash.geometry(f"{new_width}x{new_height}+{x}+{y}")

# Force immediate update
splash.update_idletasks()
splash.update()

# Get today's date
current_date = pd.Timestamp.today().normalize()

def custom_f1_score(preds, dtrain):
    labels = dtrain.get_label()
    preds = preds.reshape(3, -1).argmax(axis=0)
    return 'f1_macro', f1_score(labels, preds, average='macro')

def log_error(error):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("log.txt", "a") as f:
        f.write(f"\n\n[{timestamp}] ERROR:\n")
        f.write(f"Message: {str(error)}\n")
        f.write(f"Traceback:\n{traceback.format_exc()}")
        f.write("-" * 40)

# Load models
events_model = joblib.load('events_outcome_model.pkl')
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
pkl_files = [f for f in os.listdir(script_dir) if f.endswith('.pkl') and f != 'events_outcome_model.pkl']
if not pkl_files:
    messagebox.showerror("Error", "No model files (.pkl) found in directory!")
    sys.exit()
with open('feature_names.json') as f:
    feature_order = json.load(f)

# Database
conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")

def get_latest_goalkeeper(club_id):
    gk_lineups = game_lineups[
        (game_lineups['club_id'] == club_id) &
        (game_lineups['position'] == 'Goalkeeper') &
        (game_lineups['type'] == 'starting_lineup')
    ].sort_values('game_id', ascending=False)
    if not gk_lineups.empty:
        return gk_lineups.iloc[0]['player_id']
    return None

def get_latest_formation(club_id):
    recent_games = games_df[
        (games_df['date'] >= (current_date - pd.Timedelta(days=7))) &
        (games_df['date'] <= current_date) &
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)

    for _, game in recent_games.iterrows():
        if game['home_club_id'] == club_id and pd.notna(game['home_club_formation']):
            return game['home_club_formation']
        elif game['away_club_id'] == club_id and pd.notna(game['away_club_formation']):
            return game['away_club_formation']

    all_games = games_df[
        (games_df['date'] <= current_date) &
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)

    for _, game in all_games.iterrows():
        if game['home_club_id'] == club_id and pd.notna(game['home_club_formation']):
            return game['home_club_formation']
        elif game['away_club_id'] == club_id and pd.notna(game['away_club_formation']):
            return game['away_club_formation']

    return '4-4-2'

def calculate_formation_strength(club_id, formation, is_home=True):
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
    return 0.0

def get_latest_position(team_name):
    team_games = games_df[
        (games_df['season'] == '2024') &
        (games_df['date'].between(current_date - pd.Timedelta(days=28), current_date)) &
        ((games_df['home_club_name'] == team_name) | (games_df['away_club_name'] == team_name))
    ].sort_values('date', ascending=False)

    if team_games.empty:
        team_games = games_df[
            (games_df['season'] == '2024') &
            (games_df['date'] <= current_date) &
            ((games_df['home_club_name'] == team_name) | (games_df['away_club_name'] == team_name))
        ].sort_values('date', ascending=False)

    for _, game in team_games.iterrows():
        position = game['home_club_position'] if game['home_club_name'] == team_name else game['away_club_position']
        if pd.notna(position):
            return pd.to_numeric(position, errors='coerce')

    club_stats = get_club_stats(team_name)
    return club_stats['avg_position']

def calculate_form_last5(club_id, is_home):
    """
    Calculates average form (points per game) for the last 5 matches played by the given club.
    Looks at both home and away games and only includes matches before today.
    """
    current_date = pd.Timestamp.today().normalize()

    # Filter games played by the club before today
    relevant_games = games_df[
        (games_df['date'] < current_date) & 
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)

    recent_games = relevant_games.head(5)

    points = 0
    total_games = 0

    for _, game in recent_games.iterrows():
        if game['home_club_id'] == club_id:
            goals_for = game['home_club_goals']
            goals_against = game['away_club_goals']
        else:
            goals_for = game['away_club_goals']
            goals_against = game['home_club_goals']

        if pd.notna(goals_for) and pd.notna(goals_against):
            if goals_for > goals_against:
                points += 3
            elif goals_for == goals_against:
                points += 1
            # else 0 points
            total_games += 1

    return points / total_games if total_games > 0 else 0.0

def get_clean_sheets_last5(gk_player_id, club_id):
    """Calculate number of clean sheets in last 5 matches for a goalkeeper"""
    # Get all games where the GK played for this club
    gk_games = game_lineups[
        (game_lineups['player_id'] == gk_player_id) &
        (game_lineups['club_id'] == club_id)
    ]['game_id'].unique()
    
    # Get match data for these games
    club_games = games_df[
        (games_df['game_id'].isin(gk_games)) &
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id)) &
        (games_df['date'] < current_date)
    ].sort_values('date', ascending=False)
    
    # Take last 5 games
    last5 = club_games.head(5)
    
    # Count clean sheets
    clean_sheets = 0
    for _, game in last5.iterrows():
        if game['home_club_id'] == club_id:
            conceded = game['away_club_goals']
        else:
            conceded = game['home_club_goals']
        if pd.notna(conceded) and conceded == 0:
            clean_sheets += 1
            
    return clean_sheets

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

# Combobox with autocomplete
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


#model = joblib.load("match_outcome_model_v2.pkl")

# Get list of available models (excluding events model)
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
pkl_files = [f for f in os.listdir(script_dir) 
            if f.endswith('.pkl') and f != 'events_outcome_model.pkl']

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

# Load data
clubs_df = pd.read_sql("SELECT * FROM clubs", conn)
games_df = pd.read_sql("SELECT * FROM games", conn)
player_valuations_df = pd.read_sql("SELECT * FROM player_valuations", conn)
competitions_df = pd.read_sql("SELECT DISTINCT country_id, country_name, competition_id, name as competition_name FROM competitions", conn)
games_df['date'] = pd.to_datetime(games_df['date'])
games_df['season'] = games_df['season'].astype(str)
game_lineups = pd.read_sql("SELECT * FROM game_lineups", conn)  # Added

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

splash.destroy()  # <-- Kill splash !
# Create main window
root = tk.Tk()
root.title("VARIO Engine v0.54")
root.geometry("600x700")  # Increased height for new dropdowns
root.configure(bg="#f9f9f9")

# Model Selection
tk.Label(root, text="🤖 Select Model", bg="#f9f9f9").pack(pady=(10,5))
model_var = tk.StringVar()
model_combo = ttk.Combobox(root, textvariable=model_var, 
                          values=pkl_files, state='readonly', width=50)
model_combo.pack()

# Set default model
default_model = "footpredict/models/match_outcome_model_v2.pkl" if "footpredict/models/match_outcome_model_v2.pkl" in pkl_files else pkl_files[0]
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

def calculate_rolling_average(club_id, goal_column, is_home=True):
    """Calculate rolling average goals for a club"""
    if is_home:
        club_games = games_df[
            (games_df['home_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
        if not club_games.empty:
            return club_games[goal_column].head(5).mean()
    else:
        club_games = games_df[
            (games_df['away_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
        if not club_games.empty:
            return club_games[goal_column].head(5).mean()
    return 0.0

def calculate_win_rate(club_id, is_home=True):
    """Calculate win rate for last 5 matches"""
    if is_home:
        club_games = games_df[
            (games_df['home_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False).head(5)
        wins = (club_games['home_club_goals'] > club_games['away_club_goals']).sum()
    else:
        club_games = games_df[
            (games_df['away_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False).head(5)
        wins = (club_games['away_club_goals'] > club_games['home_club_goals']).sum()
    return wins / 5 if len(club_games) == 5 else 0.0
def calculate_form_last5(club_id, is_home=True):
    """Calculate form (points per game) for last 5 matches"""
    current_date = pd.Timestamp.today().normalize()
    
    # Filter games played by the club before today
    relevant_games = games_df[
        (games_df['date'] < current_date) & 
        ((games_df['home_club_id'] == club_id) | (games_df['away_club_id'] == club_id))
    ].sort_values('date', ascending=False)

    recent_games = relevant_games.head(5)

    points = 0
    total_games = 0

    for _, game in recent_games.iterrows():
        if game['home_club_id'] == club_id:
            goals_for = game['home_club_goals']
            goals_against = game['away_club_goals']
        else:
            goals_for = game['away_club_goals']
            goals_against = game['home_club_goals']

        if pd.notna(goals_for) and pd.notna(goals_against):
            if goals_for > goals_against:
                points += 3
            elif goals_for == goals_against:
                points += 1
            # else 0 points
            total_games += 1

    return points / total_games if total_games > 0 else 0.0

def calculate_form_trend(club_id, is_home=True):
    """Calculate form trend (difference between recent and earlier form)"""
    current_date = pd.Timestamp.today().normalize()
    
    if is_home:
        club_games = games_df[
            (games_df['home_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
    else:
        club_games = games_df[
            (games_df['away_club_id'] == club_id) &
            (games_df['date'] < current_date)
        ].sort_values('date', ascending=False)
    
    if len(club_games) >= 6:
        # Calculate form for recent 3 games
        recent_form = 0
        recent_games = club_games.iloc[:3]
        for _, game in recent_games.iterrows():
            if game['home_club_id'] == club_id:
                goals_for = game['home_club_goals']
                goals_against = game['away_club_goals']
            else:
                goals_for = game['away_club_goals']
                goals_against = game['home_club_goals']
                
            if pd.notna(goals_for) and pd.notna(goals_against):
                if goals_for > goals_against:
                    recent_form += 3
                elif goals_for == goals_against:
                    recent_form += 1
        
        # Calculate form for previous 3 games (games 3-6)
        earlier_form = 0
        earlier_games = club_games.iloc[3:6]
        for _, game in earlier_games.iterrows():
            if game['home_club_id'] == club_id:
                goals_for = game['home_club_goals']
                goals_against = game['away_club_goals']
            else:
                goals_for = game['away_club_goals']
                goals_against = game['home_club_goals']
                
            if pd.notna(goals_for) and pd.notna(goals_against):
                if goals_for > goals_against:
                    earlier_form += 3
                elif goals_for == goals_against:
                    earlier_form += 1
        
        recent_avg = recent_form / 3 if len(recent_games) == 3 else 0
        earlier_avg = earlier_form / 3 if len(earlier_games) == 3 else 0
        
        return recent_avg - earlier_avg
    
    return 0.0

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
        log_error(e)  # <-- Add this
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

def predict():
    try:
        # Load selected model
        selected_model = model_var.get()
        model = joblib.load(selected_model)
        
        home_team = home_var.get()
        away_team = away_var.get()
        referee = ref_var.get()
        venue = venue_var.get()

        if home_team == away_team:
            messagebox.showwarning("Error", "Home and away teams must be different.")
            return

        home = clubs_df[clubs_df["name"] == home_team]
        away = clubs_df[clubs_df["name"] == away_team]

        if home.empty or away.empty:
            messagebox.showerror("Error", "One or both teams not found.")
            return

        home_id = home.iloc[0]['club_id']
        away_id = away.iloc[0]['club_id']

        # Goalkeepers and form
        home_gk = get_latest_goalkeeper(home_id)
        away_gk = get_latest_goalkeeper(away_id)
        home_form_last5 = calculate_form_last5(home_id, is_home=True)
        away_form_last5 = calculate_form_last5(away_id, is_home=False)
        form_diff = home_form_last5 - away_form_last5

        home_gk_clean_sheets = get_clean_sheets_last5(home_gk, home_id) if home_gk else 0
        away_gk_clean_sheets = get_clean_sheets_last5(away_gk, away_id) if away_gk else 0
        clean_sheets_diff = home_gk_clean_sheets - away_gk_clean_sheets

        # Formations
        home_formation = get_latest_formation(home_id)
        away_formation = get_latest_formation(away_id)
        home_form_strength = calculate_formation_strength(home_id, home_formation, is_home=True)
        away_form_strength = calculate_formation_strength(away_id, away_formation, is_home=False)
        form_strength_diff = home_form_strength - away_form_strength

        home_position = get_latest_position(home_team)
        away_position = get_latest_position(away_team)
        position_diff = home_position - away_position if not np.isnan(home_position) and not np.isnan(away_position) else 0

        # Team value info
        def get_team_value(club_id):
            valuations = player_valuations_df[player_valuations_df["current_club_id"] == club_id]
            latest_dates = valuations.groupby("player_id")["date"].idxmax()
            latest_valuations = valuations.loc[latest_dates]
            return latest_valuations["market_value_in_eur"].sum()

        home_team_value = get_team_value(home_id)
        away_team_value = get_team_value(away_id)
        value_diff = home_team_value - away_team_value

        # Calculate additional features from v2.py
        relative_strength = (home_form_last5 + 0.1) / (away_form_last5 + 0.1)
        form_x_position_home = home_form_last5 * (1/max(home_position, 1))  # Avoid division by zero
        form_x_position_away = away_form_last5 * (1/max(away_position, 1))
        
        # Calculate power rankings (rolling average goals)
        home_power = calculate_rolling_average(home_id, 'home_club_goals', is_home=True)
        away_power = calculate_rolling_average(away_id, 'away_club_goals', is_home=False)
        power_rank_diff = home_power - away_power
        
        # Calculate win rates
        home_win_rate = calculate_win_rate(home_id, is_home=True)
        away_win_rate = calculate_win_rate(away_id, is_home=False)
        win_rate_diff = home_win_rate - away_win_rate
        
        # Calculate conceded goals
        home_conceded = calculate_rolling_average(home_id, 'away_club_goals', is_home=True)
        away_conceded = calculate_rolling_average(away_id, 'home_club_goals', is_home=False)
        conceded_diff = away_conceded - home_conceded
        
        # Calculate form trends
        home_trend = calculate_form_trend(home_id, is_home=True)
        away_trend = calculate_form_trend(away_id, is_home=False)
        trend_diff = home_trend - away_trend

        # Input vector for prediction - must match exactly with v2.py features
        X_input = pd.DataFrame([{
            #'position_diff': position_diff,
            'form_diff': form_diff,
            'clean_sheets_diff': clean_sheets_diff,
            #'home_club_position': home_position,
            #'away_club_position': away_position,
            'form_x_position_home': form_x_position_home,
            'form_x_position_away': form_x_position_away,
            'home_gk_clean_sheets_last5': home_gk_clean_sheets,
            'away_gk_clean_sheets_last5': away_gk_clean_sheets,
            'relative_strength': relative_strength,
            #'power_rank_diff': power_rank_diff,
            'win_rate_diff': win_rate_diff,
            #'conceded_diff': conceded_diff,
            'away_conceded': away_conceded,
            'home_conceded': home_conceded,
            #'trend_diff': trend_diff
        }])

        probs = model.predict_proba(X_input)[0]
        prediction = model.predict(X_input)[0]
        outcome = {0: "🏠 Home Win", 1: "🤝 Draw", 2: "🚗 Away Win"}

        result_text.delete("1.0", tk.END)
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
            result_text.insert(tk.END, f"  {k}: {v:.4f}\n")

        ref_avg = get_avg_cards_per_referee(referee)
        home_stats = get_club_stats(home_team)
        away_stats = get_club_stats(away_team)

        event_features = pd.DataFrame([{
            #'attendance': attendance,
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
        log_error(e)  # <-- Add this before showing messagebox
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
        log_error(e)  # <-- Add this
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
