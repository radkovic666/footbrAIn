import pandas as pd
import glob

# 1. Get all CSVs in the folder
csv_files = glob.glob("data/*.csv")  # <-- put your .csv files inside a folder called 'data'

# 2. Columns we need
required_columns = [
    "club_name",
    "squad_size",
    "average_age",
    "foreigners_number",
    "national_team_players",
    "league_position",
    "avg_attendance"
]

master_df = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv(file)

    # Check if all required columns are present
    if all(col in df.columns for col in required_columns):
        df_filtered = df[required_columns]
        master_df = pd.concat([master_df, df_filtered], ignore_index=True)
    else:
        print(f"⚠️ Skipping {file} — missing one or more required columns.")

# 3. Remove duplicates based on club_name (keeping last occurrence)
master_df = master_df.drop_duplicates(subset="club_name", keep="last")

# 4. Save
master_df.to_csv("team_reference.csv", index=False)
print("✅ Built team_reference.csv with", len(master_df), "teams.")
