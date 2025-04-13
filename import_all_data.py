import sqlite3
import pandas as pd
import os

DATA_DIR = "/home/magilinux/footpredict"
DB_PATH = os.path.join(DATA_DIR, "football_data.db")

# Map of CSV filenames to their desired SQL table names
csv_table_map = {
    "clubs.csv": "clubs",
    "players.csv": "players",
    "games.csv": "games",
    "club_games.csv": "club_games",
    "appearances.csv": "appearances",
    "game_events.csv": "game_events",
    "player_valuations.csv": "player_valuations",
    "game_lineups.csv": "game_lineups",
    "transfers.csv": "transfers",
    "competitions.csv": "competitions"
}

def create_or_update_table_from_csv(csv_path, table_name, conn):
    print(f"\n📄 Loading {os.path.basename(csv_path)} into table `{table_name}`")
    
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Convert date columns to yyyy-mm-dd
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')

    # Create or append to table
    df.to_sql(table_name, conn, if_exists='append', index=False)

    # Check existing columns
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_cols = set(row[1] for row in cursor.fetchall())

    # Add missing columns
    for col in df.columns:
        if col not in existing_cols:
            dtype = df[col].dtype
            if dtype == 'int64':
                sql_type = 'INTEGER'
            elif dtype == 'float64':
                sql_type = 'REAL'
            else:
                sql_type = 'TEXT'
            try:
                cursor.execute(f'ALTER TABLE {table_name} ADD COLUMN {col} {sql_type}')
                print(f"➕ Added missing column `{col}` ({sql_type}) to `{table_name}`")
            except sqlite3.OperationalError:
                print(f"⚠️ Failed to add column `{col}` — might already exist or conflict")
    conn.commit()

def main():
    conn = sqlite3.connect(DB_PATH)
    print(f"📂 Connected to database: {DB_PATH}")

    for csv_file, table_name in csv_table_map.items():
        full_path = os.path.join(DATA_DIR, csv_file)
        if os.path.exists(full_path):
            create_or_update_table_from_csv(full_path, table_name, conn)
        else:
            print(f"❌ Missing CSV: {csv_file}")
    
    conn.close()
    print("\n✅ All done!")

if __name__ == "__main__":
    main()
