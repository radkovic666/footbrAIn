import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")
clubs = pd.read_sql("SELECT * FROM game_events", conn)
print(clubs.columns.tolist())
