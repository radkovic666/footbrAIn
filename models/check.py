import sqlite3
import pandas as pd

conn = sqlite3.connect("/home/magilinux/footpredict/football_data.db")
clubs = pd.read_sql("SELECT * FROM games", conn)
print(clubs.columns.tolist())
