from pathlib import Path
import sqlite3
import pandas as pd

conn = sqlite3.connect("database/cmapss.db")

queries = Path("sql/queries.sql").read_text(encoding="utf-8").split(";")

query = queries[0].strip()  # 0 = première requête, 1 = deuxième, etc.

df = pd.read_sql_query(query, conn)

print(df)

conn.close()