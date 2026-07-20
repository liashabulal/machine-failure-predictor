"""One-time ingestion: load data/ai4i2020.csv into a local SQLite database.

Simulates how a production system would store incoming sensor readings
(a queryable table) instead of reading flat files directly. Run once:

    python setup_db.py
"""

import sqlite3
import pandas as pd

CSV_PATH = "data/ai4i2020.csv"
DB_PATH = "data/sensor_data.db"
TABLE_NAME = "sensor_readings"

df = pd.read_csv(CSV_PATH)

conn = sqlite3.connect(DB_PATH)
df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
conn.close()

print(f"Loaded {len(df)} rows from {CSV_PATH} into {DB_PATH} (table: {TABLE_NAME})")
