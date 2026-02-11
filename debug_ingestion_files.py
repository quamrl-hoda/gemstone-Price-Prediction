
import pandas as pd
import os

files = [
    'artifacts/data_ingestion/train.csv',
    'artifacts/data_ingestion/gemstone.csv',
    'artifacts/data_ingestion/data.csv'
]

for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f, nrows=1)
            print(f"File: {f}")
            print(f"Columns: {df.columns.tolist()}")
        except Exception as e:
            print(f"Error reading {f}: {e}")
    else:
        print(f"File not found: {f}")
