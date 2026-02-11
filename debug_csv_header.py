
import pandas as pd
try:
    df = pd.read_csv('artifacts/data_ingestion/train.csv', nrows=1)
    print("Train Columns:", df.columns.tolist())
    df_test = pd.read_csv('artifacts/data_ingestion/test.csv', nrows=1)
    print("Test Columns:", df_test.columns.tolist())
except Exception as e:
    print(e)
