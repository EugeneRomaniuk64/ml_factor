import pyreadr
import pandas as pd

df = pyreadr.read_r('data/raw/data_ml.RData')

df['data_ml']['yyyymm'] = pd.to_datetime(df['data_ml']['date']).dt.year * 100 + pd.to_datetime(df['data_ml']['date']).dt.month

df['data_ml'].to_parquet('data/processed/test_clean_data.parquet')