import pyreadr

df = pyreadr.read_r('data/raw/data_ml.RData')

df['data_ml'].to_parquet('data/processed/test_clean_data.parquet')