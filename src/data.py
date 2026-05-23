import openassetpricing as oap
import pathlib
import pandas as pd
import numpy as np

signals = [
    "BM",
    "EP"
]


def download_data():
    openap = oap.OpenAP()

    df = openap.dl_signal('pandas', signals)

    df.to_parquet('data/raw/signals.parquet', index = False)

    return df
    
if not pathlib.Path('data/raw/signals.parquet').exists():
    df = download_data()
else:
    df = pd.read_parquet('data/raw/signals.parquet')

df['date'] = pd.to_datetime(df['yyyymm'], format='%Y%m') + pd.offsets.MonthEnd(0)
np.random.seed(69)    
df['ret'] = np.random.normal(0, 0.05, len(df)) # placeholder
    
df = df.dropna(subset=['permno', 'yyyymm', 'ret'])

missing_df = df.isna()
for signal in signals:
    df[f'{signal}_missing'] = df[signal].isna().astype(int)

df_ranked = df
df_ranked[signals] = df.groupby('yyyymm')[signals].transform(
    lambda x: x.rank(pct=True)
)

df_ranked[missing_df] = 0.5

df_ranked['ret'] = df_ranked.groupby('yyyymm')['ret'].clip(
    lower=df_ranked['ret'].quantile(0.01),
    upper=df_ranked['ret'].quantile(0.99)
)

df_ranked.to_parquet('data/processed/clean_data.parquet')