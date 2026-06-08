import wrds
import pandas as pd
import os
from functools import cache

def get_data(countries, features):
    country_str = ', '.join(f"'{c}'" for c in countries)
    char_str = ',\n        '.join(f'g.{c}' for c in features)

    sql_query = f"""
        WITH sec AS (
            SELECT DISTINCT ON (s.gvkey)
                    s.gvkey,
                    c.conm,
                    s.tic,
                    s.isin,
                    s.sedol
            FROM comp.g_security s
            LEFT JOIN comp.g_company c
                ON s.gvkey = c.gvkey
            WHERE s.iid = '01'          
            ORDER BY s.gvkey, s.iid
        )
        SELECT
            g.id,
            g.eom,
            g.excntry,
            g.gvkey,
            g.permno,
            g.size_grp,
            g.me,
            g.ret_exc_lead1m,
            s.conm,
            s.tic,
            s.isin,
            s.sedol,
            {char_str}
        FROM contrib.global_factor g
        LEFT JOIN sec s
            ON g.gvkey = s.gvkey
        WHERE g.common = 1
            AND g.exch_main = 1
            AND g.primary_sec = 1
            AND g.obs_main = 1
            AND g.excntry IN ({country_str})
    """
    
    wrds_db = wrds.Connection(verbose=True)
    
    for country in countries:
        out_path = f'data/raw/countries/{country}.parquet'
        if os.path.exists(out_path):
            print(f"[{country}] already downloaded, skipping")
            continue
        
        country_query = sql_query + f"        AND g.excntry = '{country}'"
        
        print(f"[{country}] downloading...", flush=True)
        df = wrds_db.raw_sql(country_query, date_cols=['eom'])
        df.to_parquet(out_path, index=False)
        print(f"{len(df):,} rows saved to {out_path}")
        

    panel = pd.concat(
        [pd.read_parquet(f'data/raw/countries/{c}.parquet') for c in countries],
        ignore_index=True
    ).sort_values(['excntry', 'gvkey', 'eom'])

    panel.to_parquet('data/raw/eu_data.parquet', index=False)
    print(f"Full panel: {panel.shape[0]:,} rows, {panel.shape[1]} columns")
    
    return panel



def preprocess(df, char_cols):
    df = df.copy()
    
    df = df[df['size_grp'] != 'nano']
    
    df = df.dropna(subset=['ret_exc_lead1m'])
    
    # Adding missing indicator column for characteristics with high missing %
    cols_with_na = [c for c in char_cols if df[c].isna().mean() > 0.05]
    
    indicators = pd.concat(
        [df[col].isna().astype(float).rename(f'{col}_missing') * 2 - 1 for col in cols_with_na],
        axis=1
    )
    
    df = pd.concat([df, indicators], axis=1)
    
    # Uniformizing characteristics on [-1, 1]
    def _rank(x):
        r = x.rank(pct=True, na_option='keep')
        return 2 * r - 1
    
    df[char_cols] = df.groupby('eom')[char_cols].transform(_rank)
    
    df[char_cols] = df[char_cols].fillna(0)
    
    # Winsorizing returns
    df['ret_exc_lead1m'] = df.groupby('eom')['ret_exc_lead1m'].transform(
        lambda x: x.clip(x.quantile(0.01), x.quantile(0.99))
    )
    
    df.to_parquet('data/processed/clean_data.parquet')
    
    return df
    