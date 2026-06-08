import pandas as pd
import os

from data import get_data, preprocess
from src.models.linear import train_any_linear
from src.models.trees import train_random_forest, train_xgboost
from src.models.neural import MLP, train_mlp
from src.model_io import load_sklearn_model, load_pytorch_model
from src.config import (
    TRAIN_END,
    TEST_START,
    COUNTRIES,
    EXCLUDE,
    LINEAR_MODEL_NAMES
)

def main():
    chars = pd.read_excel('data/raw/factor_details.xlsx')
    features = chars[chars['abr_jkp'].notna()]['abr_jkp'].tolist()
    
    if os.path.exists('data/processed/clean_data.parquet'):
        print("Loading clean data...")
        df = pd.read_parquet('data/processed/clean_data.parquet')
    elif os.path.exists('data/raw/eu_data.parquet'):
        print("Loading raw data...")
        df_raw = pd.read_parquet('data/raw/eu_data.parquet')
        print("Preprocessing data...")
        df = preprocess(df_raw, features)
    else:
        print("Downloading raw data...")
        df_raw = get_data(COUNTRIES, features)
        print("Preprocessing data...")
        df = preprocess(df_raw, features)
        
    features = list(set(df.columns).difference(set(EXCLUDE)))
    

    train_df = df[df['eom'] <= TRAIN_END]

    X_train = train_df[features]
    y_train = train_df['ret_exc_lead1m']
    dates_train = train_df['eom'].values
    
    test_df = df[df['eom'] >= TEST_START]
    
    X_test = test_df[features]
    y_test = test_df['ret_exc_lead1m']
    dates_test = test_df['eom'].values


    models = {}


    # for name in LINEAR_MODEL_NAMES:
    #     if os.path.exists(f'models/{name}/model.pkl'):
    #         models[name] = load_sklearn_model(name)
    #         print(f"Successfully loaded {name} model data")
    #     else:
    #         models[name] = train_any_linear(X_train, y_train, months_train, name)

    # if os.path.exists(f'models/random_forest/model.pkl'):
    #     models['random_forest'] = load_sklearn_model('random_forest')
    #     print("Successfully loaded random forest model")
    # else:
    #     models['random_forest'] = train_random_forest(X_train, y_train, months_train)


    # if os.path.exists(f'models/xgboost/model.pkl'):
    #     models['xgboost'] = load_sklearn_model('xgboost')
    #     print("Successfully loaded xgboost model")
    # else:
    #     models['xgboost'] = train_xgboost(X_train, y_train, months_train)
        
    if os.path.exists(f'models/mlp/model.pt'):
        models['mlp'] = load_pytorch_model(MLP, 'mlp', len(features))
        print("Successfully loaded mlp model")
    else:
        models['mlp'] = train_mlp(X_train, y_train, dates_train, len(features))
        
if __name__ == '__main__':
    main()