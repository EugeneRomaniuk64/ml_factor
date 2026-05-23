import pandas as pd
from src.models.linear import train_any_linear
from src.models.trees import train_random_forest, train_xgboost
from src.model_io import load_sklearn_model
import os
from src.config import (
    TRAIN_END,
    FEATURES,
    LINEAR_MODEL_NAMES
)

df = pd.read_parquet('data/processed/test_clean_data.parquet')

train = df[df['yyyymm'] <= TRAIN_END]

X_train = train[FEATURES]
y_train = train['R1M_Usd']
months_train = train['yyyymm'].values

models = {}

for name in LINEAR_MODEL_NAMES:
    if os.path.exists(f'models/{name}/model.pk1'):
        models[name] = load_sklearn_model(name)
        print(f"Successfully loaded {name} model data")
    else:
        models[name] = train_any_linear(X_train, y_train, months_train, name)

if os.path.exists(f'models/random_forest/model.pk1'):
    models['random_forest'] = load_sklearn_model('random_forest')
    print("Successfully loaded random forest model")
else:
    models['random_forest'] = train_random_forest(X_train, y_train, months_train)


if os.path.exists(f'models/xgboost/model.pk1'):
    models['xgboost'] = load_sklearn_model('xgboost')
    print("Successfully loaded xgboost model")
else:
    models['xgboost'] = train_xgboost(X_train, y_train, months_train)
      