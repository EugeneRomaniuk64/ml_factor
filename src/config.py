import torch
import pandas as pd
from dateutil.relativedelta import relativedelta

# Data Parameters
TRAIN_END = pd.Timestamp(year=2018, month=12, day=31)
TEST_START = pd.Timestamp(year=2019, month=1, day=31)
TEST_FOLD_LEN = relativedelta(years=1)
TEST_EMBARGO_LEN = relativedelta(months=1)

COUNTRIES = [
    'AUT', 'BEL', 'DNK', 'FIN', 'FRA', 'DEU', 'IRL',
    'ITA', 'NLD', 'NOR', 'PRT', 'ESP', 'SWE', 'CHE', 'GBR'
]

EXCLUDE = [
    'id', 'eom', 'excntry', 'gvkey', 'permno', 'size_grp',
    'me', 'ret_exc_lead1m', 'conm', 'tic', 'isin', 'sedol'
]

# Purged Walk-Ahead Cross-Validation Parameters

CV_START = pd.Timestamp(year=2012, month=12, day=31)
CV_FOLD_LEN = relativedelta(years=1)
CV_EMBARGO_LEN = relativedelta(months=1)

# Model Hyperparameters
LINEAR_MODEL_NAMES = ['ridge', 'lasso', 'elasticnet']

RIDGE_PARAMS = {
    'alpha': [0.01, 0.1, 1, 10, 100]
}

LASSO_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1]
}

ELASTICNET_PARAMS = {
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'l1_ratio': [0.3, 0.5, 0.7]
}


RF_PARAMS = {              
    'max_features': [5, 10, 25, 50],   
    'min_samples_leaf': [1000, 5000]      
}

XGB_PARAMS = {
    'max_depth': [1, 2],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.5, 0.8],
    'colsample_bytree': [0.5, 0.8],
    'min_child_weight': [1000, 5000]
}

MLP_NUM_EPOCHS = 100
MLP_PATIENCE = 15
MLP_STAGE1_LR = 0.001
MLP_STAGE1_BATCH_SIZE = 10000

MLP_PARAMS_STAGE1 = {
    'hidden_sizes': [
        (32,),
        (32, 16),
        (32, 16, 8),
        (64, 32),
        (64, 32, 16)
    ],
    'dropout': [0.1, 0.3, 0.5]
}

MLP_PARAMS_STAGE2 = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [4096, 10000] 
}

DEVICE = torch.device('cuda')
