from cuml.linear_model import Ridge, Lasso, ElasticNet
import numpy as np
from itertools import product
from sklearn.metrics import r2_score

from src.config import (
    CV_EMBARGO_LEN,
    CV_FOLD_LEN,
    CV_START,
    RIDGE_PARAMS,
    LASSO_PARAMS,
    ELASTICNET_PARAMS
)
from src.cv import PurgedWalkForwardCV
from src.model_io import save_sklearn_model


def train_linear_model(name, Model, param_grid, X_train, y_train, dates_train):
    print("=" * 100)
    print(f"\nTraining {name}...")
    
    cv = PurgedWalkForwardCV(
        val_length=CV_FOLD_LEN,
        embargo_length=CV_EMBARGO_LEN,
        val_start=CV_START
    )
    
    best_score = -np.inf
    best_params = None
   
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    n_splits = cv.get_n_splits(groups=dates_train)
    n_combinations = len(combinations)
    
    print(f"Fitting {n_splits} folds of {n_combinations} candidates, totalling {n_splits * n_combinations} fits")
    
    for params in combinations:
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X_train, groups=dates_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = Model(**params)
            
            model.fit(X_fold_train, y_fold_train)
            
            preds = model.predict(X_fold_val)
            score = r2_score(y_fold_val, preds)
            fold_scores.append(score)
                        
            print(f"END {params}, score={score:.4f}")
        
        mean_score = np.mean(fold_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"New best {best_params}, R2 = {mean_score:.4f}")
            
    final_model = Model(**best_params)
            
    print("=" * 100)
    print(f"{name} best params: {best_params}")
    print(f"{name} best R2: {best_score:.4f}")
    
    final_model.fit(X_train, y_train)
    
    save_sklearn_model(final_model, name)
    
    return final_model

def train_ridge(X_train, y_train, dates_train):
    Model = Ridge
    return train_linear_model('ridge', Model, RIDGE_PARAMS, X_train, y_train, dates_train)

def train_lasso(X_train, y_train, dates_train):
    Model = Lasso
    return train_linear_model('lasso', Model, LASSO_PARAMS, X_train, y_train, dates_train)

def train_elasticnet(X_train, y_train, dates_train):
    Model = ElasticNet
    return train_linear_model('elasticnet', Model, ELASTICNET_PARAMS, X_train, y_train, dates_train)


def train_any_linear(X_train, y_train, dates_train, name):
    if name == 'ridge':  
        return train_ridge(X_train, y_train, dates_train)
    elif name == 'lasso':
        return train_lasso(X_train, y_train, dates_train)
    elif name == 'elasticnet':
        return train_elasticnet(X_train, y_train, dates_train)
