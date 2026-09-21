from cuml.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np
from itertools import product


from src.cv import PurgedWalkForwardCV
from src.model_io import save_sklearn_model
from src.config import (
    CV_EMBARGO_LEN,
    CV_FOLD_LEN,
    CV_START,
    RF_PARAMS,
    XGB_PARAMS
)



def train_random_forest(X_train, y_train, dates_train):
    print("=" * 100)
    print(f"\nTraining random forest...")
    
    
    cv = PurgedWalkForwardCV(
        val_length=CV_FOLD_LEN,
        embargo_length=CV_EMBARGO_LEN,
        val_start=CV_START
    )
    
    best_score = -np.inf
    best_params = None
   
    keys = list(RF_PARAMS.keys())
    values = list(RF_PARAMS.values())
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
            
            model = RandomForestRegressor(
                n_estimators=500,
                random_state=69,
                **params
            )
            
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
            
    
    final_model = RandomForestRegressor(
        n_estimators=500,
        random_state=69,
        **best_params
    )
    
    print("=" * 100)
    print(f"random_forest best params: {best_params}")
    print(f"random_forest best R2: {best_score:.4f}")
    
    final_model.fit(X_train, y_train)
    
    save_sklearn_model(final_model, 'random_forest')
    
    return final_model  


def train_xgboost(X_train, y_train, dates_train):
    print("=" * 100)
    print(f"\nTraining xgboost...")
    
    cv = PurgedWalkForwardCV(
        val_length=CV_FOLD_LEN,
        embargo_length=CV_EMBARGO_LEN,
        val_start=CV_START
    )
    
    best_score = -np.inf
    best_params = None
    best_iter = None
   
    keys = list(XGB_PARAMS.keys())
    values = list(XGB_PARAMS.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    n_splits = cv.get_n_splits(groups=dates_train)
    n_combinations = len(combinations)
    
    print(f"Fitting {n_splits} folds of {n_combinations} candidates, totalling {n_splits * n_combinations} fits")
    
    for params in combinations:
        fold_scores = []
        fold_best_iter = []
        
        for train_idx, val_idx in cv.split(X_train, groups=dates_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = XGBRegressor(
                device='cuda',
                early_stopping_rounds=20,
                eval_metric='rmse',
                random_state=69,
                n_jobs=-1,
                verbosity=0,
                n_estimators=3000,
                **params
            )
            
            model.fit(
                X_fold_train,
                y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False
            )
            
            preds = model.predict(X_fold_val)
            score = r2_score(y_fold_val, preds)
            fold_scores.append(score)
            fold_best_iter.append(model.best_iteration)
            
            print(f"END {params}, score={score:.4f}")
        
        mean_score = np.mean(fold_scores)
        mean_iter = int(np.mean(fold_best_iter))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_iter = mean_iter
            print(f"New best {best_params}, R2 = {mean_score:.4f}")
            
    
    final_model = XGBRegressor(
        n_estimators=best_iter,
        device='cuda',
        random_state=69,
        n_jobs=-1,
        verbosity=0,
        **best_params
    )
    
    print("=" * 100)
    print(f"xgboost best params: {best_params}")
    print(f"xgboost best R2: {best_score:.4f}")
    
    final_model.fit(X_train, y_train)
    
    save_sklearn_model(final_model, 'xgboost')
    
    return final_model  
    

