from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import numpy as np
from itertools import product


from src.cv import PurgedWalkForwardCV
from src.model_io import save_sklearn_model
from src.config import (
    CV_MIN_TRAIN_MONTHS,
    CV_VAL_MONTHS,
    CV_EMBARGO_MONTHS,
    RF_PARAMS,
    XGB_PARAMS
)



def train_random_forest(X_train, y_train, months_train):
    print("=" * 100)
    print(f"\nTraining random forest...")
    
    model = RandomForestRegressor(
        n_jobs=-1,
        random_state=69
    )
    
    cv = PurgedWalkForwardCV(
        min_train_months=CV_MIN_TRAIN_MONTHS,
        val_months=CV_VAL_MONTHS,
        embargo_months=CV_EMBARGO_MONTHS
    )
    
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=RF_PARAMS,
        n_iter=20,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        refit=True,
        verbose=3,
        random_state=69
    )
    
    search.fit(X_train, y_train, groups=months_train)
    
    print("=" * 100)
    print(f"random forest best params: {search.best_params_}")
    print(f"random forest best R2: {search.best_score_:.4f}")
    
    save_sklearn_model(search.best_estimator_, 'random_forest')
    
    return search.best_estimator_


def train_xgboost(X_train, y_train, months_train):
    print("=" * 100)
    print(f"\nTraining xgboost...")
    
    cv = PurgedWalkForwardCV(
        min_train_months=CV_MIN_TRAIN_MONTHS,
        val_months=CV_VAL_MONTHS,
        embargo_months=CV_EMBARGO_MONTHS
    )
    
    best_score = -np.inf
    best_params = None
   
    keys = list(XGB_PARAMS.keys())
    values = list(XGB_PARAMS.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    n_splits = cv.get_n_splits(groups=months_train)
    
    print(f"Fitting {n_splits} folds of {len(combinations)} candidates, totalling {n_splits * len(combinations)} fits")
    
    for params in combinations:
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X_train, groups=months_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            model = XGBRegressor(
                **params,
                device='cuda',
                early_stopping_rounds=20,
                eval_metric='rmse',
                random_state=69,
                n_jobs=-1,
                verbosity=0
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
            
            print(f"END {params}, score={score:.4f}")
        
        mean_score = np.mean(fold_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"New best {params}, R2 = {mean_score:.4f}")
            
    
    final_model = XGBRegressor(
        **best_params,
        device='cuda',
        random_state=69,
        n_jobs=-1,
        verbosity=0
    )
    
    print("=" * 100)
    print(f"xgboost best params: {best_params}")
    print(f"xgboost best R2: {best_score:.4f}")
    
    final_model.fit(X_train, y_train)
    
    save_sklearn_model(final_model, 'xgboost')
    
    return final_model  
    

