from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
import pandas as pd

from src.config import (
    TEST_START,
    TRAIN_END,
    FEATURES,
    CV_MIN_TRAIN_MONTHS,
    CV_VAL_MONTHS,
    CV_EMBARGO_MONTHS,
    RIDGE_PARAMS,
    LASSO_PARAMS,
    ELASTICNET_PARAMS
)
from src.cv import PurgedWalkForwardCV
from src.model_io import save_sklearn_model


def train_linear_model(name, model, param_grid, X_train, y_train, months_train):
    print("=" * 100)
    print(f"\nTraining {name}...")
    
    cv = PurgedWalkForwardCV(
        min_train_months=CV_MIN_TRAIN_MONTHS,
        val_months=CV_VAL_MONTHS,
        embargo_months=CV_EMBARGO_MONTHS
    )
    
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,
        refit=True,
        verbose=3
    )
    
    search.fit(X_train, y_train, groups=months_train)
    
    print("=" * 100)
    print(f"{name} best params: {search.best_params_}")
    print(f"{name} best R2: {search.best_score_:.4f}")
    
    save_sklearn_model(search.best_estimator_, name)
    
    return search.best_estimator_

def train_ridge(X_train, y_train, months_train):
    model = Ridge()
    return train_linear_model('ridge', model, RIDGE_PARAMS, X_train, y_train, months_train)

def train_lasso(X_train, y_train, months_train):
    model = Lasso()
    return train_linear_model('lasso', model, LASSO_PARAMS, X_train, y_train, months_train)

def train_elasticnet(X_train, y_train, months_train):
    model = ElasticNet()
    return train_linear_model('elasticnet', model, ELASTICNET_PARAMS, X_train, y_train, months_train)


def train_any_linear(X_train, y_train, months_train, name):
     
    if name == 'ridge':  
        return train_ridge(X_train, y_train, months_train)
    elif name == 'lasso':
        return train_lasso(X_train, y_train, months_train)
    elif name == 'elasticnet':
        return train_elasticnet(X_train, y_train, months_train)
