import pandas as pd
import numpy as np
import torch
from dateutil.relativedelta import relativedelta
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

from src.models.neural import MLP, refit_mlp
from src.config import (
    TEST_START,
    TEST_FOLD_LEN,
    TEST_EMBARGO_LEN,
    DEVICE
)

class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self  # already trained, no-op
    
    def predict(self, X):
        return predict_torch(pd.DataFrame(X), self.model)

def predict_torch(X_test, model):
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X_test_tensor)
    
    return y_pred.cpu().numpy().flatten()

def predict_sklearn(X_test, model):
    y_pred = model.predict(X_test)
    return y_pred.to_numpy() if hasattr(y_pred, 'to_numpy') else y_pred

def predict(model, model_name, X, test_idx, df, fold_start, fold_end):
    if isinstance(model, MLP):
        df.loc[
            (df['eom'] > fold_start) &
            (df['eom'] <= fold_end),
            f'ret_exc_pred_{model_name}'
        ] = predict_torch(X.iloc[test_idx], model)
    else:
        df.loc[
            (df['eom'] > fold_start) &
            (df['eom'] <= fold_end),
            f'ret_exc_pred_{model_name}'
        ] = predict_sklearn(X.iloc[test_idx], model)

def refit(model, X, y, refit_idx, mlp_params):
    if isinstance(model, MLP):
        model = refit_mlp(
            X.iloc[refit_idx],
            y.iloc[refit_idx],
            **mlp_params
        )
    else:
        model.fit(X.iloc[refit_idx], y.iloc[refit_idx])
    
    return model

def get_fold_idx(dates, fold_start, fold_end, embargo):
    test_idx = np.where((dates > fold_start) & (dates <= fold_end))[0]
    refit_idx = np.where(dates <= pd.Timestamp(fold_end) - embargo)[0]
    
    return test_idx, refit_idx

def test(X_test, y_test, dates_test, X_train, y_train, dates_train, gvkeys_test, models, mlp_params): 
    df = pd.DataFrame({
        'eom': dates_test,
        'gvkey': gvkeys_test,
        'ret_exc_true': y_test.values
    })
    
    X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
    y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
    dates = np.concatenate([dates_train, dates_test], axis=0)
    
    unique_dates = np.sort(np.unique(dates))
    
    for model_name, model in models.items():
        print(f"Testing {model_name}...")
        
        fold_start = TEST_START
    
        while fold_start < unique_dates[-1]:
            fold_end = min(fold_start + TEST_FOLD_LEN + relativedelta(months=1), unique_dates[-1])

            test_idx, refit_idx = get_fold_idx(dates, fold_start, fold_end, TEST_EMBARGO_LEN)
        
            predict(model, model_name, X, test_idx, df, fold_start, fold_end)
            
            if fold_end == unique_dates[-1]:
                break
            
            model = refit(model, X, y, refit_idx, mlp_params)
                
            fold_start = fold_end
            
    
    df.to_parquet('data/processed/test_data.parquet')
    
    return df
    
    
def get_feature_importance(models, features, X_test, y_test):
    importance_dict = {}
    
    for name, model in models.items():
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if hasattr(coef, 'to_numpy'):
                coef = coef.to_numpy()
            imp = pd.Series(coef.flatten(), index=features).abs()
        elif hasattr(model, 'feature_importances_'):
            imp = pd.Series(model.feature_importances_, index=features)
        elif isinstance(model, xgb.XGBRegressor):
            imp = pd.Series(model.get_booster().get_score(importance_type='gain'))
        else:
            wrapped = MLPWrapper(model)
            result = permutation_importance(wrapped, X_test.values, y_test.values, n_repeats=5, scoring='r2')
            imp = pd.Series(result.importances_mean, index=features)
        
        importance_dict[name] = imp.sort_values(ascending=False)
    
    df = pd.DataFrame(importance_dict)
    df.to_parquet('data/processed/feature_importance.parquet')
    
    return df