import numpy as np
import pandas as pd
import datetime as dt
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import BaseCrossValidator

class PurgedWalkForwardCV(BaseCrossValidator):
    def __init__(self, val_length: relativedelta, embargo_length: relativedelta, val_start: dt.datetime):
        self.val_length = val_length
        self.embargo_length = embargo_length
        self.val_start = val_start

    def split(self, X=None, y=None, groups=None):
        unique_dates = np.sort(np.unique(groups))
        
        fold_val_start = self.val_start
        
        while fold_val_start + self.val_length <= unique_dates[-1]: 
            fold_train_end = fold_val_start - self.embargo_length - relativedelta(months=1)
            fold_val_end = fold_val_start + self.val_length
            
            train_idx = np.where(groups <= fold_train_end)[0]
            val_idx = np.where((groups >= fold_val_start) & (groups < fold_val_end))[0]
            
            yield train_idx, val_idx
            
            # Advance by one val period
            fold_val_start += self.val_length

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups is None:
            return 0
        unique_dates = np.sort(np.unique(groups))
        
        fold_val_start = self.val_start
        n_folds = 0
        
        while fold_val_start + self.val_length <= unique_dates[-1]:
            n_folds += 1
            fold_val_start += self.val_length
            
        return n_folds
        
