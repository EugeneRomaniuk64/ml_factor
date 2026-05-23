import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedWalkForwardCV(BaseCrossValidator):
    def __init__(self, min_train_months, val_months, embargo_months):
        self.min_train_months = min_train_months
        self.val_months = val_months
        self.embargo_months = embargo_months

    def split(self, X=None, y=None, groups=None):
        unique_months = np.sort(np.unique(groups))
        n_months = len(unique_months)
        
        # First validation fold starts after min training window + embargo
        val_start = self.min_train_months + self.embargo_months
        
        while val_start + self.val_months <= n_months:
            train_months = unique_months[:val_start - self.embargo_months]
            val_months = unique_months[val_start:val_start + self.val_months]
            
            train_idx = np.where(np.isin(groups, train_months))[0]
            val_idx = np.where(np.isin(groups, val_months))[0]
            
            yield train_idx, val_idx
            
            # Advance by one val period
            val_start += self.val_months

    def get_n_splits(self, X=None, y=None, groups=None):
            if groups is None:
                return 0
            unique_months = np.sort(np.unique(groups))
            n_months = len(unique_months)
            val_start = self.min_train_months + self.embargo_months
            n_folds = 0
            
            while val_start + self.val_months <= n_months:
                n_folds += 1
                val_start += self.val_months
                
            return n_folds
        
