import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedWalkForwadCV(BaseCrossValidator):
    def __init__(self, min_train_months=240, val_months=24, embargo_months=1):
        self.min_train_months = min_train_months
        self.val_months = val_months
        self.embargo_months = embargo_months

    def split(self, X, y=None, groups=None):
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
        

if __name__ == '__main__':        
    import pandas as pd

    df = pd.read_parquet('data/processed/test_clean_data.parquet')

    X = df[['Ebit_Bv', 'Capex_Ps_Cf']]
    y = df['R1M_Usd']
    groups = pd.to_datetime(df['date']).dt.year * 100 + pd.to_datetime(df['date']).dt.month
    groups = groups.values

    cv = PurgedWalkForwadCV(min_train_months=72)

    splits = list(cv.split(X, y, groups=groups))
    print(f"Number of folds: {len(splits)}")


    for i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        train_months = np.unique(groups[train_idx])
        val_months = np.unique(groups[val_idx])
        
        assert train_months.max() < val_months.min(), \
            f"Fold {i}: train bleeds into val!"
        
        print(f"Fold {i}: train ends {train_months.max()}, val starts {val_months.min()}")