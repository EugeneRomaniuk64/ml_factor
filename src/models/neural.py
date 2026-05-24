import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import r2_score
import copy
import numpy as np
from itertools import product

from src.model_io import save_pytorch_model
from src.cv import PurgedWalkForwardCV
from src.config import (
    DEVICE,
    CV_MIN_TRAIN_MONTHS,
    CV_VAL_MONTHS,
    CV_EMBARGO_MONTHS,
    MLP_PARAMS_STAGE1,
    MLP_PARAMS_STAGE2,
    MLP_INPUT_SIZE,
    MLP_NUM_EPOCHS,
    MLP_PATIENCE,
    MLP_STAGE1_LR,
    MLP_STAGE1_BATCH_SIZE
)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout):
        super().__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

def train_fold(
    X_fold_train,
    y_fold_train,
    X_fold_val,
    y_fold_val, 
    hidden_sizes,
    dropout,
    learning_rate,
    batch_size,
    num_epochs
):
    X_train_tensor = torch.tensor(X_fold_train.values, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_fold_train.values, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_fold_val.values, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_fold_val.values, dtype=torch.float32).to(DEVICE)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLP(
        input_size=MLP_INPUT_SIZE,
        hidden_sizes=hidden_sizes,
        dropout=dropout
    ).to(DEVICE)
    
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    patience_counter = 0
    best_val_loss = float('inf')
    best_r2 = None
    best_epoch = None
    
    
    for epoch in range(num_epochs):
        model.train() # Activates Dropout and makes BatchNorm use the curren batch's mean and variance
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        model.eval() # Disables training mode, i.e., undoes what model.train() did
        with torch.no_grad(): # Don't need computation graph for validation
            val_pred = model(X_val_tensor)
            val_loss = loss_fn(val_pred, y_val_tensor).item()
            r2 = r2_score(y_fold_val, val_pred.cpu().numpy())
            
        print(f"Epoch {epoch + 1}: val_loss={val_loss:.4f}, r2={r2:.4f}")
        
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_r2 = r2
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= MLP_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    
    return best_r2, best_epoch
   
def refit_mlp(
    X_train,
    y_train,
    hidden_sizes,
    dropout,
    learning_rate,
    batch_size,
    num_epochs
):
    print("=" * 100)
    print(f"\nRefittng the model")
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(DEVICE)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = MLP(
        input_size=MLP_INPUT_SIZE,
        hidden_sizes=hidden_sizes,
        dropout=dropout
    ).to(DEVICE)
    
    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}")
    
    return model
 
# We find the ideal hidden layer size and dropout first with other params fixed
def run_stage1(X_train, y_train, months_train, cv):
    best_score = -np.inf
    best_params = None
    
    keys = list(MLP_PARAMS_STAGE1.keys())
    values = list(MLP_PARAMS_STAGE1.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    n_splits = cv.get_n_splits(groups=months_train)
    n_combinations = len(combinations)
    
    print("=" * 100)
    print(f"\nStage 1. Fitting {n_splits} folds of {n_combinations} candidates, totalling {n_splits * n_combinations} fits")
    
    for params in combinations:
        fold_scores = []
        
        for train_idx, val_idx in cv.split(X_train, groups=months_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            score, _ = train_fold(
                X_fold_train,
                y_fold_train,
                X_fold_val,
                y_fold_val,
                **params,
                learning_rate=MLP_STAGE1_LR,
                batch_size=MLP_STAGE1_BATCH_SIZE,
                num_epochs=MLP_NUM_EPOCHS
            )
            
            fold_scores.append(score)
            
            print(f"Fold done | {params} | score={score:.4f}")
        
        mean_score = np.mean(fold_scores)
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            print(f"New best {best_params}, R2 = {mean_score:.4f}")
            
        print(f"Combination done | params={params} | mean_score={mean_score:.4f}")
            
    return best_params

# We now find the best learning rate and batch size, with fixed params from stage 1
def run_stage2(X_train, y_train, months_train, cv, hidden_sizes, dropout):
    best_score = -np.inf
    best_avg_epoch = None
    best_params = None
    
    keys = list(MLP_PARAMS_STAGE2.keys())
    values = list(MLP_PARAMS_STAGE2.values())
    combinations = [dict(zip(keys, combo)) for combo in product(*values)]
    n_splits = cv.get_n_splits(groups=months_train)
    n_combinations = len(combinations)
    
    print("=" * 100)
    print(f"\nStage 2. Fitting {n_splits} folds of {n_combinations} candidates, totalling {n_splits * n_combinations} fits")
    
    for params in combinations:
        fold_scores, fold_epochs = [], []

        for train_idx, val_idx in cv.split(X_train, groups=months_train):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            score, epoch = train_fold(
                X_fold_train,
                y_fold_train,
                X_fold_val,
                y_fold_val,
                **params,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
                num_epochs=MLP_NUM_EPOCHS
            )
            
            fold_scores.append(score)
            fold_epochs.append(epoch)
            
            print(f"Fold done | params={params} | score={score:.4f}")
        
        mean_score = np.mean(fold_scores)
        mean_epoch = int(np.mean(fold_epochs))
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_avg_epoch = mean_epoch
            print(f"New best {best_params}, R2 = {mean_score:.4f}")
        
        print(f"Combination done | params={params} | mean_score={mean_score:.4f}")
            
    return best_params, best_avg_epoch
    
def train_mlp(X_train, y_train, months_train):
    print("=" * 100)
    print(f"\nTraining mlp...")
    
    cv = PurgedWalkForwardCV(
        min_train_months=CV_MIN_TRAIN_MONTHS,
        val_months=CV_VAL_MONTHS,
        embargo_months=CV_EMBARGO_MONTHS
    )
    
    best_stage1_params = run_stage1(X_train, y_train, months_train, cv)
    
    best_stage2_params, best_avg_epochs = run_stage2(X_train, y_train, months_train, cv, **best_stage1_params)
    
    best_params = {**best_stage1_params, **best_stage2_params, 'num_epochs': best_avg_epochs}
    
    model = refit_mlp(
        X_train,
        y_train,
        **best_params
    )
    
    save_pytorch_model(
        model, 
        params=best_params,
        name='mlp'
    )
    
    return model