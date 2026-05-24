import os
import joblib
import torch
import json
from src.config import DEVICE, MLP_INPUT_SIZE

def save_sklearn_model(model, name):
    model_dir = os.path.join('models', name)
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pk1'))
    
    print(f"Saved {name} to {model_dir}")
    
def load_sklearn_model(name):
    model_dir = os.path.join('models', name)
    
    model = joblib.load(os.path.join(model_dir, 'model.pk1'))
    
    return model

def save_pytorch_model(model, params, name):
    model_dir = os.path.join('models', name)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    
    with open(os.path.join(model_dir, 'params.json'), 'w') as f:
        json.dump(params, f)
    
def load_pytorch_model(model_class, name):
    model_dir = os.path.join('models', name)
    
    with open(os.path.join(model_dir, 'params.json'), 'r') as f:
        params = json.load(f)
    
    model = model_class(
        input_size=MLP_INPUT_SIZE,
        hidden_sizes=tuple(params['hidden_sizes']),
        dropout=params['dropout']
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(os.path.join(model_dir, 'model.pt'), map_location=DEVICE))
    
    return model