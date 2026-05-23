import os
import joblib
import json

def save_sklearn_model(model, name):
    model_dir = os.path.join('models', name)
    os.makedirs(model_dir, exist_ok=True)
    
    joblib.dump(model, os.path.join(model_dir, 'model.pk1'))
    
    print(f"Saved {name} to {model_dir}")
    
def load_sklearn_model(name):
    model_dir = os.path.join('models', name)
    
    model = joblib.load(os.path.join(model_dir, 'model.pk1'))
    
    return model