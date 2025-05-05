import os
import sys
import json
import torch
import numpy as np
from tasks import KernelRidgeFixedBaseline, RFFRegressionFixed, get_task_sampler
from samplers import get_data_sampler
import models
from eval import eval_batch, eval_model, aggregate_metrics, build_evals, compute_evals
from tqdm import tqdm

# Path to the fixed kernel parameters
fixed_pool_path = "models/rff_fixed/3b8a0ce2-a1bb-4acd-92ca-a482a650e3f6/fixed_pool.pt"

# Load the fixed kernel parameters
fixed_pool = torch.load(fixed_pool_path)
w_rff = fixed_pool['w_rff']
b_rff = fixed_pool['b_rff']

print(f"Loaded w_rff with shape: {w_rff.shape}")
print(f"Loaded b_rff with shape: {b_rff.shape}")

# Parameters
n_dims = 5  #
rff_dim = w_rff.shape[0]  
batch_size = 64
n_points = 60  # Number of points in the sequence
num_eval_examples = 1280  # Total examples to evaluate

# Create kernel ridge regresxsion baseline
class KernelRidgeWrapper:
    def __init__(self, w_rff, b_rff, alpha=1e-6):
        self.kernel_ridge = KernelRidgeFixedBaseline(w_rff, b_rff, alpha)
        self.name = "kernel_ridge_fixed"
    
    def __call__(self, xs, ys, inds=None):
        """
        In-context learning style prediction for model evaluation.
        xs: [batch_size, n_points, n_dims]
        ys: [batch_size, n_points]
        inds: Which indices to predict (default: all)
        Returns: Predictions for each index in inds
        """
        B, T, _ = xs.shape
        if inds is None:
            inds = range(T)
            
        device = xs.device
        preds = []
        
        for i in inds:
            batch_preds = torch.zeros(B, device=device)
            if i == 0:
                preds.append(batch_preds)
                continue
                
            for b in range(B):
                train_xs, train_ys = xs[b, :i], ys[b, :i]
                self.kernel_ridge.fit(train_xs, train_ys)
                
                test_x = xs[b, i:i+1]
                pred = self.kernel_ridge.predict(test_x)
                
                if isinstance(pred, torch.Tensor):
                    batch_preds[b] = pred.squeeze()
                else:
                    batch_preds[b] = torch.tensor(pred, device=device).squeeze()
                    
            preds.append(batch_preds)
            
        return torch.stack(preds, dim=1)

# Create our models list
kernel_ridge = KernelRidgeWrapper(w_rff, b_rff, alpha=1e-6)
all_models = [kernel_ridge]

# Add ALL other baseline models for comparison (matching models.py rff_fixed list)
baselines = [
    models.NNModel(n_neighbors=3),
    models.DecisionTreeModel(max_depth=4),
    models.DecisionTreeModel(max_depth=None),
    models.XGBoostModel(),
    models.AveragingModel()
]
all_models.extend(baselines)

try:
    from eval import get_model_from_run
    model_path = "models/rff_fixed/3b8a0ce2-a1bb-4acd-92ca-a482a650e3f6"
    trained_model, conf = get_model_from_run(model_path)
    if trained_model is not None:
        trained_model = trained_model.cuda().eval()
        all_models.insert(0, trained_model)  
        print("Loaded trained model from", model_path)
except Exception as e:
    print(f"Could not load trained model: {e}")

# Evaluation
def evaluate_models():
    base_kwargs = {
        "task_name": "rff_fixed",
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": "gaussian",
        "prompting_strategy": "standard",
        "task_sampler_kwargs": {"rff_dim": rff_dim, "pool_dict": {"w_rff": w_rff, "b_rff": b_rff}}
    }
    
    evaluation_kwargs = {"standard": base_kwargs}
    
    print(f"Evaluation kwargs: {evaluation_kwargs.keys()}")
    
    # Evaluate all models
    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path="rff_fixed_metrics.json", recompute=True)
    
    print(f"Evaluated settings: {all_metrics.keys()}")
    print(f"Models evaluated: {list(all_metrics['standard'].keys())}")
    
    return all_metrics

if __name__ == "__main__":
    metrics = evaluate_models()
    
    # Save full results
    with open("FINAL_rff_fixed_evaluation_results.json", "w") as f:
        json.dump(metrics, f, indent=2) 