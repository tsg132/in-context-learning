import os
import json
import torch
from tqdm import tqdm
import sys
from munch import Munch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import models
from samplers import get_data_sampler
from tasks import get_task_sampler
from eval import eval_model, aggregate_metrics

def evaluate_untrained_model(
    model_type="gpt2",
    task_name="sinusoidal_regression_5d",
    data_name="gaussian",
    n_dims=5,
    n_points=60,
    n_layer=12,
    n_head=8,
    n_embd=256,
    prompting_strategy="standard",
    num_eval_examples=1280,
    batch_size=64,
    save_path="untrained_sinusoidal_metrics.json",
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate an untrained model on the sinusoidal_regression_5d task.
    """
    print(f"Creating untrained {model_type} model...")
    model_config = Munch({
        "family": model_type,
        "n_dims": n_dims,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "n_positions": 2048, 
        "normalization_type": "prenorm",
        "attention_type": "linear",
    })
    
    model = models.build_model(model_config)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    print(f"Evaluating untrained model on {task_name}...")
    
    metrics = eval_model(
        model=model,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=n_points,
        prompting_strategy=prompting_strategy,
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_sampler_kwargs,
        task_sampler_kwargs=task_sampler_kwargs,
    )
    
    all_metrics = {
        "standard": {
            f"{model_type}_embd={n_embd}_layer={n_layer}_head={n_head}_untrained": metrics
        }
    }
    
    with open(save_path, "w") as fp:
        json.dump(all_metrics, fp, indent=2)
    
    print(f"Results saved to {save_path}")
    return all_metrics

if __name__ == "__main__":
    evaluate_untrained_model(
        model_type="gpt2",
        task_name="sinusoidal_regression_5d",
        data_name="gaussian",
        n_dims=5,
        n_points=60,
        n_layer=12,
        n_head=8,
        n_embd=256,
        save_path="untrained_sinusoidal_metrics.json",
        task_sampler_kwargs={},
    ) 