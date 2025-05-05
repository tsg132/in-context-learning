import os
import json
import argparse
import torch
from tasks import FourierFitBaseline, get_task_sampler
from samplers import get_data_sampler
from eval import eval_model, aggregate_metrics

def evaluate_fourier_baseline(
    task_name="sinusoidal_regression_10d",
    n_dims=10,
    n_points=41,
    alpha=0.01,
    n_harmonics=3,
    prompting_strategy="standard",
    num_eval_examples=128,
    batch_size=32,
    output_file=None
):
    """
    Evaluate just the FourierFitBaseline on a specific task.
    
    Args:
        task_name: Name of the task to evaluate
        n_dims: Number of dimensions for the task
        n_points: Number of points to evaluate
        alpha: Regularization parameter for FourierFitBaseline
        n_harmonics: Number of harmonics for FourierFitBaseline
        prompting_strategy: Strategy for generating prompts
        num_eval_examples: Number of examples to evaluate
        batch_size: Batch size for evaluation
        output_file: File to save the metrics to
    """
    print(f"Evaluating FourierFitBaseline on {task_name} with:")
    print(f"- n_dims: {n_dims}")
    print(f"- n_harmonics: {n_harmonics}")
    print(f"- alpha: {alpha}")
    
    # Create baseline model
    fourier_baseline = FourierFitBaseline(n_dims=n_dims, n_harmonics=n_harmonics, alpha=alpha)
    
    # Prepare evaluation parameters
    data_name = "gaussian" # Default data sampler
    data_sampler_kwargs = {}
    task_sampler_kwargs = {}
    
    # Run evaluation
    metrics = eval_model(
        model=fourier_baseline,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=n_points,
        prompting_strategy=prompting_strategy,
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_sampler_kwargs,
        task_sampler_kwargs=task_sampler_kwargs
    )
    
    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({fourier_baseline.name: metrics}, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate only the FourierFitBaseline")
    parser.add_argument("--task", type=str, default="sinusoidal_regression_10d", 
                       help="Task name to evaluate")
    parser.add_argument("--n_dims", type=int, default=10, 
                       help="Number of dimensions")
    parser.add_argument("--n_points", type=int, default=41, 
                       help="Number of points to evaluate")
    parser.add_argument("--n_harmonics", type=int, default=3, 
                       help="Number of harmonics for Fourier baseline")
    parser.add_argument("--alpha", type=float, default=0.01, 
                       help="Regularization parameter for Fourier baseline")
    parser.add_argument("--num_eval", type=int, default=128, 
                       help="Number of evaluation examples")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="fourier_baseline_metrics.json", 
                       help="Output file for saving metrics")
    
    args = parser.parse_args()
    
    metrics = evaluate_fourier_baseline(
        task_name=args.task,
        n_dims=args.n_dims,
        n_points=args.n_points,
        alpha=args.alpha,
        n_harmonics=args.n_harmonics,
        num_eval_examples=args.num_eval,
        batch_size=args.batch_size,
        output_file=args.output
    )
    
    # Print summary of results
    print("\nEvaluation Results Summary:")
    print(f"Mean MSE at 10 examples: {metrics['mean'][10]:.6f}")
    print(f"Mean MSE at 20 examples: {metrics['mean'][20]:.6f}")
    print(f"Mean MSE at 30 examples: {metrics['mean'][30]:.6f}")
    print(f"Mean MSE at 40 examples: {metrics['mean'][40]:.6f}") 