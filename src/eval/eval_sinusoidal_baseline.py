import os
import json
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tasks import SinusoidalRegressionBaseline, SinusoidalRegression, FourierFitBaseline
from eval import eval_model, aggregate_metrics

def evaluate_baselines(
    n_dims=10,
    n_points=41,
    alpha=0.001,
    lr=0.05,
    n_iterations=300,
    num_eval_examples=128,
    batch_size=32,
    output_file="sinusoidal_baseline_metrics.json"
):
    """
    Evaluate the SinusoidalRegressionBaseline against the FourierFitBaseline.
    """
    print(f"Evaluating on sinusoidal_regression_{n_dims}d task with:")
    print(f"- Learning rate: {lr}")
    print(f"- Regularization (alpha): {alpha}")
    print(f"- Max iterations: {n_iterations}")
    
    # Create baselines
    sinusoidal_baseline = SinusoidalRegressionBaseline(
        n_dims=n_dims, 
        lr=lr, 
        n_iterations=n_iterations, 
        alpha=alpha
    )
    
    fourier_baseline = FourierFitBaseline(
        n_dims=n_dims, 
        n_harmonics=7, 
        alpha=alpha
    )
    
    # Create task sampler and data sampler
    task_name = f"sinusoidal_regression_{n_dims}d"
    data_name = "gaussian"
    task_sampler_kwargs = {}
    data_sampler_kwargs = {}
    
    # Run evaluation for sinusoidal baseline
    print("\nEvaluating Sinusoidal Regression Baseline...")
    sin_metrics = eval_model(
        model=sinusoidal_baseline,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=n_points,
        prompting_strategy="standard",
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_sampler_kwargs,
        task_sampler_kwargs=task_sampler_kwargs
    )
    
    # Run evaluation for Fourier baseline
    print("\nEvaluating Fourier Baseline for comparison...")
    fourier_metrics = eval_model(
        model=fourier_baseline,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=n_points,
        prompting_strategy="standard",
        num_eval_examples=num_eval_examples,
        batch_size=batch_size,
        data_sampler_kwargs=data_sampler_kwargs,
        task_sampler_kwargs=task_sampler_kwargs
    )
    
    # Save results
    metrics = {
        sinusoidal_baseline.name: sin_metrics,
        fourier_baseline.name: fourier_metrics
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Results saved to {output_file}")
    
    # Plot comparison
    plot_comparison(metrics, n_dims, output_file.replace('.json', '.png'))
    
    # Print summary
    print("\nResults Summary (Sinusoidal Baseline):")
    print(f"Mean MSE at 5 examples: {sin_metrics['mean'][5]:.6f}")
    print(f"Mean MSE at 10 examples: {sin_metrics['mean'][10]:.6f}")
    print(f"Mean MSE at 20 examples: {sin_metrics['mean'][20]:.6f}")
    print(f"Mean MSE at 30 examples: {sin_metrics['mean'][30]:.6f}")
    print(f"Mean MSE at 40 examples: {sin_metrics['mean'][40]:.6f}")
    
    print("\nResults Summary (Fourier Baseline):")
    print(f"Mean MSE at 5 examples: {fourier_metrics['mean'][5]:.6f}")
    print(f"Mean MSE at 10 examples: {fourier_metrics['mean'][10]:.6f}")
    print(f"Mean MSE at 20 examples: {fourier_metrics['mean'][20]:.6f}")
    print(f"Mean MSE at 30 examples: {fourier_metrics['mean'][30]:.6f}")
    print(f"Mean MSE at 40 examples: {fourier_metrics['mean'][40]:.6f}")
    
    return metrics

def plot_comparison(metrics, n_dims, output_file):
    """Plot comparison of baselines"""
    plt.figure(figsize=(10, 6))
    
    # Set up colors and styles
    sns.set_theme('notebook', 'darkgrid')
    palette = sns.color_palette('colorblind')
    
    # Plot each model's performance
    for i, (model_name, model_metrics) in enumerate(metrics.items()):
        display_name = "Sinusoidal Baseline" if "sinusoidal" in model_name else "Fourier Baseline"
        plt.plot(
            list(range(len(model_metrics['mean']))), 
            model_metrics['mean'], 
            label=display_name,
            color=palette[i], 
            linewidth=2
        )
        
        # Add confidence bands
        plt.fill_between(
            list(range(len(model_metrics['bootstrap_low']))),
            model_metrics['bootstrap_low'],
            model_metrics['bootstrap_high'],
            alpha=0.3,
            color=palette[i]
        )
    
    plt.title(f"Baseline Comparison on {n_dims}D Sinusoidal Regression", fontsize=16)
    plt.xlabel("Number of In-context Examples", fontsize=14)
    plt.ylabel("Mean Squared Error", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the specialized SinusoidalRegressionBaseline")
    parser.add_argument("--n_dims", type=int, default=10, 
                        help="Number of dimensions")
    parser.add_argument("--n_points", type=int, default=41, 
                        help="Number of points to evaluate")
    parser.add_argument("--alpha", type=float, default=0.001, 
                        help="Regularization parameter")
    parser.add_argument("--lr", type=float, default=0.05, 
                        help="Learning rate for optimization")
    parser.add_argument("--n_iterations", type=int, default=300, 
                        help="Maximum number of iterations for optimization")
    parser.add_argument("--num_eval", type=int, default=128, 
                        help="Number of evaluation examples")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for evaluation")
    parser.add_argument("--output", type=str, default="sinusoidal_baseline_metrics.json", 
                        help="Output file for saving metrics")
    
    args = parser.parse_args()
    
    evaluate_baselines(
        n_dims=args.n_dims,
        n_points=args.n_points,
        alpha=args.alpha,
        lr=args.lr,
        n_iterations=args.n_iterations,
        num_eval_examples=args.num_eval,
        batch_size=args.batch_size,
        output_file=args.output
    ) 