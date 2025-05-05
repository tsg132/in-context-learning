import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

# Path to your evaluation results
run_dir = "temp_evals2"
task = "sinusoidal_regression_5d"
run_id = "37d6c805-d6d8-4096-86c1-ea8fbd86fbb1"  # Your specific run ID

def basic_plot(metrics, models=None, trivial=1.0, max_y=None, min_y=None, log_scale=False):
    """Modified version of the basic_plot function from plot_utils.py"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if models is not None:
        metrics = {k: metrics[k] for k in models if k in metrics}

    color = 0
    ax.axhline(trivial, ls="--", color="gray", label="Baseline")
    
    # Sort models by performance (average of last 10 points)
    def avg_last_10(model_metrics):
        return np.mean(model_metrics["mean"][-10:])
    
    sorted_metrics = dict(sorted(metrics.items(), key=lambda x: avg_last_10(x[1])))
    
    for name, vs in sorted_metrics.items():
        # Clean up model name for display
        display_name = name
        if "gpt2" in name:
            if "untrained" in name:
                display_name = "Transformer (Untrained)"
            else:
                display_name = "Transformer (Trained)"
        elif "fourier_fit" in name:
            display_name = "Fourier Fit"
        elif "NN_n=" in name:
            display_name = f"{name.split('_')[1].split('=')[1]}-Nearest Neighbors"
        elif "decision_tree" in name:
            if "None" in name:
                display_name = "Decision Tree (Unlimited)"
            else:
                depth = name.split("=")[1]
                display_name = f"Decision Tree (Depth={depth})"
        elif "averaging" in name:
            display_name = "Averaging"
        elif "xgboost" in name:
            display_name = "XGBoost"
            
        ax.plot(vs["mean"], "-", label=display_name, color=palette[color % 10], lw=2)
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3, color=palette[color % 10])
        color += 1
        
    ax.set_xlabel("Number of In-context Examples", fontsize=14)
    ax.set_ylabel("Mean Squared Error", fontsize=14)
    ax.set_xlim(-1, len(low))
    
    if log_scale:
        ax.set_yscale('log')
    else:
        # Set y limits if specified
        if min_y is not None and max_y is not None:
            ax.set_ylim(min_y, max_y)
        elif max_y is not None:
            ax.set_ylim(0, max_y)
        elif min_y is not None:
            # Get the current maximum y value
            current_ylim = ax.get_ylim()
            ax.set_ylim(min_y, current_ylim[1])
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend outside the plot
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    for line in legend.get_lines():
        line.set_linewidth(3)

    plt.tight_layout()
    return fig, ax

def plot_results():
    """Generate and save plots for the sinusoidal regression task"""
    # Read metrics from JSON file
    metrics_path = os.path.join(run_dir, task, run_id, "metrics.json")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Plot all models
    all_metrics = metrics["standard"]
    
    # Create standard plot with better limits
    fig, ax = basic_plot(all_metrics, max_y=1.5)
    ax.set_title(f"Performance on {task.replace('_', ' ').title()}", fontsize=16)
    plt.savefig(f"{task}_results.png", dpi=300, bbox_inches='tight')
    
    # Create a more appropriate zoomed-in plot focused on the transformer's performance range
    fig2, ax2 = basic_plot(all_metrics, min_y=0.4, max_y=0.6)
    ax2.set_title(f"Performance on {task.replace('_', ' ').title()} (Zoomed)", fontsize=16)
    plt.savefig(f"{task}_results_zoomed.png", dpi=300, bbox_inches='tight')
    
    # Create log-scale plot to see all models despite large range differences
    fig3, ax3 = basic_plot(all_metrics, log_scale=True)
    ax3.set_title(f"Performance on {task.replace('_', ' ').title()} (Log Scale)", fontsize=16)
    plt.savefig(f"{task}_results_log.png", dpi=300, bbox_inches='tight')
    
    # Plot showing just the best performing models with appropriate limits
    best_models = [
        "gpt2_embd=256_layer=12_head=8",  # Trained Transformer
        "gpt2_embd=256_layer=12_head=8_untrained",  # Untrained Transformer
        "xgboost",
        "NN_n=3_uniform"
    ]
    
    fig4, ax4 = basic_plot(all_metrics, models=best_models, min_y=0.4, max_y=1.1)
    ax4.set_title(f"Top Models on {task.replace('_', ' ').title()}", fontsize=16)
    plt.savefig(f"{task}_top_models.png", dpi=300, bbox_inches='tight')
    
    # Add a new plot comparing trained vs untrained transformer
    transformer_models = [
        "gpt2_embd=256_layer=12_head=8",  # Trained Transformer
        "gpt2_embd=256_layer=12_head=8_untrained",  # Untrained Transformer
    ]
    
    fig5, ax5 = basic_plot(all_metrics, models=transformer_models, min_y=0.0, max_y=1.2)
    ax5.set_title(f"Trained vs Untrained Transformer on {task.replace('_', ' ').title()}", fontsize=16)
    plt.savefig(f"{task}_trained_vs_untrained.png", dpi=300, bbox_inches='tight')
    
    # Add a plot comparing trained transformer vs XGBoost
    comparison_models = [
        "gpt2_embd=256_layer=12_head=8",  # Trained Transformer
        "xgboost"
    ]
    
    fig6, ax6 = basic_plot(all_metrics, models=comparison_models, min_y=0.0, max_y=1.2)
    ax6.set_title(f"Transformer vs XGBoost on {task.replace('_', ' ').title()}", fontsize=16)
    plt.savefig(f"{task}_transformer_vs_xgboost.png", dpi=300, bbox_inches='tight')
    
    print(f"Generated plots for {task}:")
    print(f"1. {task}_results.png - Standard plot with all models")
    print(f"2. {task}_results_zoomed.png - Zoomed in to see transformer performance")
    print(f"3. {task}_results_log.png - Log scale to compare all models") 
    print(f"4. {task}_top_models.png - Only the top performing models including both transformers")
    print(f"5. {task}_trained_vs_untrained.png - Direct comparison of trained vs untrained transformer")
    print(f"6. {task}_transformer_vs_xgboost.png - Direct comparison of transformer and XGBoost")

if __name__ == "__main__":
    plot_results() 