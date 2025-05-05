import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style
sns.set_theme('notebook', 'darkgrid')
palette = sns.color_palette('colorblind')

# Path to your evaluation results
metrics_file = "FINAL_rff_fixed_evaluation_results.json"

def basic_plot(metrics, models=None, trivial=1.0, max_y=None, log_scale=False):
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
            display_name = "Transformer"
        elif "kernel_ridge_fixed" in name:
            display_name = "Kernel Ridge"
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
    elif max_y is not None:
        ax.set_ylim(0, max_y)
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Place legend outside the plot
    legend = ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=12)
    for line in legend.get_lines():
        line.set_linewidth(3)

    plt.tight_layout()
    return fig, ax

def plot_results():
    """Generate and save plots for the RFF fixed task"""
    # Read metrics directly from JSON file
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Plot all models
    all_metrics = metrics["standard"]
    
    # Create standard plot
    fig, ax = basic_plot(all_metrics, max_y=3.0)
    ax.set_title(f"Performance on RFF Fixed Kernel Regression", fontsize=16)
    plt.savefig("rff_fixed_results.png", dpi=300, bbox_inches='tight')
    
    # Create zoomed-in plot to better see the best performers
    fig2, ax2 = basic_plot(all_metrics, max_y=0.5)
    ax2.set_title(f"Performance on RFF Fixed Kernel Regression (Zoomed)", fontsize=16)
    plt.savefig("rff_fixed_results_zoomed.png", dpi=300, bbox_inches='tight')
    
    # Create log-scale plot to see all models despite large range differences
    fig3, ax3 = basic_plot(all_metrics, log_scale=True)
    ax3.set_title(f"Performance on RFF Fixed Kernel Regression (Log Scale)", fontsize=16)
    plt.savefig("rff_fixed_results_log.png", dpi=300, bbox_inches='tight')
    
    # Look at which models perform best overall
    model_final_errors = {}
    for model_name, model_data in all_metrics.items():
        # Use average of last 10 points for stability
        model_final_errors[model_name] = np.mean(model_data["mean"][-10:])
    
    # Sort models by final error
    sorted_models = sorted(model_final_errors.items(), key=lambda x: x[1])
    top_models = [model[0] for model in sorted_models[:4]]  # Get top 4 models
    
    print("Models by final error (best to worst):")
    for model, error in sorted_models:
        print(f"{model}: {error:.6f}")
    
    # Plot showing just the best performing models
    fig4, ax4 = basic_plot(all_metrics, models=top_models, max_y=None)
    ax4.set_title(f"Top Models on RFF Fixed Kernel Regression", fontsize=16)
    plt.savefig("rff_fixed_top_models.png", dpi=300, bbox_inches='tight')
    
    # Also make a super-zoomed plot to highlight kernel ridge performance
    if "kernel_ridge_fixed" in all_metrics:
        fig5, ax5 = basic_plot(all_metrics, models=["kernel_ridge_fixed"], max_y=1.0)
        ax5.set_title(f"Kernel Ridge Performance on RFF Fixed Kernel Regression", fontsize=16)
        plt.savefig("rff_fixed_kernel_ridge.png", dpi=300, bbox_inches='tight')
        
        # Log-scale plot for kernel ridge to see convergence
        fig6, ax6 = basic_plot(all_metrics, models=["kernel_ridge_fixed"], log_scale=True)
        ax6.set_title(f"Kernel Ridge Performance (Log Scale)", fontsize=16)
        plt.savefig("rff_fixed_kernel_ridge_log.png", dpi=300, bbox_inches='tight')
    
    print(f"\nGenerated plots for RFF Fixed Kernel Regression:")
    print(f"1. rff_fixed_results.png - Standard plot with all models")
    print(f"2. rff_fixed_results_zoomed.png - Zoomed in to see best models clearly")
    print(f"3. rff_fixed_results_log.png - Log scale to compare all models") 
    print(f"4. rff_fixed_top_models.png - Only the top performing models")
    if "kernel_ridge_fixed" in all_metrics:
        print(f"5. rff_fixed_kernel_ridge.png - Just the kernel ridge model")
        print(f"6. rff_fixed_kernel_ridge_log.png - Kernel ridge model with log scale")

if __name__ == "__main__":
    plot_results() 