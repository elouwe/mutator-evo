# src/mutator_evo/scripts/visualize_evolution.py
import pathlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import dill
import traceback
import logging

logger = logging.getLogger(__name__)

def plot_operator_impact_rates(evolution_df, data_dir):
    """Visualize operator impact rates with usage counts"""
    if evolution_df.empty or 'operator_stats' not in evolution_df.columns:
        print("No operator stats data available")
        return

    # Collect all operators
    all_operators = set()
    for stats in evolution_df['operator_stats']:
        if isinstance(stats, dict):
            all_operators.update(stats.keys())
    
    if not all_operators:
        print("No operators found in stats")
        return

    # Create figure
    n_operators = len(all_operators)
    n_cols = 3
    n_rows = (n_operators + n_cols - 1) // n_cols
    plt.figure(figsize=(15, 5 * n_rows))
    
    # Plot each operator
    for i, op in enumerate(sorted(all_operators)):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Prepare data
        avg_impacts = []
        usage_counts = []
        for _, row in evolution_df.iterrows():
            stats = row['operator_stats']
            if op in stats and stats[op]['n'] > 0:
                # Calculate average impact
                avg_impacts.append(stats[op]['total_impact'] / stats[op]['n'])
                usage_counts.append(stats[op]['n'])
            else:
                avg_impacts.append(0)
                usage_counts.append(0)
        
        # Plot average impact
        plt.plot(evolution_df['generation'], avg_impacts, 'b-', label='Avg Impact')
        plt.title(f"{op} (Total Uses: {sum(usage_counts)})")
        plt.xlabel("Generation")
        plt.ylabel("Avg Impact")
        plt.grid(True)
        
        # Plot usage count on secondary axis
        ax2 = plt.gca().twinx()
        ax2.plot(evolution_df['generation'], usage_counts, 'r--', alpha=0.7, label='Usage Count')
        ax2.set_ylabel('Usage Count')
        ax2.set_ylim(0, max(usage_counts) * 1.1 if usage_counts else 5)
        
        # Combine legends
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='best')

    plt.tight_layout()
    plot_path = data_dir / "operator_impact_rates.png"
    plt.savefig(plot_path, dpi=120)
    print(f"Saved operator impact rates plot: {plot_path}")
    plt.close()

def plot_evolution(checkpoint_dir: str):
    """Generate visualizations of evolutionary progress"""
    # Create data directory if not exists
    data_dir = pathlib.Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize data containers
    generations = []
    best_scores = []
    avg_scores = []
    feature_counts = []
    mutation_rates = []
    top_features = []
    oos_scores = []
    oos_drawdowns = []
    oos_win_rates = []
    penalties = []
    sharpe_diffs = []
    trade_counts = []
    rl_usage = []
    rl_layer_counts = []
    operator_stats = []
    
    # Find checkpoint files
    checkpoint_files = sorted(
        pathlib.Path(checkpoint_dir).glob("mutator_checkpoint_*.pkl"),
        key=lambda f: f.stat().st_mtime
    )
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Process each checkpoint
    valid_checkpoints = 0
    for checkpoint in checkpoint_files:
        try:
            with open(checkpoint, 'rb') as f:
                state = dill.load(f)
            
            # Skip if no strategy pool
            if "pool" not in state or not state["pool"]:
                print(f"Skipping {checkpoint.name}: empty strategy pool")
                continue
                
            # Decompress strategies for analysis
            for s in state["pool"]:
                if hasattr(s, 'decompress'):
                    s.decompress()
            
            pool = state["pool"]
            config_params = state.get("config_params", {})
            importance = state.get("importance", {})
            op_stats = state.get("operator_stats", {})
            
            # Collect basic metrics
            scores = [s.score() for s in pool]
            best_score = max(scores) if scores else 0
            avg_score = sum(scores)/len(scores) if scores else 0
            
            generations.append(valid_checkpoints)
            best_scores.append(best_score)
            avg_scores.append(avg_score)
            feature_counts.append(len(pool[0].features) if pool else 0)
            operator_stats.append(op_stats)
            
            # Mutation probabilities
            mutation_rates.append(config_params.get("mutation_probs", {}))
            
            # Feature importance
            if importance and isinstance(importance, dict):
                top_features.append(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            else:
                top_features.append([])
            
            # Best strategy metrics
            best = max(pool, key=lambda s: s.score()) if pool else None
            if best and best.oos_metrics:
                oos_scores.append(best.oos_metrics.get('oos_sharpe', 0))
                oos_drawdowns.append(best.oos_metrics.get('oos_max_drawdown', 0))
                oos_win_rates.append(best.oos_metrics.get('oos_win_rate', 0))
                penalties.append(best.oos_metrics.get('overfitting_penalty', 0))
                sharpe_diffs.append(best.oos_metrics.get('is_sharpe', 0) - best.oos_metrics.get('oos_sharpe', 0))
                trade_counts.append(best.oos_metrics.get('trade_count', 0))
            else:
                oos_scores.append(0)
                oos_drawdowns.append(0)
                oos_win_rates.append(0)
                penalties.append(0)
                sharpe_diffs.append(0)
                trade_counts.append(0)
            
            # RL agent stats
            rl_count = sum(1 for s in pool if hasattr(s, 'features') and "rl_agent" in s.features) if pool else 0
            rl_usage.append(rl_count / len(pool) if pool else 0)
            
            layer_sum = 0
            for s in pool:
                if hasattr(s, 'features') and "rl_agent" in s.features:
                    agent = s.features["rl_agent"]
                    if isinstance(agent, dict) and "hidden_layers" in agent:
                        layers = agent["hidden_layers"]
                        if isinstance(layers, list):
                            layer_sum += len(layers)
            
            rl_layer_counts.append(layer_sum / rl_count if rl_count > 0 else 0)
            
            valid_checkpoints += 1
            print(f"Processed checkpoint {checkpoint.name} (gen {valid_checkpoints})")
            
            # Recompress strategies to save memory
            for s in pool:
                if hasattr(s, 'compress'):
                    s.compress()
                    
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint}: {str(e)}")
            traceback.print_exc()
    
    if not generations:
        print("No valid checkpoints found. Cannot generate visualizations.")
        return None
    
    # Create evolution dataframe
    evolution_df = pd.DataFrame({
        'generation': generations,
        'best_score': best_scores,
        'avg_score': avg_scores,
        'feature_count': feature_counts,
        'mutation_rates': mutation_rates,
        'top_features': top_features,
        'operator_stats': operator_stats,
        'oos_score': oos_scores,
        'oos_drawdown': oos_drawdowns,
        'oos_win_rate': oos_win_rates,
        'penalty': penalties,
        'sharpe_diff': sharpe_diffs,
        'trade_count': trade_counts,
        'rl_usage': rl_usage,
        'rl_avg_layers': rl_layer_counts
    })
    
    # Generate operator impact rates plot
    plot_operator_impact_rates(evolution_df, data_dir)
    
    # Main evolution plot
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Scores
    plt.subplot(3, 2, 1)
    plt.plot(evolution_df['generation'], evolution_df['best_score'], 'g-', label="Best score")
    plt.plot(evolution_df['generation'], evolution_df['oos_score'], 'b-', label="OOS Sharpe")
    plt.title("Strategy Scores Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Feature count
    plt.subplot(3, 2, 2)
    plt.plot(evolution_df['generation'], evolution_df['feature_count'], 'r-')
    plt.title("Strategy Complexity Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Feature count")
    plt.grid(True)
    
    # Plot 3: Mutation probabilities
    plt.subplot(3, 2, 3)
    mutation_types = ['add', 'drop', 'shift', 'invert', 'metaboost', 'crossover', 'uniform_crossover', 'rl_mutation']
    for mtype in mutation_types:
        rates = [r.get(mtype, 0) for r in evolution_df['mutation_rates']]
        plt.plot(evolution_df['generation'], rates, label=mtype)
    plt.title("Mutation Probabilities Dynamics")
    plt.xlabel("Generation")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Overfitting metrics
    plt.subplot(3, 2, 4)
    plt.plot(evolution_df['generation'], evolution_df['penalty'], 'k-', label="Penalty")
    plt.plot(evolution_df['generation'], evolution_df['sharpe_diff'], 'm--', label="Sharpe Diff (IS-OOS)")
    plt.title("Overfitting Metrics")
    plt.xlabel("Generation")
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Trade activity
    plt.subplot(3, 2, 5)
    plt.bar(evolution_df['generation'], evolution_df['trade_count'], color='purple')
    plt.title("Trade Activity")
    plt.xlabel("Generation")
    plt.ylabel("Trade Count")
    plt.grid(True)
    
    # Plot 6: RL agent stats
    plt.subplot(3, 2, 6)
    plt.plot(evolution_df['generation'], evolution_df['rl_usage'], 'b-', label="RL Usage")
    plt.plot(evolution_df['generation'], evolution_df['rl_avg_layers'], 'r-', label="Avg RL Layers")
    plt.title("RL Agent Evolution")
    plt.xlabel("Generation")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    main_plot_path = data_dir / "evolution_progress.png"
    plt.savefig(main_plot_path, dpi=100)
    print(f"Saved main evolution plot: {main_plot_path}")
    plt.close()
    
    # Interactive plot with Plotly
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "Scores Evolution", 
            "Complexity Evolution",
            "Mutation Probabilities", 
            f"Top Features (Gen {generations[-1]})",
            "Overfitting Penalty",
            "Sharpe Ratio Difference",
            "Trade Activity",
            "RL Agent Evolution"
        )
    )
    
    # Add traces
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['best_score'], name="Best Score"),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['oos_score'], name="OOS Sharpe"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['feature_count'], name="Feature Count"),
        row=1, col=2
    )
    
    for mtype in mutation_types:
        fig.add_trace(
            go.Scatter(
                x=evolution_df['generation'],
                y=[r.get(mtype, 0) for r in evolution_df['mutation_rates']],
                name=mtype,
                mode='lines'
            ),
            row=2, col=1
        )
    
    if evolution_df.iloc[-1]['top_features']:
        fig.add_trace(
            go.Bar(
                x=[f[0] for f in evolution_df.iloc[-1]['top_features']],
                y=[f[1] for f in evolution_df.iloc[-1]['top_features']],
                name="Feature Importance"
            ),
            row=2, col=2
        )
    
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['penalty'], name="Penalty"),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=evolution_df['generation'],
            y=evolution_df['sharpe_diff'],
            name="Sharpe Diff",
            marker_color=np.where(np.array(evolution_df['sharpe_diff']) > 0, 'red', 'green')
        ),
        row=3, col=2
    )
    
    fig.add_trace(
        go.Bar(x=evolution_df['generation'], y=evolution_df['trade_count'], name="Trade Count"),
        row=4, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['rl_usage'], name="RL Usage"),
        row=4, col=2
    )
    fig.add_trace(
        go.Scatter(x=evolution_df['generation'], y=evolution_df['rl_avg_layers'], name="Avg RL Layers"),
        row=4, col=2
    )
    
    fig.update_layout(
        height=1500,
        title_text="Evolutionary Strategy Optimization Dashboard",
        showlegend=True
    )
    
    interactive_path = data_dir / "evolution_interactive.html"
    fig.write_html(str(interactive_path))
    print(f"Saved interactive plot: {interactive_path}")
    
    # Save data to CSV
    csv_path = data_dir / "evolution_data.csv"
    evolution_df.to_csv(csv_path, index=False)
    print(f"Saved evolution data: {csv_path}")
    
    return evolution_df