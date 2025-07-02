# src/mutator_evo/scripts/visualize_evolution.py
import pathlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots
from mutator_evo.core.strategy_mutator import StrategyMutatorV2

def plot_evolution(checkpoint_dir: str):
    """Generate visualizations of evolutionary progress"""
    # Collect data from checkpoints
    generations = []
    best_scores = []
    avg_scores = []
    feature_counts = []
    mutation_rates = []
    top_features = []
    
    # Sort checkpoints by creation time
    checkpoint_files = sorted(
        pathlib.Path(checkpoint_dir).glob("mutator_checkpoint_*.pkl"),
        key=lambda f: f.stat().st_ctime
    )
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Process each checkpoint
    for i, checkpoint in enumerate(checkpoint_files):
        try:
            # Load checkpoint
            mutator = StrategyMutatorV2.load_checkpoint(str(checkpoint))
            
            # Skip if no strategies
            if not mutator.strategy_pool:
                print(f"  Checkpoint {i}: strategy pool is empty")
                continue
                
            # Calculate scores
            scores = [s.score() for s in mutator.strategy_pool]
            best_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            generations.append(i)
            best_scores.append(best_score)
            avg_scores.append(avg_score)
            feature_counts.append(len(mutator.strategy_pool[0].features))
            
            # Get mutation rates
            mutation_rates.append({
                'add': mutator.config.mutation_probs.get('add', 0),
                'drop': mutator.config.mutation_probs.get('drop', 0),
                'shift': mutator.config.mutation_probs.get('shift', 0),
                'invert': mutator.config.mutation_probs.get('invert', 0),
                'metaboost': mutator.config.mutation_probs.get('metaboost', 0)
            })
            
            # Get top features
            if hasattr(mutator, 'importance') and mutator.importance:
                sorted_features = sorted(
                    mutator.importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                top_features.append(sorted_features[:5])
            else:
                top_features.append([])
                
            print(f"  Checkpoint {i}: best_score={best_score:.2f}, features={feature_counts[-1]}")
            
        except Exception as e:
            print(f"Error loading checkpoint {checkpoint}: {str(e)}")
    
    # Check if we have any data
    if not generations:
        print("No valid checkpoints found. Cannot generate visualizations.")
        return None
    
    # Create DataFrame for analysis
    evolution_df = pd.DataFrame({
        'generation': generations,
        'best_score': best_scores,
        'avg_score': avg_scores,
        'feature_count': feature_counts,
        'mutation_rates': mutation_rates,
        'top_features': top_features
    })
    
    # Static Matplotlib plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Scores evolution
    plt.subplot(2, 2, 1)
    plt.plot(evolution_df['generation'], evolution_df['best_score'], 'g-', label="Best score")
    plt.plot(evolution_df['generation'], evolution_df['avg_score'], 'b--', label="Average score")
    plt.title("Strategy Scores Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Complexity evolution
    plt.subplot(2, 2, 2)
    plt.plot(evolution_df['generation'], evolution_df['feature_count'], 'r-')
    plt.title("Strategy Complexity Evolution")
    plt.xlabel("Generation")
    plt.ylabel("Feature count")
    plt.grid(True)
    
    # Plot 3: Mutation probabilities
    plt.subplot(2, 2, 3)
    mutation_types = ['add', 'drop', 'shift', 'invert', 'metaboost']
    for mtype in mutation_types:
        rates = [r[mtype] for r in evolution_df['mutation_rates']]
        plt.plot(evolution_df['generation'], rates, label=mtype)
    plt.title("Mutation Probabilities Dynamics")
    plt.xlabel("Generation")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Top features of last generation
    plt.subplot(2, 2, 4)
    if not evolution_df.empty:
        last_gen_features = evolution_df.iloc[-1]['top_features']
        if last_gen_features:
            names, importances = zip(*last_gen_features)
            plt.bar(names, importances, color='purple')
            plt.title(f"Top-5 Features (Gen {len(generations)})")
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("evolution_progress.png")
    print("Saved static plot: evolution_progress.png")
    
    # Interactive Plotly visualization
    fig = make_subplots(
        rows=2, 
        cols=2,
        subplot_titles=(
            "Scores Evolution", 
            "Complexity Evolution",
            "Mutation Probabilities",
            f"Top Features (Gen {len(generations)})"
        ),
        specs=[[{}, {}], [{}, {}]]
    )
    
    # Scores plot
    fig.add_trace(
        go.Scatter(
            x=evolution_df['generation'], 
            y=evolution_df['best_score'], 
            name='Best score',
            line=dict(color='green')
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=evolution_df['generation'], 
            y=evolution_df['avg_score'], 
            name='Average score',
            line=dict(color='blue', dash='dash')
        ),
        row=1, col=1
    )
    
    # Complexity plot
    fig.add_trace(
        go.Scatter(
            x=evolution_df['generation'], 
            y=evolution_df['feature_count'], 
            name='Feature count',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # Mutation probabilities
    for mtype in mutation_types:
        rates = [r[mtype] for r in evolution_df['mutation_rates']]
        fig.add_trace(
            go.Scatter(
                x=evolution_df['generation'], 
                y=rates, 
                name=mtype,
                mode='lines'
            ),
            row=2, col=1
        )
    
    # Top features
    if not evolution_df.empty and evolution_df.iloc[-1]['top_features']:
        names, importances = zip(*evolution_df.iloc[-1]['top_features'])
        fig.add_trace(
            go.Bar(
                x=names, 
                y=importances,
                name='Importance',
                marker_color='purple'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        title="Strategy Evolutionary Progress",
        height=800,
        showlegend=True
    )
    
    fig.write_html("evolution_interactive.html")
    print("Saved interactive plot: evolution_interactive.html")
    
    # Save data to CSV
    evolution_df.to_csv("evolution_data.csv", index=False)
    print("Saved evolution data: evolution_data.csv")
    
    return evolution_df