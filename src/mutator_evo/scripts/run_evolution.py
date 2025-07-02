# src/mutator_evo/scripts/run_evolution.py
import sys
import os
from mutator_evo.core.strategy_mutator import StrategyMutatorV2
from mutator_evo.core.strategy_embedding import StrategyEmbedding

def initialize_mutator() -> StrategyMutatorV2:
    """Initialize mutator with base parameters"""
    return StrategyMutatorV2(
        top_k=8,
        n_mutants=5,
        max_age=20,
        checkpoint_every=5,  # Save checkpoint every 5 epochs
        mutation_probs={
            "add": 0.35,
            "drop": 0.20, 
            "shift": 0.20,
            "invert": 0.10,
            "metaboost": 0.05,
            "crossover": 0.30,
        }
    )

def initialize_feature_bank(mutator: StrategyMutatorV2):
    """Initialize feature bank"""
    mutator.feature_bank = {
        'use_ema', 'use_rsi', 'use_macd',
        'threshold_low', 'threshold_high',
        'window_small', 'window_large',
        'trend_filter', 'volatility_filter'
    }

def run_evolution(mutator: StrategyMutatorV2, epochs: int = 100):
    """Main evolution loop"""
    print("Starting evolutionary process...")
    for epoch in range(1, epochs + 1):
        print(f"\n=== Epoch {epoch}/{epochs} ===")
        try:
            mutator.evolve()
        except Exception as e:
            print(f"Critical error in epoch {epoch}: {str(e)}")
            mutator.strategy_pool = [
                StrategyEmbedding.create_random(mutator.feature_bank)
                for _ in range(20)
            ]

def generate_visualizations():
    """Generate evolution visualizations after completion"""
    try:
        from mutator_evo.scripts.visualize_evolution import plot_evolution
        print("Generating evolution visualizations...")
        
        # Create visualization directory if not exists
        vis_dir = "evolution_visualizations"
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate and save plots
        plot_evolution("checkpoints")
        
        # Move generated files to visualization directory
        for file in ["evolution_progress.png", "evolution_interactive.html", "evolution_data.csv"]:
            if os.path.exists(file):
                os.rename(file, os.path.join(vis_dir, file))
        
        print(f"Visualizations saved in: {vis_dir}/")
    except Exception as e:
        print(f"Failed to generate visualizations: {str(e)}")

def main():
    try:
        # Setup paths
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
        
        # Initialize
        mutator = initialize_mutator()
        initialize_feature_bank(mutator)
        
        # Create initial strategy pool
        print("Creating initial strategy pool...")
        mutator.strategy_pool = [
            StrategyEmbedding.create_random(mutator.feature_bank)
            for _ in range(20)
        ]
        
        # Run evolution
        run_evolution(mutator, epochs=100)
        
        # Show results
        print("\nEvolution completed!")
        if mutator.strategy_pool:
            best = max(mutator.strategy_pool, key=lambda s: s.score())
            print(f"Best strategy: {best}")
            print(f"Features: {list(best.features.keys())}")
            print(f"Score: {best.score():.2f}")
        else:
            print("Strategy pool is empty")
        
        # Generate visualizations
        generate_visualizations()
            
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        # Try to generate visualizations even if error occurred
        try:
            generate_visualizations()
        except:
            pass
        raise

if __name__ == "__main__":
    main()