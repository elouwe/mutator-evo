<p align="left">
  <img src="logo/me.png" width="240" height="110" alt="Mutatorevo Logo" />
</p>

<p align="left">
  <a href="https://github.com/elouwe/mutatorevo"><img src="https://img.shields.io/badge/Mutator_Evo-Official-blueviolet?style=for-the-badge" alt="Official Mutator Evo Project" /></a>
  <a href="https://github.com/elouwe/mutatorevo/pulls"><img src="https://img.shields.io/badge/contributions-welcome-success?style=for-the-badge" alt="Contributions welcome" /></a>
  <a href="https://x.com/akuraalina"><img src="https://img.shields.io/badge/follow_on-x.com-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="X.com" /></a>
</p>

**Mutatorevo** is an experimental Python framework for evolving trading strategies based on their features.  
You provide initial strategies — Mutatorevo mutates, crosses, filters, and returns optimized **mutants** ready for battle on the market. 

> Think of it like natural selection for trading bots 🧠⚔️📈

## ⚙️ Features

1. **Adaptive Genetic Engine**
- Self-tuning mutation system (`DynamicConfig`) with UCB-based operator selection
- 7+ modular genetic operators (crossover, RL mutation, etc.)
- Real-time performance adaptation (impact-driven probability adjustment)

2. **Intelligent Evaluation System**
- Robust backtesting adapter with built-in safeguards
- Multi-metric scoring (Sharpe, drawdown, overfitting penalty)
- Automated train-test split (70/30 IS/OOS validation)

3. **RL-Ready Architecture**
- Neural network mutation operators (layer size, hyperparameters)
- Integrated state-action space for RL agents
- Compatible with future DRL integrations

4. **Enterprise-Grade Features**
- Automated checkpointing (emergency saves, version control)
- SHAP-based feature importance analysis
- Interactive visualization suite (Plotly, Matplotlib)
- Structured logging (Sentry-ready)

## ✅ Completed Tasks

- [x] Step 8: Exploration/exploitation balance:
    * UCB algorithm for mutation selection
    * Auto-adjust mutation_probs
- [x] Step 9: Distributed computing
- [x] Step 10: Memory optimization:
* Strategy compression (protobuf).
* LRU-cache for backtests
- [ ] Step 11: Benchmarks:
    * Performance tests on 10K strategies
- [ ] Step 12: Monitoring system:
- Prometheus metrics (generations, fitness)
- Alert on degradation (Slack/webhooks)

**What i Did:**  
1. **Squeezed Data Like a Packed Suitcase**  
   - Before: 76KB of data  
   - After: 31KB (2.4x smaller!)  
   - Test: Unpacked it perfectly—nothing lost!  

2. **Added a "Cheat Sheet" for Faster Answers**  
   - First time solving a problem: 5 minutes  
   - Next time: 1 second (just checks the cheat sheet)  
   - Now it remembers everything it learns  

3. **Made It Work Like a Super-Team**  
   - 1 computer: 10 tasks/hour  
   - 4 computers: 40 tasks/hour (teamwork!)  
   - Tested: Handles 4x more work without breaking  

**How We Tested It:**  
- Crunched data → squeezed it → unpacked → compared: 100% match  
- Asked the same question repeatedly: Answers instantly after first time  
- Gave it more and more work: No crashes, just gets faster  

**Real-Life Benefits:**  
- 🚀 **Speed**: Now 2-10x faster  
- 📦 **Space-Saving**: Uses 2.4x less memory  
- 💪 **Power**: Handles 4x heavier workloads  
- 🔁 **Reliability**: Works perfectly even after 357 tests  

**Simple Example:**  
Imagine moving houses:  
- **Old Way**: You carry boxes alone (10 hours)  
- **New Way**:  
  - Packs boxes tighter (compression)  
  - Labels everything (cache)  
  - Friends help (distributed computing)  
  - Result: Done in 2 hours!  

**Bottom Line**: It’s like upgrading from a bicycle to a rocketship—faster, stronger, and never breaks! 🚀✨ 

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/elouwe/mutator-evo.git
cd mutator-evo
echo "from setuptools import setup, find_packages
setup(
    name='mutator_evo',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
)" > setup.py

pip install -r requirements.txt
pip install -e .
```

## 🧪 Running the Evolution

Start the evolution process:

```bash
python -m mutator_evo.scripts.run_evolution
```

You can modify the config file in `configs/` to set:
- The initial strategies
- Mutation parameters
- Evaluation metrics
- Checkpointing and logging behavior

## 🧪 Benchmark Testing

To run performance and correctness tests for the system:

```bash
python -m pytest src/tests/benchmark -v
```

## Example Output

After running the full evolution cycle, you’ll see a detailed log like this:

```
Evolution completed successfully!
2025-07-04 01:08:48,366 - __main__ - INFO - Best strategy: mut_5bd2a890
2025-07-04 01:08:48,366 - __main__ - INFO - Features: ['ema_period', 'bollinger_period', 'stop_loss', 'use_stoch', 'macd_slow', 'rsi_period', 'rl_agent', 'stoch_k', 'stoch_d', 'use_rsi', 'sma_period', 'take_profit', 'use_sma', 'trade_size', 'adx_period']
2025-07-04 01:08:48,366 - __main__ - INFO - Score: 10.00
2025-07-04 01:08:48,366 - __main__ - INFO - OOS Sharpe: 23.97
2025-07-04 01:08:48,366 - __main__ - INFO - OOS Drawdown: 8.15%
2025-07-04 01:08:48,366 - __main__ - INFO - OOS Win Rate: 0.39
2025-07-04 01:08:48,366 - __main__ - INFO - Trade count: 359
2025-07-04 01:08:48,515 - __main__ - INFO - Generating visualizations...
Found 6 checkpoint files
Processed checkpoint mutator_checkpoint_20250704-005954.pkl (gen 1)
Processed checkpoint mutator_checkpoint_20250704-010057.pkl (gen 2)
Processed checkpoint mutator_checkpoint_20250704-010209.pkl (gen 3)
Processed checkpoint mutator_checkpoint_20250704-010333.pkl (gen 4)
Processed checkpoint mutator_checkpoint_20250704-010514.pkl (gen 5)
Processed checkpoint mutator_checkpoint_20250704-010715.pkl (gen 6)
Saved operator impact rates plot: data/operator_impact_rates.png
Saved main evolution plot: data/evolution_progress.png
Saved interactive plot: data/evolution_interactive.html
Saved evolution data: data/evolution_data.csv
2025-07-04 01:08:49,945 - __main__ - INFO - Visualizations generated
2025-07-04 01:08:49,946 - __main__ - INFO - All operations completed successfully!
```

### 🧠 What it means:

* **Best strategy** — The top-performing strategy from the final generation.
* **Features** — The active features (indicators, parameters) selected during evolution.
* **Score** — The internal fitness score for selection.
* **OOS Sharpe** — Sharpe ratio on out-of-sample data.
* **OOS Drawdown** — Maximum drawdown on OOS.
* **Win Rate** — Percent of profitable trades on OOS data.
* **Trade count** — Total number of trades simulated.
* **Checkpoints** — Strategies saved at each evolutionary epoch.
* **Visualizations** — Summary plots and interactive dashboards for review:

  * `evolution_progress.png` — Fitness over time
  * `operator_impact_rates.png` — Contribution of each mutation operator
  * `evolution_interactive.html` — Interactive Plotly dashboard
  * `evolution_data.csv` — Raw data behind the evolution

> You can now reuse the top strategy, analyze its trades, or re-evolve it with new constraints.


## 📁 Project Structure Overview

This is a modular, scalable architecture for **Mutator Evo** — a framework for evolving, testing, and deploying algorithmic trading strategies.

```
.
├── checkpoints/
├── configs/
├── data/
├── docs/
├── logo/
├── mutator_evo.egg-info/
├── src/
│   └── mutator_evo/
│       ├── backtest/
│       ├── core/
│       └── ray_pool.py
│       ├── operators/   
│       └── scripts/
├── tests/
│   ├── benchmark/
│   ├── integration/
│   └── unit/
├── .coverage
├── pyproject.toml
├── README.md
├── requirements.txt
└── setup.py
```

### 🧠 Core Logic — `src/mutator_evo/`

#### `backtest/`

Modules responsible for backtesting and execution logic:

* `backtrader_adapter.py` — Standard backtesting via Backtrader.
* `risk_manager.py` — SL/TP logic and position sizing.
* `vectorized_backtester.py` — Fast, Pandas/Numpy-based backtester.

#### `core/`

Foundational logic and infrastructure:

* `config.py` — Global configuration loader.
* `logger.py` — Custom logging setup.
* `strategy_embedding.py` — Encodes strategies as feature vectors.
* `strategy_mutator.py` — Handles mutation, crossover, and strategy evolution.
* `ray_pool.py` implements a thread-safe Ray pool for distributed computing with automatic resource management and nested call handling.

### 🔁 Operators — `operators/`

#### Standard Modules:

* `importance_calculator.py` — Scores feature importance.
* `mutation_impl.py` — Basic mutation implementations.
* `interfaces.py` — Abstract base classes and mutation APIs.

#### 🧪 Advanced Modules:

* `_uniform_crossover.py` — Uniform crossover between parent strategies.
* `rl_mutation.py` — Reinforcement learning-based mutation selector.
* `shap_importance.py` — Feature importance via SHAP values.
  
### 🚀 Scripts — `scripts/`

Ready-to-run utilities:

* `run_evolution.py` — Main evolutionary loop.
* `deploy_strategy.py` — Exports top strategies for deployment.
* `visualize_evolution.py` — Plots fitness progress and stats.
* `visualize_results.py` — OOS performance visualizations and plots.

#### `benchmark/`

Performance tests for critical system components:

* `test_cache_performance.py` — Validates backtest caching efficiency (hit rate ≥50%)
* `test_ray_performance.py` — Measures distributed computing speedup with Ray
* `test_compression.py` — Tests strategy data compression ratio (>1.5x) and integrity

### 🧩 Other Directories

* `checkpoints/` — Serialized `.pkl` strategy checkpoints per epoch.
* `configs/` — YAML configuration files by market type.
* `data/` — Exported CSVs, plots, and visualizations.
* `docs/` — Markdown documentation for architecture, APIs, and mutation logic.
* `logo/` — Branding assets like `me.png`, logos, etc.
* `tests/` — Full unit + integration test suite.

### ⚙️ Build & Environment

* `pyproject.toml` — Modern build config (PEP 518).
* `setup.py` — Legacy setup script for editable installs.
* `requirements.txt` — Python dependencies.
* `.coverage` — Code coverage report.
* `README.md` — Project overview, usage guide, and example outputs.
  
## 🤝 Contributing

Mutatorevo is still in the experimental phase.  
If you want to add your own mutation operator, evaluation filter, or visualization — PRs are welcome!

## 🔒 License

This project is **source-available** under the following conditions:

- You may **view, clone, and modify** the code for non-commercial purposes.
- You may **not use** this code in any **production**, **commercial**, or **hosted** product.
- All rights are reserved by the original author (© Alina Akura).
- For collaboration or licensing — contact [x.com/akuraalina](https://x.com/akuraalina).

Violators will be publicly shamed and digitally hexed.
