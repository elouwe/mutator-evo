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

- Plug-and-play interface for strategy mutation & evaluation
- Auto-tuning mutation system (`DynamicConfig`)
- Modular and scalable architecture
- Checkpointing and reproducibility
- Ready for future integrations: logging, metrics, visual dashboards

## ✅ Completed Tasks

- [x] Step 1: StrategyMutatorV2 Refactor
- [x] Step 2: Structlog + Sentry for tracing errors
- [x] Step 3: Visualization of evolutionary progress
- [ ] Step 4: Adapter for Backtrader
- [ ] Step 5: Validation on OOS data
- [ ] Step 6: Tests on historical crises

**Logging & Monitoring Implementation (Structlog + Sentry + Visualizations)**  

### **1. Logging System**  
- **Structlog**  
  - Structured JSON logging for all events (strategy creation, mutations, errors)  
  - Automatic metadata: timestamps, log levels (INFO/ERROR), context (strategy names, scores)  
  - Human-readable console output during development  

- **Sentry**  
  - Real-time error tracking with crash reports  
  - Alerts developers via email/Slack on critical failures  
  - Captures full error context (stack traces, variable states)  

### **2. Evolution Progress Tracking**  
- **Data Collected Each Generation**:  
  - Best/average strategy scores  
  - Number of features in top strategies  
  - Mutation probabilities (add/drop/shift rates)  
  - Top 5 most important features  

- **Visualizations**:  
  - **Static Plots (Matplotlib)**: PNG images showing score trends and complexity growth  
  - **Interactive Dashboards (Plotly)**: HTML files with zoomable charts and tooltips  
  - **CSV Export**: Raw data for further analysis  

### **Key Benefits**  
1. **Transparency**: See exactly how strategies evolve  
2. **Debugging**: Instant error alerts with Sentry + detailed logs  
3. **Optimization**: Adjust mutation parameters based on visualized trends  

**Automation**:  
- Logs write themselves during runtime  
- Visualizations auto-generate after each run (`evolution_progress.png`, `evolution_interactive.html`)  
- Errors immediately notify developers  

**Tech Stack**:  
- `structlog` (structured logging)  
- `sentry-sdk` (error monitoring)  
- `matplotlib` + `plotly` (visualizations)  
- `pandas` (data aggregation)  

No manual steps required — everything runs on pipeline hooks.

## 🧱 Coming Up

### Logging & Monitoring System

Planned improvements:
- `structlog` + `Sentry` for structured logging and error tracing
- Visualization of evolution progress (mutations, scores, survivor tree)
- Interactive dashboard to monitor strategy performance over time

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/elouwe/mutatorevo.git
cd mutatorevo
pip install -r requirements.txt
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

## Example Output

After 100 evolutionary epochs, you’ll see a summary like this:

```
=== Epoch 100/100 ===
[EVO] 2025-07-02 pool=13 new=5 best=mut_b77fcb30 best_score=1.71
💾 checkpoint → mutator_checkpoint_20250702-143618.pkl

Evolution completed!
Best strategy: Strategy(name=mut_4385417f, features=9)
Features: ['volatility_filter', 'window_small', 'use_rsi', 'window_large', 'threshold_high', 'use_ema', 'threshold_low', 'use_macd', 'trend_filter']
Score: 1.76
```

### 🧠 What it means:

- **Epoch 100/100** — Final generation of evolution  
- **pool=13** — Total number of strategies kept in memory  
- **new=5** — Newly generated strategies in this epoch  
- **best=mut_b77fcb30** — Best performing strategy in this round  
- **checkpoint** — Best mutant was saved for future reuse  
- **Best strategy** — Final evolved strategy that scored highest  
- **Features** — List of active features (indicators, parameters)  
- **Score** — Fitness score based on performance metrics  

You can now take this best strategy and test, visualize, or re-evolve it further.

## 🗂 Project Structure

```bash
mutator-evo/
├── src/
│   ├── mutator_evo/
│   │   ├── core/
│   │   │   ├── strategy_mutator.py
│   │   │   ├── strategy_embedding.py
│   │   │   └── config.py
│   │   ├── operators/
│   │   │   ├── interfaces.py
│   │   │   ├── mutation_impl.py
│   │   │   └── importance_calculator.py
│   │   └── scripts/
│   │       └── run_evolution.py
│   └── tests/
│       ├── unit/
│       └── integration/
├── data/
│   ├── market_data/
│   └── checkpoints/
├── configs/
│   ├── default.yaml
│   ├── crypto_config.yaml
│   └── equity_config.yaml
├── docs/
│   ├── architecture.md
│   └── user_guide.md
├── requirements.txt
├── setup.py
└── README.md
```

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
