# WhyLab: Causal Safety Monitoring Framework for Stable Agent Self-Improvement

> NeurIPS 2026 Submission

## Overview

WhyLab is a causal safety monitoring framework that prevents cognitive policy oscillation in self-improving AI agents. It provides three contributions:

- **C1**: Information-theoretic drift detection
- **C2**: Sensitivity-aware effect filtering (E-values + partial R^2)
- **C3**: Lyapunov-bounded adaptive damping

```mermaid
flowchart LR
    Agent[Agent Policy $\theta_t$] --> Evaluate[Test & Reward $\Delta R$]
    Evaluate --> Audit{Audit Layer: C1, C2, C3}
    Audit --> Safe[Safe Update $\theta_{t+1}$]
    Safe --> Agent
```

## Repository Structure

```text
paper/          # LaTeX source (main.tex, references.bib)
experiments/    # All experiment code and results
  results/      # CSV/Parquet output files
  prompts/      # LLM prompt templates for E5/E7
  cache/        # Cached API responses (gitignored)
  data/         # Downloaded datasets (gitignored)
submission/     # Packaged ZIP files for OpenReview
```

## Reproducing Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

### Synthetic Experiments (No API key required)

These experiments are fully reproducible with no external dependencies:

```bash
# E1 & E2: Core components
python -m experiments.e1_drift_detection
python -m experiments.e2_sensitivity_filter

# E3a: Stability Validation (PID, SGD, Adam, Lyapunov)
python -m experiments.e3a_stationary

# Proxy Correlation Analysis (Theorem 1 validation)
python -m experiments.proxy_correlation_analysis

# E6: Non-stationary Agent Environment
python -m experiments.e6_nonstationary_agent
```

### LLM Agent Experiments (API key required)

These experiments require a `.env` file containing API keys (e.g., `GEMINI_API_KEY` or `OPENAI_API_KEY` depending on the model):

```bash
cp .env.example .env
# Edit .env and add your respective API Keys

# E5: SWE-bench Lite (300 problems x 5 seeds)
python -m experiments.e5_swebench_benchmark

# E7: Dynamic ReAct Benchmark with GPT-5.4
python -m experiments.e7_react_dynamic_benchmark
```

> **Note**: E5/E7 results are non-deterministic due to LLM sampling.
> Pre-computed results are available in `experiments/results/`.

### Analysis Scripts (Use cached results)

```bash
# E5 subset analysis (oscillating vs non-oscillating)
python -m experiments.e5_subset_analysis

# Safety baseline comparison (Best-of-N, Rollback, etc.)
python -m experiments.e5_safety_baselines
```

> **Statistical Reproducibility Note**: All non-deterministic differences reported in the paper (e.g., Pass@1 improvements in E5) are validated using Bootstrap 95% Confidence Intervals ($N=10,000$). For reviewers wishing to recalculate the CI, you can adapt the following snippet:
> ```python
> import numpy as np
> # Example: Bootstrap CI for Pass@1 differences
> diffs = []
> for _ in range(10000):
>     idx = np.random.randint(0, len(pass_reflexion), len(pass_reflexion))
>     diff = pass_whylab[idx].mean() - pass_reflexion[idx].mean()
>     diffs.append(diff)
> print(f"Pass@1 diff 95% CI: {np.percentile(diffs, [2.5, 97.5])}")
> ```

## Building the Paper

Requires a TeX distribution (MiKTeX or TeX Live):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Important Notes

- **Best-of-N baselines** in Table 6 are estimated via `1-(1-p_1)^N` under an independence assumption (not actual parallel runs). This is stated in the paper.
- **SWE-bench evaluation (E5)** uses lightweight (string-match) test execution.
- **Dynamic ReAct Framework (E7)** utilizes OpenAI's GPT-5.4 model to demonstrate that the framework generalizes to flagship out-of-family models.

## License

This code is provided for academic review purposes.
