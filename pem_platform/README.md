# PEM Platform

PEM Platform is a GitHub-ready, cross-platform Python project that provides a unified workflow for:

- DoE design generation (full factorial, fractional factorial, Plackett–Burman, Latin hypercube)
- Classical DoE analysis (effects, interactions, ANOVA, alias indicators)
- Surrogate modeling with uncertainty (Gaussian Process Regression)
- Hybrid DoE + ML modeling patterns
- Next-experiment recommendation for process optimization
- Retrieval for process documentation and reports

It is designed to be "universal" by using:

1. A YAML configuration that declares your **factor list** and **responses**
2. A consistent dataset schema (tabular runs)
3. Optional process-specific plugins for constraints and ingestion

This repository includes:

- Working examples under `examples/`
- A CLI tool called `pem`
- GitHub Actions CI workflows for Windows and Linux

---

## Repository structure

The code is packaged under `src/pem/`.

```
pem_platform/
  src/pem/
    core/        # config + dataset utilities
    doe/         # design generation + DoE analysis
    model/       # GPR surrogates + CV
    optimize/    # recommenders (simple acquisition + Ax BO)
    rag/         # local retrieval index + retrieval chat
    viz/         # plotting helpers (saves PNGs)
    adapters/    # example constraint plugin
    cli.py       # CLI entrypoint
```

If you were expecting the conceptual layout shown earlier (for example `core/schema.py`, `surrogate/gpr_sklearn.py`, etc.), PEM Platform implements the same concepts but uses a Python package layout that groups features into subpackages (`core/`, `doe/`, `model/`, `optimize/`, `rag/`, `viz/`).

---

## Install

### Development install from source

```bash
git clone <your-repo-url>
cd pem-platform

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -e ".[dev]"
```

### Optional extras

Better embeddings for retrieval:

```bash
pip install -e ".[dev,embeddings]"
```

Bayesian optimization via Ax and BoTorch:

```bash
pip install -e ".[dev,ax]"
```

You can combine extras:

```bash
pip install -e ".[dev,embeddings,ax]"
```

---

## Dataset schema

Your dataset is a table of experiment runs.

Minimum requirement:

- One column per **factor** declared in your YAML config
- One column per **response** declared in your YAML config

Example header:

```csv
temperature_C,pressure_mTorr,rf_power_W,gas_ratio,chamber,thickness_nm,roughness_nm
```

You may add other metadata columns (run_id, tool_id, timestamp, lot, operator). PEM will ignore them unless you add stricter validation.

---

## Configuration

The YAML file defines your process-specific factor list and objectives.

See the example:

- `examples/configs/tool_demo.yaml`

### Factor types

- `continuous`: numeric with `bounds: [low, high]`
- `discrete`: numeric with explicit `levels: [...]` (recommended for setpoints)
- `categorical`: string levels (chamber, tool, recipe variant)

### Single response and multi response

PEM trains one model per response and returns a mean and uncertainty per response.

- Single response training: `--responses thickness_nm`
- Multi response training: `--responses thickness_nm roughness_nm`

For optimization, multiple responses are converted into a scalar objective using `scalarization.weights`.

---

## Quickstart

All commands below run with the included example config and example datasets.

### Validate data

```bash
pem validate-data --config examples/configs/tool_demo.yaml examples/datasets/demo_single_response.csv
```

### Generate DoE designs

Full factorial 2-level:

```bash
pem design full-factorial --config examples/configs/tool_demo.yaml --output design_full.csv
```

Fractional factorial 2-level:

```bash
pem design fractional-factorial --config examples/configs/tool_demo.yaml \
  --generator "a b c ab" --output design_frac.csv
```

Plackett–Burman screening:

```bash
pem design plackett-burman --config examples/configs/tool_demo.yaml --output design_pb.csv
```

Latin hypercube:

```bash
pem design lhs --config examples/configs/tool_demo.yaml --n-samples 25 --output design_lhs.csv
```

### Run DoE analysis

```bash
pem doe analyze --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_factorial_runs.csv \
  --response thickness_nm \
  --max-interaction-order 2 \
  --out-prefix out/doe_thickness
```

This writes:

- `out/doe_thickness_coef.csv`
- `out/doe_thickness_anova.csv`
- `out/doe_thickness_alias.csv`

### Plot DoE effects

```bash
pem plot doe-effects --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_factorial_runs.csv \
  --response thickness_nm \
  --out-dir plots
```

Outputs:

- `plots/doe_thickness_nm_pareto.png`

### Train a surrogate model

Trend + residual GP:

```bash
pem model train --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern trend_residual_gp \
  --out model.joblib
```

Predict:

```bash
pem model predict --config examples/configs/tool_demo.yaml \
  --model model.joblib \
  --data examples/datasets/demo_predict_points.csv \
  --output predictions.csv
```

### Plot surrogate diagnostics

```bash
pem plot model-diagnostics --config examples/configs/tool_demo.yaml \
  --model model.joblib \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --out-dir plots
```

Outputs:

- `plots/parity_thickness_nm.png`
- `plots/residual_thickness_nm.png`

### Recommend next experiments

Simple acquisition-based recommender:

```bash
pem recommend --config examples/configs/tool_demo.yaml \
  --model model.joblib \
  --n-candidates 2000 \
  --n-recommendations 8 \
  --acquisition ucb \
  --kappa 2.0 \
  --output next_runs.csv
```

---

## Hybrid DoE and ML patterns

Select using `--pattern` in `pem model train`.

These map to the patterns discussed earlier.

### Pattern A

Screen with DoE then train GP on the retained factors:

```bash
pem model train --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern doe_screen_then_gp \
  --top-k 3 \
  --out model_screen.joblib
```

### Pattern B

Use a linear plus interaction trend and model the residuals with a GP:

```bash
pem model train --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern trend_residual_gp \
  --ridge-alpha 1.0 \
  --out model_trend_resid.joblib
```

### Pattern C

Add engineered interaction features then fit GP:

```bash
pem model train --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern interaction_features_gp \
  --out model_interactions.joblib
```

### Pattern D

DoE-guided GP initialization:

- PEM uses DoE main-effect magnitudes to initialize GP lengthscales when possible.
- In scikit-learn GPR this is a heuristic initialization, not a full Bayesian prior.

You get this automatically with `pure_gp` and `doe_screen_then_gp`.

### Pattern E

Alias awareness for fractional factorials:

- `pem doe analyze` reports alias indicators using correlations among columns in the fitted design matrix.
- This is a practical warning system, not a full symbolic alias resolver.

---

## Model tuning

GPR tuning knobs are exposed in the CLI.

Common options:

- `--kernel-kind matern|rbf`
- `--nu <float>` for Matern kernels
- `--n-restarts <int>` for GPR hyperparameter optimization

Example:

```bash
pem model train --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern pure_gp \
  --kernel-kind matern \
  --nu 1.5 \
  --n-restarts 5 \
  --out model_tuned.joblib
```

Cross-validation:

```bash
pem model cv --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm \
  --pattern pure_gp \
  --n-splits 5 \
  --n-restarts 3
```

---

## Bayesian optimization with Ax

PEM includes an optional Ax-based Bayesian optimization recommender.

1. Install the optional dependency:

```bash
pip install -e ".[ax]"
```

2. Run Ax recommendation using historical data:

```bash
pem ax recommend --config examples/configs/tool_demo.yaml \
  --data examples/datasets/demo_single_response.csv \
  --responses thickness_nm roughness_nm \
  --n-recommendations 8 \
  --output ax_next_runs.csv
```

How it works:

- PEM computes a scalar objective called `utility` from the selected responses using your config goals and `scalarization.weights`.
- Ax optimizes `utility` and proposes new factor settings.
- If you have a constraint plugin, suggestions are filtered for feasibility.

---

## Retrieval and chat

PEM includes a local retrieval system over `.txt`, `.md`, and `.rst` files.

Index a directory:

```bash
pem rag index --docs examples/knowledge --index-dir .rag_index
```

Query:

```bash
pem rag query --index-dir .rag_index --question "What does RF power affect?" --top-k 5
```

Interactive retrieval chat:

```bash
pem rag chat --index-dir .rag_index
```

Notes:

- `pem rag chat` is retrieval-only. It prints the top passages relevant to your question.
- In a production system, you typically pass the retrieved passages into your preferred LLM to generate a grounded answer and optionally call tools.

---

## Testing and linting

Run tests:

```bash
pytest
```

Run lint:

```bash
ruff check .
```

---

## CI and releases

GitHub Actions workflows are included:

- `.github/workflows/ci.yml` runs tests and lint on Windows and Linux.
- `.github/workflows/release.yml` builds sdist and wheels on tags.

Create a release tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

---

## Limitations

- Classic DoE analysis is focused on 2-level coding and is best interpreted with true 2-level designs.
- Fractional factorial alias detection is a pragmatic indicator, not a full alias algebra engine.
- The Ax recommender uses scalarization into a single objective `utility`. True Pareto multi-objective optimization can be added if needed.

---

## License

MIT
