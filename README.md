# Incremental Deep Learning Model for Large-Scale Demand Prediction
**End-to-End Neural Network System · BZAN 554 – Deep Learning · University of Tennessee, Knoxville**

---

## Business Objective

Modern e-commerce systems generate datasets that exceed available memory, making traditional in-memory modeling approaches impractical. The objective was to design and implement a scalable, incremental deep learning pipeline that accurately predicts demand while maintaining low memory usage and full computational efficiency — producing interpretable outputs that translate clearly to non-technical stakeholders.

Each record in the dataset represents a product selling on an e-commerce website. If a product goes out of stock and returns, a new record is created — meaning the data reflects real lifecycle and replenishment dynamics rather than static snapshots.

---

## Data & Variables

**Input variables:** `sku` · `price` · `order` · `duration` · `category`

**Response variable:** `quantity` (total quantity sold)

> Note: All categorical variables are integer encoded. All numeric variables are divided by a constant. The dataset is confidential and copyrighted and is not included in this repository.

---

## System Architecture (End-to-End)

| Stage | Component | Purpose |
|---|---|---|
| Ingestion | Chunked CSV reader | Incremental data loading, no full dataset in memory |
| Scaling | Streaming statistics | On-the-fly standardization using running means and stds |
| Feature Engineering | Raw + log-transformed inputs | Captures nonlinear relationships across price and order |
| Training | Mini-batch neural network | Incremental weight updates across epochs |
| Evaluation | Robust test alignment | Handles distribution mismatch between train and test |
| Interpretability | PDPs + permutation importance | Feature influence diagnostics for stakeholder communication |

The architecture scales to datasets exceeding system memory, with peak RAM usage of 299.6 MB and final RAM usage of 358.3 MB during full training runs.

---

## Neural Network Architecture

**Input layer:**
- Price (raw)
- Order (raw)
- log(Price)
- log(Order)
- Product Category (one-hot encoded)

**Hidden layers (sigmoid activation):**
1. 96 nodes
2. 48 nodes
3. 24 nodes

**Output layer:**
- 1 node (linear activation)

**Training configuration:**

| Parameter | Value |
|---|---|
| Learning rate | 0.0003 |
| Loss function | Mean Squared Error |
| Batch size | 1,024 |
| Epochs | 10 |
| Target transformation | log(1 + quantity) |
| Feature clipping | ±10σ z-clip |

---

## Model Design & Engineering Approach

- Designed a 3-layer feed-forward neural network (96 → 48 → 24 nodes) with sigmoid activations and a linear output layer
- Implemented log-transformed target variable (`log1p(quantity)`) to stabilize variance and improve learning dynamics
- Engineered hybrid feature representation — both raw and log-transformed inputs — to capture nonlinear pricing and order effects
- Applied chunk-based z-score standardization with z-clipping (±10σ) to maintain numerical stability across large, heterogeneous data
- Used mini-batch incremental training (`train_on_batch`) to scale beyond RAM limitations
- Achieved **R² ≈ 0.107** on unseen test data, demonstrating learning under strict computational constraints

---

## Key Engineering Contributions

- Built a **streaming training pipeline** (`chunked pd.read_csv`) that processes millions of records without memory overflow
- Developed **custom dual scaling functions** (`compute_dual_scaling_parameters`) using running statistics computed in a single pass over training data
- Implemented **learning curve tracking** with moving average smoothing (window = 200 batches) to monitor model convergence in real time
- Designed **robust test-time price alignment** using IQR-based normalization to handle distributional shift between train and test sets
- Engineered **partial dependence plots** for price, order, and category to interpret nonlinear feature-outcome relationships
- Generated **permutation-based variable importance analysis** with 3-repeat averaging for stable, model-agnostic explainability
- Managed **GitHub repository** as Repo Owner, reviewed contributions, and maintained a stable main branch

---

## Model Insights & Interpretability

**Variable importance (permutation-based, R² drop — ranked):**

| Rank | Feature | R² Drop | Interpretation |
|---|---|---|---|
| 1 | Category | ~0.036 | Strongest driver of predicted demand |
| 2 | Price | ~0.034 | Second most influential feature |
| 3 | Order | ~0.016 | Moderate importance with nonlinear dynamics |

**Partial dependence findings:**

**Price PDP:**
- Strong inverse relationship between price and predicted quantity
- At the lowest price points, predicted quantity averages ~15 units
- At the highest price points (~2.5), predicted quantity flattens near ~3 units
- Consistent with economic demand theory — higher price suppresses quantity

**Order PDP:**
- U-shaped nonlinear relationship
- Predicted quantity dips to a minimum around order 10–20, then increases steadily through order ~150+
- Suggests early-lifecycle products and high-repeat-stock products both drive higher quantities; mid-range lifecycle products show the lowest predicted demand

**Category PDP (top 6 most frequent categories: 7, 5, 17, 10, 30, 6):**
- Category 17 shows the highest average predicted demand (~14 units)
- Category 30 follows at ~10 units
- Categories 5 and 6 show the lowest average predicted demand (~7 units)
- Substantial variation across categories confirms category is the most important predictor

**Learning curve:**
- MSE started near 2.2 at early batches and decreased sharply through the first ~500,000 instances
- Curve continued to decline gradually, reaching approximately 1.23 by ~5 million instances
- Smooth, consistent convergence validates incremental training effectiveness with no signs of divergence or instability

---

## Performance & Scalability

| Metric | Value |
|---|---|
| Test R² | ~0.107 |
| Training time | ~143 seconds |
| Peak RAM usage | ~300 MB |
| Final RAM usage | ~358 MB |
| Dataset handling | Fully scalable beyond system memory |
| Batch size | 1,024 records |
| Epochs | 10 |

---

## Technical Implementation Notes

**`compute_scaling.py`** — Computes train-only scaling parameters in a single streaming pass; returns raw and log-transformed means and standard deviations for price and order without loading the full dataset

**`finalcode.py`** — Orchestrates the full pipeline: scaling computation, model construction, incremental training, test evaluation, partial dependence plot generation, and permutation variable importance analysis

**`chunk_to_features()`** — Applies dual-scale standardization, z-clipping, one-hot encoding of category, and log-transformation of the target in a single vectorized operation per batch

**`partial_dependence_numeric()` / `partial_dependence_category()`** — Sweeps feature values across a quantile grid while holding all other features fixed; averages predictions across a baseline sample for interpretable marginal effect plots

**`permutation_importance()`** — Randomly permutes each feature independently across 3 repeats and measures average R² drop relative to baseline, producing a stable, model-agnostic importance ranking

---

## Runbook Structure

**BZAN 554 Deep Learning Runbook**

**I. Compute Scaling**
- A. Configuration setup
- B. Scaling

**II. Training and Testing**
- A. Configuration setup
- B. R²
- C. Neural network
- D. Standardize
- E. Column selection
- F. Scaling test
- G. Data combination
- H. Batches
- I. Test on test data

**III. Modeling**
- A. RAM usage and run time
- B. Partial dependence plots
- C. Variable importance plot

---

## Model Outputs

| File | Description |
|---|---|
| `learning_curve.png` | Moving average of MSE across training instances (~5M records over 10 epochs) |
| `pdp_price.png` | Partial dependence of predicted quantity on price (inverse, nonlinear) |
| `pdp_order.png` | Partial dependence of predicted quantity on order (U-shaped, nonlinear) |
| `pdp_category.png` | Category-level average predicted demand (top 6 most frequent categories) |
| `variable_importance.png` | Permutation importance ranked by R² decrease |

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Pandas` · `Matplotlib` · `Chunked CSV Processing` · `Mini-Batch Training` · `Permutation Importance` · `Partial Dependence Analysis` · `Git / GitHub` · `VSCode`

---

## My Role

**Model Architect · ML Engineer · Feature Engineering Lead · Repo Owner · Performance Analyst**

I served as Repo Owner and led key contributions across neural network architecture, RAM profiling, runbook management, and presentation design. I was responsible for ensuring code reproducibility and system reliability, and for coordinating team contributions through GitHub throughout the project.
