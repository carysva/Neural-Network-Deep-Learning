# Deep Learning Runbook (Incremental Neural Network)

## Purpose
Train and evaluate a feed-forward neural network for quantity prediction using price/order inputs and a product category feature. The workflow is structured to support incremental learning on large datasets while preserving explainability.

## Feature Engineering
1. Create numeric features:
   - `Price (raw)`
   - `Order (raw)`
   - `log(Price)`
   - `log(Order)`
2. Encode `Product Category` for model input (e.g., numeric/category encoding as used in the notebook).
3. Preprocess numeric inputs:
   - Standardize
   - Clip to reduce outlier sensitivity (per the project configuration)
4. Transform target variable:
   - Use `log(1 + quantity)` for training stability

## Model Configuration
- Hidden layers (sigmoid): `96 → 48 → 24`
- Output: `1` regression neuron

Hyperparameters:
- Learning rate: `0.0003`
- Loss: `Mean Squared Error`
- Batch size: `1024`
- Epochs: `10`

## Training & Monitoring
1. Train the model with the configuration above.
2. Capture:
   - Learning curves (incremental learning behavior)
   - Variable importance output
   - Partial dependence plots for interpretability
   - Resource usage (training time and RAM)

## Interpretability Outputs
Deliverables generated during/after training:
- Variable importance plot (driver ranking)
- Partial dependence plots for `Price`, `Order`, and `Category`
- Learning curve view to confirm convergence

## Resource Notes (from project presentation)
- Training time: `142.68` seconds
- Peak RAM usage: `299.6` MB
- Final RAM usage: `358.3` MB

## What to Include in the Portfolio
- `Deep Learning Presentation (1).pdf`
- `PORTFOLIO_SUMMARY.md`
- `RUNBOOK.md`

