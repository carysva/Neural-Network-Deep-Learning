# Neural Network (Incremental Learning on Large Data) — Portfolio Summary

## Project Summary
This project built an interpretable neural-network model to predict order quantity using a pricing and order feature set, optimized and trained to scale to large datasets using an incremental learning approach.

The output is designed for business stakeholders by pairing a predictive model with explainability tooling (variable importance and partial dependence plots) to show how inputs influence predictions.

## Inputs
Model features used from the dataset:
- `Price (raw)`
- `Order (raw)`
- `log(Price)`
- `log(Order)`
- `Product Category`

## Model Architecture
- Feed-forward neural network with sigmoid hidden layers
- Hidden layer sizes (in order): `96 → 48 → 24`
- Output: `1` continuous value

## Training Configuration
- Learning rate: `0.0003`
- Loss function: `Mean Squared Error`
- Batch size: `1024`
- Epochs: `10`
- Numeric preprocessing: standardized and clipped
- Target transformation: `log(1 + quantity)` (log1p-style)

## Training Performance (from project presentation)
- Training time: `142.68` seconds
- Peak RAM usage: `299.6` MB
- Final RAM usage: `358.3` MB

## Explainability & Diagnostics
- Variable importance visualization to highlight dominant drivers
- Partial dependence plots to communicate direction and magnitude of effect
- Learning curve view to confirm convergence and incremental learning behavior

## Why It Matters (Business Analyst Angle)
The combination of prediction + explanation supports decisions by:
- Identifying which input drivers have the strongest influence on quantity predictions
- Communicating how changes to price/order/category affect the expected outcome (in business terms)
- Enabling scalable training workflows suited to large datasets

