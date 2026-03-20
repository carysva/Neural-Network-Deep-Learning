import pandas as pd
import numpy as np

# Numeric columns to standardize
NUMERIC_COLS = ["price", "order", "duration"]

def compute_scaling_parameters(filepath, chunksize=10000):

    sums = np.zeros(len(NUMERIC_COLS))
    squared_sums = np.zeros(len(NUMERIC_COLS))
    total_count = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize):

        data = chunk[NUMERIC_COLS].values

        sums += data.sum(axis=0)
        squared_sums += (data ** 2).sum(axis=0)
        total_count += len(chunk)

    means = sums / total_count
    variances = (squared_sums / total_count) - (means ** 2)
    stds = np.sqrt(variances)

    # 🔥 ADD EPSILON HERE
    stds[stds < 1e-8] = 1e-8

    return means, stds


def compute_dual_scaling_parameters(filepath, chunksize=10000):
    """
    Compute train-only scaling for both raw and log-transformed features.
    Returns:
      raw_means, raw_stds, log_means, log_stds
    where raw/log vectors are aligned to ["price", "order"].
    """
    cols = ["price", "order"]
    sums = np.zeros(len(cols), dtype=np.float64)
    squared_sums = np.zeros(len(cols), dtype=np.float64)
    log_sums = np.zeros(len(cols), dtype=np.float64)
    log_squared_sums = np.zeros(len(cols), dtype=np.float64)
    total_count = 0

    for chunk in pd.read_csv(filepath, chunksize=chunksize, usecols=cols):
        data = chunk[cols].values.astype(np.float64)
        log_data = np.log1p(np.clip(data, 0, None))

        sums += data.sum(axis=0)
        squared_sums += (data ** 2).sum(axis=0)
        log_sums += log_data.sum(axis=0)
        log_squared_sums += (log_data ** 2).sum(axis=0)
        total_count += len(chunk)

    raw_means = sums / total_count
    raw_vars = (squared_sums / total_count) - (raw_means ** 2)
    raw_stds = np.sqrt(np.maximum(raw_vars, 0.0))
    raw_stds[raw_stds < 1e-8] = 1e-8

    log_means = log_sums / total_count
    log_vars = (log_squared_sums / total_count) - (log_means ** 2)
    log_stds = np.sqrt(np.maximum(log_vars, 0.0))
    log_stds[log_stds < 1e-8] = 1e-8

    return raw_means, raw_stds, log_means, log_stds
