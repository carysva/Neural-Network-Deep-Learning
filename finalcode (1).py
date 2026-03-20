import time
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from compute_scaling import compute_dual_scaling_parameters

###############################################
# Column Definitions / Training Setup
###############################################
TARGET_COL = "quantity"
TRAIN_FILE = "pricing.csv"
TEST_FILE = "pricing_test.csv"

BATCH_SIZE = 1024
EPOCHS = 10
LEARNING_RATE = 0.0003
RANDOM_SEED = 42

PRED_LOG_LOWER = -2.0
PRED_LOG_UPPER = 5.2
INPUT_Z_CLIP = 10.0
MOVING_AVG_WINDOW = 200

###############################################
# Metrics
###############################################
def r2_score_np(y_true, y_pred):
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom <= 0:
        return 0.0
    return 1 - np.sum((y_true - y_pred) ** 2) / denom


###############################################
# Model
###############################################
def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(96, activation="sigmoid"),
        tf.keras.layers.Dense(48, activation="sigmoid"),
        tf.keras.layers.Dense(24, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )
    return model


###############################################
# Feature Engineering
###############################################
def chunk_to_features(
    chunk,
    category_max,
    raw_means,
    raw_stds,
    log_means,
    log_stds,
):
    price = chunk["price"].values.astype(np.float32)
    order = chunk["order"].values.astype(np.float32)

    log_price = np.log1p(np.clip(price, 0, None))
    log_order = np.log1p(np.clip(order, 0, None))

    numeric = np.stack([
        (price - raw_means[0]) / raw_stds[0],
        (order - raw_means[1]) / raw_stds[1],
        (log_price - log_means[0]) / log_stds[0],
        (log_order - log_means[1]) / log_stds[1],
    ], axis=1).astype(np.float32)

    numeric = np.clip(numeric, -INPUT_Z_CLIP, INPUT_Z_CLIP)

    cat = np.clip(chunk["category"].values.astype(np.int32), 0, category_max)
    cat_oh = np.zeros((len(chunk), category_max + 1), dtype=np.float32)
    cat_oh[np.arange(len(chunk)), cat] = 1.0

    X = np.concatenate([numeric, cat_oh], axis=1)
    y = np.log1p(chunk[TARGET_COL].values.astype(np.float32)).reshape(-1, 1)

    return X, y


###############################################
# Incremental Training + Learning Curve
###############################################
def incremental_train(
    model,
    category_max,
    raw_means,
    raw_stds,
    log_means,
    log_stds,
):
    start = time.time()

    instances_seen = 0
    inst_curve = []
    mse_curve = []
    loss_buffer = deque(maxlen=MOVING_AVG_WINDOW)

    for epoch in range(EPOCHS):
        losses = []

        for chunk in pd.read_csv(
            TRAIN_FILE,
            chunksize=BATCH_SIZE,
            usecols=["price", "order", "quantity", "category"],
        ):
            X_batch, y_batch = chunk_to_features(
                chunk,
                category_max,
                raw_means,
                raw_stds,
                log_means,
                log_stds,
            )

            loss = model.train_on_batch(X_batch, y_batch)
            losses.append(float(loss))

            instances_seen += len(chunk)
            loss_buffer.append(float(loss))

            if len(loss_buffer) == MOVING_AVG_WINDOW:
                inst_curve.append(instances_seen)
                mse_curve.append(np.mean(loss_buffer))

        print(f"Epoch {epoch + 1}/{EPOCHS} - avg batch loss: {np.mean(losses):.6f}")

    print("Training time (seconds):", time.time() - start)
    return inst_curve, mse_curve


###############################################
# Test Utilities
###############################################
def load_test_with_order_choice(train_order_median):
    raw = pd.read_csv(TEST_FILE, header=None)
    raw.columns = ["sku", "price", "quantity", "col3", "col4", "category"]

    d3 = abs(np.log1p(np.median(raw["col3"])) - np.log1p(train_order_median))
    d4 = abs(np.log1p(np.median(raw["col4"])) - np.log1p(train_order_median))
    order_col = "col4" if d4 < d3 else "col3"

    return pd.DataFrame({
        "price": raw["price"],
        "order": raw[order_col],
        "quantity": raw["quantity"],
        "category": raw["category"],
    }), order_col


def robust_align_test_price_to_train(train_price, test_price):
    train_med = np.median(train_price)
    test_med = np.median(test_price)

    train_iqr = np.quantile(train_price, 0.75) - np.quantile(train_price, 0.25)
    test_iqr = np.quantile(test_price, 0.75) - np.quantile(test_price, 0.25)

    train_iqr = max(train_iqr, 1e-8)
    test_iqr = max(test_iqr, 1e-8)

    aligned = (test_price - test_med) / test_iqr
    aligned = aligned * train_iqr + train_med
    return aligned


###############################################
# Main
###############################################
if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    train_meta = pd.read_csv(TRAIN_FILE, usecols=["category", "order", "price"])
    category_max = int(train_meta["category"].max())
    train_order_median = np.median(train_meta["order"].values)

    raw_means, raw_stds, log_means, log_stds = compute_dual_scaling_parameters(TRAIN_FILE)

    input_dim = 4 + (category_max + 1)
    model = build_model(input_dim)

    inst_curve, mse_curve = incremental_train(
        model,
        category_max,
        raw_means,
        raw_stds,
        log_means,
        log_stds,
    )

    # ---- Save learning curve ----
    plt.figure(figsize=(8, 5))
    plt.plot(inst_curve, mse_curve)
    plt.xlabel("Number of instances learned")
    plt.ylabel("Moving average of MSE")
    plt.title("Learning Curve (Incremental Neural Network)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_curve.png")
    plt.close()

    # ---- Test evaluation ----
    test_df, order_col = load_test_with_order_choice(train_order_median)
    print(f"Using TEST order source column: {order_col}")

    test_df["price"] = robust_align_test_price_to_train(
        train_meta["price"].values,
        test_df["price"].values,
    )

    X_test, _ = chunk_to_features(
        test_df,
        category_max,
        raw_means,
        raw_stds,
        log_means,
        log_stds,
    )

    y_test = test_df[TARGET_COL].values.astype(np.float32)
    y_pred_log = model.predict(X_test, batch_size=4096).flatten()
    y_pred_log = np.clip(y_pred_log, PRED_LOG_LOWER, PRED_LOG_UPPER)
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    print(f"Test R^2: {r2_score_np(y_test, y_pred):.6f}")
#Test R^2: 0.107072

###############################################
# Partial Dependence Plots
###############################################
def partial_dependence_numeric(
    model,
    base_df,
    feature_name,
    grid,
    category_max,
    raw_means,
    raw_stds,
    log_means,
    log_stds,
    filename,
):
    pd_values = []

    for val in grid:
        df_tmp = base_df.copy()
        df_tmp[feature_name] = val

        X_tmp, _ = chunk_to_features(
            df_tmp,
            category_max,
            raw_means,
            raw_stds,
            log_means,
            log_stds,
        )

        preds = model.predict(X_tmp, batch_size=4096, verbose=0).flatten()
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        pd_values.append(preds.mean())

    plt.figure(figsize=(7, 5))
    plt.plot(grid, pd_values)
    plt.xlabel(feature_name)
    plt.ylabel("Predicted quantity (avg)")
    plt.title(f"Partial Dependence: {feature_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def partial_dependence_category(
    model,
    base_df,
    categories,
    category_max,
    raw_means,
    raw_stds,
    log_means,
    log_stds,
    filename,
):
    pd_values = []

    for cat in categories:
        df_tmp = base_df.copy()
        df_tmp["category"] = cat

        X_tmp, _ = chunk_to_features(
            df_tmp,
            category_max,
            raw_means,
            raw_stds,
            log_means,
            log_stds,
        )

        preds = model.predict(X_tmp, batch_size=4096, verbose=0).flatten()
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        pd_values.append(preds.mean())

    plt.figure(figsize=(7, 5))
    plt.bar([str(c) for c in categories], pd_values)
    plt.xlabel("category")
    plt.ylabel("Predicted quantity (avg)")
    plt.title("Partial Dependence: category")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


###############################################
# Run Partial Dependence Plots
###############################################

# ---- Train-only sample for PDP baseline ----
pdp_base = pd.read_csv(
    TRAIN_FILE,
    usecols=["price", "order", "quantity", "category"],
    nrows=3000,      # keep fast & RAM-safe
)

# ---- PRICE PDP ----
price_grid = np.quantile(
    pdp_base["price"].values,
    np.linspace(0.05, 0.95, 20)
)

partial_dependence_numeric(
    model=model,
    base_df=pdp_base,
    feature_name="price",
    grid=price_grid,
    category_max=category_max,
    raw_means=raw_means,
    raw_stds=raw_stds,
    log_means=log_means,
    log_stds=log_stds,
    filename="pdp_price.png",
)

# ---- ORDER PDP ----
order_grid = np.quantile(
    pdp_base["order"].values,
    np.linspace(0.05, 0.95, 20)
)

partial_dependence_numeric(
    model=model,
    base_df=pdp_base,
    feature_name="order",
    grid=order_grid,
    category_max=category_max,
    raw_means=raw_means,
    raw_stds=raw_stds,
    log_means=log_means,
    log_stds=log_stds,
    filename="pdp_order.png",
)

# ---- CATEGORY PDP (top 6 most frequent) ----
top_categories = (
    pdp_base["category"]
    .value_counts()
    .head(6)
    .index
    .tolist()
)

partial_dependence_category(
    model=model,
    base_df=pdp_base,
    categories=top_categories,
    category_max=category_max,
    raw_means=raw_means,
    raw_stds=raw_stds,
    log_means=log_means,
    log_stds=log_stds,
    filename="pdp_category.png",
)

###############################################
# Permutation Variable Importance
###############################################
def permutation_importance(
    model,
    base_df,
    feature_names,
    category_max,
    raw_means,
    raw_stds,
    log_means,
    log_stds,
    n_repeats=3,
):
    # Baseline performance
    X_base, _ = chunk_to_features(
        base_df,
        category_max,
        raw_means,
        raw_stds,
        log_means,
        log_stds,
    )

    y_true = base_df[TARGET_COL].values.astype(np.float32)
    y_pred_log = model.predict(X_base, batch_size=4096, verbose=0).flatten()
    y_pred = np.clip(np.expm1(y_pred_log), 0, None)

    baseline_r2 = r2_score_np(y_true, y_pred)

    importances = {}

    for feature in feature_names:
        drops = []

        for _ in range(n_repeats):
            df_perm = base_df.copy()
            df_perm[feature] = np.random.permutation(df_perm[feature].values)

            X_perm, _ = chunk_to_features(
                df_perm,
                category_max,
                raw_means,
                raw_stds,
                log_means,
                log_stds,
            )

            y_pred_log = model.predict(X_perm, batch_size=4096, verbose=0).flatten()
            y_pred = np.clip(np.expm1(y_pred_log), 0, None)

            r2_perm = r2_score_np(y_true, y_pred)
            drops.append(baseline_r2 - r2_perm)

        importances[feature] = np.mean(drops)

    return importances


def plot_importances(importances, filename):
    features = list(importances.keys())
    values = np.array(list(importances.values()))

    order = np.argsort(values)[::-1]
    features = [features[i] for i in order]
    values = values[order]

    plt.figure(figsize=(8, 5))
    plt.barh(features, values)
    plt.xlabel("Decrease in R² after permutation")
    plt.title("Permutation Variable Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

###############################################
# Run Variable Importance
###############################################

# Train-only sample (no test leakage)
vi_base = pd.read_csv(
    TRAIN_FILE,
    usecols=["price", "order", "quantity", "category"],
    nrows=4000,
)

feature_list = ["price", "order", "category"]

importances = permutation_importance(
    model=model,
    base_df=vi_base,
    feature_names=feature_list,
    category_max=category_max,
    raw_means=raw_means,
    raw_stds=raw_stds,
    log_means=log_means,
    log_stds=log_stds,
    n_repeats=3,
)

plot_importances(importances, "variable_importance.png")