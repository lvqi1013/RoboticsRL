import pandas as pd
from train_config import BASE_FEATURE_COLS, TARGET_COLS
import re
import sklearn
import numpy as np
from train_config import RegressionLabelStats

def feature_columns(df: pd.DataFrame) -> list[str]:
    """拼接激光雷达的列"""
    lidar_cols = sorted(
        [c for c in df.columns if c.startswith("lidar_")],
        key=lambda name: int(name.split("_")[1]),
    )
    cols = BASE_FEATURE_COLS + lidar_cols
    missing = [c for c in cols + TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    return cols


def prepare_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[dict[str, np.ndarray], RegressionLabelStats, sklearn.preprocessing.QuantileTransformer]:
    idx = np.arange(len(y))

    # 按照8:1:1划分训练集，验证集，测试集
    trainval_idx, test_idx = sklearn.model_selection.train_test_split(
        idx, train_size=0.9, random_state=seed
    )
    train_idx, val_idx = sklearn.model_selection.train_test_split(
        trainval_idx, train_size=8 / 9, random_state=seed
    )

    parts = {
        "train_x_raw": x[train_idx].copy(),
        "val_x_raw": x[val_idx].copy(),
        "test_x_raw": x[test_idx].copy(),
        "train_y": y[train_idx].copy(),
        "val_y": y[val_idx].copy(),
        "test_y": y[test_idx].copy(),
    }   

    # 给训练特征添加极小高斯噪声
    noise = (
        np.random.default_rng(seed)
        .normal(0.0, 1e-5, parts["train_x_raw"].shape)
        .astype(np.float32)
    )
    preprocessing = sklearn.preprocessing.QuantileTransformer(
        n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
        output_distribution="normal",
        subsample=10**9,
        random_state=seed,
    ).fit(parts["train_x_raw"] + noise)

    for part in ["train", "val", "test"]:
        parts[f"{part}_x"] = preprocessing.transform(parts[f"{part}_x_raw"]).astype(np.float32)
        parts[f"{part}_x"] = np.nan_to_num(parts[f"{part}_x"], nan=0.0)

        mean = parts["train_y"].mean(axis=0)
    std = parts["train_y"].std(axis=0)
    std[std == 0.0] = 1.0
    label_stats = RegressionLabelStats(mean=mean, std=std)
    parts["train_y_norm"] = ((parts["train_y"] - mean) / std).astype(np.float32)
    return parts, label_stats, preprocessing     