import argparse
import numpy as np
from pathlib import Path
import sklearn

from train_config import RegressionLabelStats
from train import base_metrics

def train_xgboost(
    parts: dict[str, np.ndarray],
    label_stats: "RegressionLabelStats",
    preprocessing: "sklearn.preprocessing.QuantileTransformer",
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, float], Path]:
    try:
        import joblib
        import xgboost as xgb
    except Exception as exc:
        raise RuntimeError("xgboost is not installed.") from exc

    from train_config import RegressionLabelStats

    y_train = (parts["train_y"] - label_stats.mean) / label_stats.std
    models = []
    for target_idx in range(y_train.shape[1]):
        model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(parts["train_x"], y_train[:, target_idx], verbose=False)
        models.append(model)
    pred_norm = np.column_stack([m.predict(parts["test_x"]) for m in models])
    pred = pred_norm * label_stats.std + label_stats.mean
    metrics = base_metrics(parts["test_y"], pred, parts["test_x_raw"], args.map_size)
    checkpoint = output_dir / f"xgboost_seed{seed}.joblib"
    joblib.dump(
        {
            "models": models,
            "preprocessing": preprocessing,
            "label_stats": label_stats,
            "regression_label_stats": label_stats,
            "metrics": metrics,
        },
        checkpoint,
    )
    return metrics, checkpoint


def train_catboost(
    parts: dict[str, np.ndarray],
    label_stats: "RegressionLabelStats",
    preprocessing: "sklearn.preprocessing.QuantileTransformer",
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, float], Path]:
    try:
        import joblib
        from catboost import CatBoostRegressor
    except Exception as exc:
        raise RuntimeError("catboost is not installed.") from exc

    from train_config import RegressionLabelStats

    y_train = (parts["train_y"] - label_stats.mean) / label_stats.std
    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiRMSE",
        random_seed=seed,
        verbose=False,
    )
    model.fit(parts["train_x"], y_train)
    pred_norm = np.asarray(model.predict(parts["test_x"]), dtype=np.float32)
    pred = pred_norm * label_stats.std + label_stats.mean
    metrics = base_metrics(parts["test_y"], pred, parts["test_x_raw"], args.map_size)
    checkpoint = output_dir / f"catboost_seed{seed}.joblib"
    joblib.dump(
        {
            "model": model,
            "preprocessing": preprocessing,
            "label_stats": label_stats,
            "regression_label_stats": label_stats,
            "metrics": metrics,
        },
        checkpoint,
    )
    return metrics, checkpoint