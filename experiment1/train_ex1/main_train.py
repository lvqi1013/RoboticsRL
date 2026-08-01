import argparse
from pathlib import Path
from train import Trainer, base_metrics
import pandas as pd
import numpy as np
import json
import csv
import sklearn
from dataclasses import asdict

from utils import feature_columns, prepare_split
from train_config import (
    BASE_FEATURE_COLS,
    TARGET_COLS,
    SubgoalResult,
    RegressionLabelStats,
    CHEACKPOINT_OUTPUT_DIR,
)
from boost_models import train_xgboost, train_catboost

# 非神经网络（基于树的）模型直接走各自的训练函数，不经过 Trainer 的 PyTorch 流程
BOOST_MODELS = {"xgboost", "catboost"}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=Path("experiment1/results/dataset_from_gazebo"))
    parser.add_argument("--map-size", type=int, choices=[4, 6, 10], required=True)
    parser.add_argument("--model",)# default=["tabm", "mlp", "transformer", "lstm", "xgboost", "catboost"],
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--output-dir", type=Path, default="experiment1/results")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-missing-optional", action="store_true")
    return parser

def write_results(results: list[SubgoalResult], output_dir: Path, seed: int, map_size: int, model_name: str) -> None:
    seed_path = output_dir / "ex1_metrics" / f"{model_name}"/f"map_size{map_size}" /f"metrics_{model_name}_s{seed}_m{map_size}.csv"
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    with seed_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    metric_names = [
        "mse",
        "rmse",
        "mae",
        "r2",
        "mde",
        "median_de",
        "in_bounds_rate",
        "valid_subgoal_rate",
        "obstacle_hit_rate",
        "reachable_segment_rate",
        "progress_rate",
        "mean_progress",
    ]
    # aggregate_rows = []
    # for model in sorted({r.model for r in results}):
    #     subset = [r for r in results if r.model == model]
    #     row = {"model": model, "n_seeds": len(subset)}
    #     for metric in metric_names:
    #         values = np.array([getattr(r, metric) for r in subset], dtype=np.float64)
    #         row[f"{metric}_mean"] = float(values.mean())
    #         row[f"{metric}_std"] = float(values.std(ddof=0))
    #     aggregate_rows.append(row)

    # aggregate_path = output_dir  / "ex1_metrics" / "aggregate_metrics.csv"
    # with aggregate_path.open("w", newline="", encoding="utf-8") as f:
    #     writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
    #     writer.writeheader()
    #     writer.writerows(aggregate_rows)

    print(f"Wrote {seed_path}")
    # print(f"Wrote {aggregate_path}")

def main():
    args = build_parser().parse_args()

    map_size = args.map_size
    seed = args.seed
    device = args.device
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds_path = args.dataset_dir / f"map_size_{map_size}" /f"subgoal_gazebo_maze_map{map_size}_seed{seed}.csv"

    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {ds_path}")

    df = pd.read_csv(ds_path).dropna()
    cols = feature_columns(df)
    x = df[cols].to_numpy(dtype=np.float32) # 获取69维的输入数据
    y = df[TARGET_COLS].to_numpy(dtype=np.float32) # 标签数据：子目标

    # 保存此次运行的参数配置
    config_data = {
        "dataset": str(ds_path),   
        "map_size": map_size,
        "model": args.model,
        "seeds": seed,
        "epochs": args.epochs,
        "feature_columns": cols,
    }
    config_path = args.output_dir / "run_config" / f"run_config_{args.model}_ms{map_size}_seed{seed}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)
    
    parts, label_stats, preprocessing = prepare_split(x, y, seed)
    # dict_keys(['train_x_raw', 'val_x_raw', 'test_x_raw', 'train_y', 'val_y', 'test_y', 'train_x', 'val_x', 'test_x', 'train_y_norm'])


    model_name = args.model.lower()
    print(f"\n=== model={model_name} seed={seed} ===")  

    if model_name in BOOST_MODELS:
        CHEACKPOINT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if model_name == "xgboost":
            metrics, checkpoint = train_xgboost(
                parts, label_stats, preprocessing, args, seed, CHEACKPOINT_OUTPUT_DIR
            )
        else:
            metrics, checkpoint = train_catboost(
                parts, label_stats, preprocessing, args, seed, CHEACKPOINT_OUTPUT_DIR
            )
    else:
        trainer = Trainer(
            model_name,
            parts=parts,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_stats=label_stats,
            seed=seed,
            device=device,
            patience=args.patience,
            eval_every=args.eval_every,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            map_size=map_size,
        )

        metrics, checkpoint = trainer.run(preprocessing)
    results: list[SubgoalResult] = []
    results.append(
                SubgoalResult(
                    model=model_name,
                    seed=seed,
                    checkpoint=str(checkpoint),
                    **metrics,
                ))
    print(
            f"{model_name} seed={seed}: "
            f"RMSE={metrics['rmse']:.5f} MDE={metrics['mde']:.5f} "
            f"Valid={metrics['valid_subgoal_rate']:.3f} "
            f"ObstacleHit={metrics['obstacle_hit_rate']:.3f} "
            f"SegmentFree={metrics['reachable_segment_rate']:.3f}"
            )

    if not results:
        raise RuntimeError("No model results were produced.")
    write_results(results, args.output_dir, seed=seed, map_size=map_size, model_name=model_name)


if __name__ == '__main__':
    main()
