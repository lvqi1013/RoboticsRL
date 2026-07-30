import argparse
from pathlib import Path
from train import Trainer
import pandas as pd
import numpy as np
import json
import csv
from dataclasses import asdict

from utils import get_map_seed, feature_columns, prepare_split
from train_config import BASE_FEATURE_COLS, TARGET_COLS, SubgoalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    # parser.add_argument("--map-size", type=int, choices=[4, 6, 10], required=True)
    parser.add_argument(
        "--model",
        # default=["tabm", "mlp", "transformer", "lstm", "xgboost", "catboost"],
    )
    # parser.add_argument("--seed", type=int,)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--boost-rounds", type=int, default=180)
    parser.add_argument("--output-dir", type=Path, default="experiment1/results")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-missing-optional", action="store_true")
    return parser

def write_results(results: list[SubgoalResult], output_dir: Path) -> None:
    per_seed_path = output_dir / "per_seed_metrics.csv"
    with per_seed_path.open("w", newline="", encoding="utf-8") as f:
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
    aggregate_rows = []
    for model in sorted({r.model for r in results}):
        subset = [r for r in results if r.model == model]
        row = {"model": model, "n_seeds": len(subset)}
        for metric in metric_names:
            values = np.array([getattr(r, metric) for r in subset], dtype=np.float64)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=0))
        aggregate_rows.append(row)

    aggregate_path = output_dir / "aggregate_metrics.csv"
    with aggregate_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"Wrote {per_seed_path}")
    print(f"Wrote {aggregate_path}")

def main():
    args = build_parser().parse_args()

    map_size, seed = get_map_seed(str(args.dataset))
    device = args.device

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dataset).dropna()
    cols = feature_columns(df)
    x = df[cols].to_numpy(dtype=np.float32) # 获取69维的输入数据
    y = df[TARGET_COLS].to_numpy(dtype=np.float32) # 标签数据：子目标

    # 保存此次运行的参数配置
    config_data = {
        "dataset": str(args.dataset),
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
    
    parts, label_stats, _preprocessing = prepare_split(x, y, seed)
    # dict_keys(['train_x_raw', 'val_x_raw', 'test_x_raw', 'train_y', 'val_y', 'test_y', 'train_x', 'val_x', 'test_x', 'train_y_norm'])


    model_name = args.model.lower()
    print(f"\n=== model={model_name} seed={seed} ===")  
      
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

    metrics, checkpoint = trainer.run()
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
    write_results(results, args.output_dir)


if __name__ == '__main__':
    main()
