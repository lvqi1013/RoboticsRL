"""计算数据集的均值和标准差并汇总所有模型的"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from config import METRICS_DIR

def cal_mean_std(model_name: str, map_size: int) -> None:
    """计算数据集的均值和标准差并汇总所有模型的"""
    ex1_metrics_model_dir = METRICS_DIR / model_name
    merged_file_path = ex1_metrics_model_dir / "summarize" / f"{model_name}_map_size{map_size}.csv"
    
    # 读取CSV文件
    df = pd.read_csv(merged_file_path)
    
    # 计算均值和标准差
    mean_values = df.mean()
    std_values = df.std()
    
    # print(f"Mean values: {mean_values}")
    # print(f"Std values: {std_values}")

    # 构建 mean/std 两行，seed 列替换为标签
    mean_row = mean_values.copy()
    std_row = std_values.copy()
    # mean_row["seed"] = "mean"
    # std_row["seed"] = "std"

    # 追加到原 DataFrame 末尾
    result_df = pd.concat(
        [df, mean_row.to_frame().T, std_row.to_frame().T],
        ignore_index=True
    )
    result_df_path = ex1_metrics_model_dir / "summarize" / f"ms_{model_name}_map_size{map_size}.csv"
    result_df_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存结果到Excel文件
    result_df.to_csv(result_df_path, index=False)
    



if __name__ == "__main__":
    cal_mean_std("catboost", 10)