"""合并所有种子文件"""

from pathlib import Path
import pandas as pd

from config import METRICS_DIR

def get_csv_files_one_model(model_name: str, map_size: int):
    """获取单个模型的所有一个地图尺寸的所有seed的CSV文件
    : param model_name: 模型名称 options: ['mlp', 'lstm', 'transformer', 'tabm', 'xgboost', 'catboost']
    : param map_size: 地图尺寸 options: [4, 6, 10]
    """
    csv_files = []
    ex1_metrics_model_dir = METRICS_DIR / model_name
    all_seeds_files_dir = ex1_metrics_model_dir / f"map_size{map_size}"
    
    if all_seeds_files_dir.exists():
        for file in all_seeds_files_dir.glob("*.csv"):
            csv_files.append(file)
    
    return csv_files

def merge_all_seeds(model_name: str, map_size: int):
    """合并所有种子文件"""
    csv_files = get_csv_files_one_model(model_name, map_size)
    
    # 剔除文件中的model列和checkpoint列
    df = pd.concat([pd.read_csv(file).drop(columns=['model', 'checkpoint']) for file in csv_files])

    # seed列按照种子由小到大排序
    df.sort_values(by=['seed'], inplace=True)

    # 除了seed列的数据，其他列均保留四位小数
    df = df.round(4)

    csv_path = METRICS_DIR / f"{model_name}" / "summarize" / f"{model_name}_map_size{map_size}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    for model_name in ['mlp', 'lstm', 'transformer', 'tabm', 'xgboost', 'catboost']:
        merge_all_seeds(model_name, 10)