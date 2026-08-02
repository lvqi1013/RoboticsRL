from pathlib import Path

CURRENT_FILE_PATH = Path(__file__).resolve()
EX1_DIR = CURRENT_FILE_PATH.parent.parent
ROOT_DIR = EX1_DIR.parent

# print(f"CURRENT_FILE_PATH: {CURRENT_FILE_PATH}")
# print(f"EX1_DIR: {EX1_DIR}")
# print(f"ROOT_DIR: {ROOT_DIR}")

METRICS_DIR = EX1_DIR / "results" / "ex1_metrics"
# print(f"METRICS_DIR: {METRICS_DIR}")