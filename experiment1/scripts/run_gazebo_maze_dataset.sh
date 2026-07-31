#!/usr/bin/env bash
# Gazebo 数据集生成启动脚本
# 用法:
#   ./experiment1/scripts/run_generate.sh              # 交互式输入 map_size / seed
#   ./experiment1/scripts/run_generate.sh 4 0          # 直接传入 map_size seed
#   ./experiment1/scripts/run_generate.sh 6            # 指定 map_size，seed 交互式输入
#   可通过环境变量覆盖: SAMPLES=50000 WORLD_NAME=custom_maze ./experiment1/scripts/run_generate.sh

set -euo pipefail

# ====================== 路径与默认参数 ======================
PROJECT_ROOT=$PWD
VENV_DIR="${PROJECT_ROOT}/.venv"
SCRIPT_PATH="${PROJECT_ROOT}/experiment1/generate_dataset_code/generate_dataset_gazebo.py"

# 默认参数
DEFAULT_MAP_SIZE=4
DEFAULT_SEED=0

# 可被环境变量覆盖的生成参数
SAMPLES="${SAMPLES:-20000}"
WORLD_NAME="${WORLD_NAME:-maze}"
ASTAR_RESOLUTION="${ASTAR_RESOLUTION:-0.10}"
MAX_WAIT="${MAX_WAIT:-10.0}"
REPORT_EVERY="${REPORT_EVERY:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiment1/results/dataset_from_gazebo}"

# ====================== 交互式输入函数 ======================
prompt_input() {
    local prompt="$1"
    local default="$2"
    local var_name="$3"
    local input
    local current_value

    # 检查变量是否已设置（通过命令行参数或环境变量）
    eval "current_value=\${${var_name}:-}"
    if [[ -n "${current_value}" ]]; then
        return 0
    fi

    read -rp "${prompt} [默认: ${default}]: " input
    if [[ -z "${input}" ]]; then
        eval "${var_name}=\"${default}\""
    else
        eval "${var_name}=\"${input}\""
    fi
}

# ====================== 参数解析 ======================
usage() {
    cat <<EOF
Usage: $(basename "$0") [MAP_SIZE] [SEED]

  交互式: 不传参数时，会依次询问 地图大小、seed
  命令行: 按顺序传入 地图大小、seed

  MAP_SIZE  可选: 4 | 6 | 10                        (默认: ${DEFAULT_MAP_SIZE})
  SEED      整数                                    (默认: ${DEFAULT_SEED})

环境变量可覆盖: SAMPLES, WORLD_NAME, ASTAR_RESOLUTION, MAX_WAIT, REPORT_EVERY, OUTPUT_DIR
EOF
}

# 先检查 help 参数
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# 解析命令行参数或交互输入
if [[ $# -eq 0 ]]; then
    # 完全交互式模式
    prompt_input "请输入地图大小 (4/6/10)" "${DEFAULT_MAP_SIZE}" "MAP_SIZE"
    prompt_input "请输入 seed" "${DEFAULT_SEED}" "SEED"
elif [[ $# -eq 1 ]]; then
    MAP_SIZE="$1"
    prompt_input "请输入 seed" "${DEFAULT_SEED}" "SEED"
elif [[ $# -eq 2 ]]; then
    MAP_SIZE="$1"
    SEED="$2"
else
    echo "[ERROR] 参数过多"
    usage
    exit 1
fi

# ====================== 参数校验 ======================
MAP_SIZE="${MAP_SIZE:-${DEFAULT_MAP_SIZE}}"
SEED="${SEED:-${DEFAULT_SEED}}"

case "${MAP_SIZE}" in
    4|6|10) ;;
    *) echo "[ERROR] 不支持的地图大小: ${MAP_SIZE} (可选: 4, 6, 10)"; usage; exit 1 ;;
esac

if ! [[ "${SEED}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] seed 必须为非负整数: ${SEED}"
    exit 1
fi

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "[ERROR] 找不到生成脚本: ${SCRIPT_PATH}"
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[ERROR] 找不到虚拟环境: ${VENV_DIR}"
    exit 1
fi

# 确保输出目录存在
mkdir -p "${OUTPUT_DIR}"

# ====================== 启动生成 ======================
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "============================================================"
echo " MapSize       : ${MAP_SIZE}"
echo " Seed          : ${SEED}"
echo " Samples       : ${SAMPLES}"
echo " WorldName     : ${WORLD_NAME}"
echo " AstarRes      : ${ASTAR_RESOLUTION}"
echo " MaxWait       : ${MAX_WAIT}"
echo " ReportEvery   : ${REPORT_EVERY}"
echo " OutputDir     : ${OUTPUT_DIR}"
echo "============================================================"

python "${SCRIPT_PATH}" \
    --map-size "${MAP_SIZE}" \
    --seed "${SEED}" \
    --samples "${SAMPLES}" \
    --world-name "${WORLD_NAME}" \
    --astar-resolution "${ASTAR_RESOLUTION}" \
    --max-wait-for-observation "${MAX_WAIT}" \
    --report-every "${REPORT_EVERY}" \
    --output-dir "${OUTPUT_DIR}"

echo "[DONE] Gazebo dataset written to: ${OUTPUT_DIR}"