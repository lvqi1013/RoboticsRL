#!/usr/bin/env bash
# 训练子目标预测神经网络的启动脚本
# 用法:
#   ./experiment1/scripts/run_train.sh                       # 交互式输入模型/地图大小/seed
#   ./experiment1/scripts/run_train.sh mlp 4 0               # 直接传入 模型 地图大小 seed
#   ./experiment1/scripts/run_train.sh tabm                  # 指定模型，其余交互式输入
#   ./experiment1/scripts/run_train.sh mlp path/to/data.csv  # 指定模型和数据集 (向后兼容)
#   可通过环境变量覆盖: DEVICE=cuda:1 EPOCHS=300 ./experiment1/scripts/run_train.sh

set -euo pipefail

# ====================== 路径与默认参数 ======================
PROJECT_ROOT=$PWD
VENV_DIR="${PROJECT_ROOT}/.venv"
SCRIPT_DIR="${PROJECT_ROOT}/experiment1/train_ex1"

# 默认参数
DEFAULT_MODEL="mlp"
DEFAULT_MAP_SIZE=4
DEFAULT_SEED=0
DATASET_DIR="${PROJECT_ROOT}/experiment1/results/dataset_from_gazebo"

# 可被环境变量覆盖的超参数
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
PATIENCE="${PATIENCE:-40}"
EVAL_EVERY="${EVAL_EVERY:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiment1/results}"

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
Usage: $(basename "$0") [MODEL] [MAP_SIZE] [SEED]
       $(basename "$0") [MODEL] [DATASET_PATH]  (向后兼容)

  交互式: 不传参数时，会依次询问 模型、地图大小、seed
  命令行: 按顺序传入 模型、地图大小、seed

  MODEL     可选: mlp | lstm | transformer | tabm  | xgboost | catboost (默认: ${DEFAULT_MODEL})
  MAP_SIZE  可选: 4 | 6 | 10                        (默认: ${DEFAULT_MAP_SIZE})
  SEED      整数                                    (默认: ${DEFAULT_SEED})

环境变量可覆盖: DEVICE, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, EVAL_EVERY, OUTPUT_DIR, DATASET_DIR
EOF
}

# 先检查 help 参数
case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# 解析命令行参数或交互输入
if [[ $# -eq 0 ]]; then
    # 交互式模式
    prompt_input "请输入模型 (mlp/lstm/transformer/tabm/xgboost/catboost)" "${DEFAULT_MODEL}" "MODEL"
    prompt_input "请输入地图大小 (4/6/10)" "${DEFAULT_MAP_SIZE}" "MAP_SIZE"
    prompt_input "请输入 seed" "${DEFAULT_SEED}" "SEED"
elif [[ $# -eq 1 ]]; then
    MODEL="$1"
    prompt_input "请输入地图大小 (4/6/10)" "${DEFAULT_MAP_SIZE}" "MAP_SIZE"
    prompt_input "请输入 seed" "${DEFAULT_SEED}" "SEED"
elif [[ $# -eq 2 ]]; then
    MODEL="$1"
    # 检查第二个参数是地图大小(数字)还是数据集路径
    if [[ "$2" =~ ^[0-9]+$ ]]; then
        MAP_SIZE="$2"
        prompt_input "请输入 seed" "${DEFAULT_SEED}" "SEED"
    else
        # 向后兼容: 第二个参数是数据集路径
        DATASET="$2"
    fi
elif [[ $# -eq 3 ]]; then
    MODEL="$1"
    MAP_SIZE="$2"
    SEED="$3"
else
    echo "[ERROR] 参数过多"
    usage
    exit 1
fi

# ====================== 参数校验 ======================
# 如果没有 DATASET 变量，根据 MAP_SIZE 和 SEED 构建
if [[ -z "${DATASET:-}" ]]; then
    DATASET="${DATASET_DIR}/map_size_${MAP_SIZE}/subgoal_gazebo_maze_map${MAP_SIZE}_seed${SEED}.csv"
fi

# 如果通过数据集路径传入，尝试从中解析 MAP_SIZE、SEED 和 DATASET_DIR
if [[ -z "${MAP_SIZE:-}" ]] || [[ -z "${SEED:-}" ]]; then
    if [[ -f "${DATASET}" ]]; then
        basename_file=$(basename "${DATASET}")
        if [[ "${basename_file}" =~ map([0-9]+)_seed([0-9]+) ]]; then
            MAP_SIZE="${BASH_REMATCH[1]}"
            SEED="${BASH_REMATCH[2]}"
            # 从完整路径中提取目录
            DATASET_DIR=$(dirname "${DATASET}")
        fi
    fi
fi

# 如果仍未解析出，使用默认值
MAP_SIZE="${MAP_SIZE:-${DEFAULT_MAP_SIZE}}"
SEED="${SEED:-${DEFAULT_SEED}}"

case "${MODEL}" in
    mlp|lstm|transformer|tabm|xgboost|catboost) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 不支持的模型: ${MODEL}"; usage; exit 1 ;;
esac

if [[ -n "${MAP_SIZE:-}" ]]; then
    case "${MAP_SIZE}" in
        4|6|10) ;;
        *) echo "[ERROR] 不支持的地图大小: ${MAP_SIZE} (可选: 4, 6, 10)"; exit 1 ;;
    esac
fi

if [[ ! -f "${DATASET}" ]]; then
    echo "[ERROR] 数据集文件不存在: ${DATASET}"
    exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
    echo "[ERROR] 找不到虚拟环境: ${VENV_DIR}"
    exit 1
fi

# ====================== 启动训练 ======================
cd "${PROJECT_ROOT}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "============================================================"
echo " Model       : ${MODEL}"
echo " MapSize     : ${MAP_SIZE}"
echo " Seed        : ${SEED}"
echo " Dataset     : ${DATASET}"
echo " Device      : ${DEVICE}"
echo " Epochs      : ${EPOCHS}"
echo " BatchSize   : ${BATCH_SIZE}"
echo " LR          : ${LR}"
echo " WeightDecay : ${WEIGHT_DECAY}"
echo " Patience    : ${PATIENCE}"
echo " EvalEvery   : ${EVAL_EVERY}"
echo " OutputDir   : ${OUTPUT_DIR}"
echo "============================================================"

PYTHONPATH="${PROJECT_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}" \
python "${SCRIPT_DIR}/main_train.py" \
    --dataset-dir "${DATASET_DIR}" \
    --map-size "${MAP_SIZE}" \
    --seed "${SEED}" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --patience "${PATIENCE}" \
    --eval-every "${EVAL_EVERY}" \
    --output-dir "${OUTPUT_DIR}"

echo "[DONE] 结果已写入: ${OUTPUT_DIR}"
