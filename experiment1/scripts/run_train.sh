#!/usr/bin/env bash
# 训练子目标预测神经网络的启动脚本
# 用法:
#   ./experiment1/scripts/run_train.sh                       # 使用默认参数 (mlp + map4_seed0)
#   ./experiment1/scripts/run_train.sh tabm                  # 指定模型: mlp/lstm/transformer/tabm
#   ./experiment1/scripts/run_train.sh mlp path/to/data.csv  # 指定模型和数据集
#   可通过环境变量覆盖: DEVICE=cuda:1 EPOCHS=300 ./experiment1/scripts/run_train.sh

set -euo pipefail

# ====================== 路径与默认参数 ======================
PROJECT_ROOT="/home/ps/nav_code"
VENV_DIR="${PROJECT_ROOT}/.venv"
SCRIPT_DIR="${PROJECT_ROOT}/experiment1/train_tabm"

# 默认数据集 (文件名需包含 map{N}_seed{M} 以便程序解析)
DEFAULT_DATASET="${PROJECT_ROOT}/experiment1/generate_dataset/subgoal_gazebo_maze_map4_seed0.csv"
DEFAULT_MODEL="mlp"

# 可被环境变量覆盖的超参数
MODEL="${1:-${MODEL:-${DEFAULT_MODEL}}}"
DATASET="${2:-${DATASET:-${DEFAULT_DATASET}}}"
DEVICE="${DEVICE:-cuda:0}"
EPOCHS="${EPOCHS:-180}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-2e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-5}"
PATIENCE="${PATIENCE:-40}"
EVAL_EVERY="${EVAL_EVERY:-5}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/experiment1/results}"

# ====================== 参数校验 ======================
usage() {
    cat <<EOF
Usage: $(basename "$0") [MODEL] [DATASET]
  MODEL     可选: mlp | lstm | transformer | tabm   (默认: ${DEFAULT_MODEL})
  DATASET   CSV 文件路径, 文件名需含 map{N}_seed{M}  (默认: ${DEFAULT_DATASET})

环境变量可覆盖: DEVICE, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY, PATIENCE, EVAL_EVERY, OUTPUT_DIR
EOF
}

case "${MODEL}" in
    mlp|lstm|transformer|tabm) ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] 不支持的模型: ${MODEL}"; usage; exit 1 ;;
esac

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

PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
python "${SCRIPT_DIR}/main_neural.py" \
    --dataset "${DATASET}" \
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
