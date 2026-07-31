source .venv/bin/activate

PROJECT_ROOT=$PWD
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" 
echo $PYTHONPATH
python experiment1/train_ex1/main_neural.py \
    --map-size 4 \
    --seed 0 \
    --dataset-dir experiment1/results/dataset_from_gazebo \
    --model mlp \
    --device cuda:0