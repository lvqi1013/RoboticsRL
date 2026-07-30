source .venv/bin/activate
python experiment1/train_tabm/main_neural.py \
    --dataset experiment1/generate_dataset/subgoal_gazebo_maze_map4_seed0.csv\
    --model mlp \
    --device cuda:0