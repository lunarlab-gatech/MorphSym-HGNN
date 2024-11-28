#! /bin/bash

# conda activate mi-hgnn

pip install .

# # Main Experiment 1: Classification C2
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_c2\
#     --model_type heterogeneous_gnn_c2\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-c2.yaml

# # Main Experiment 1: Classification K4
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_k4\
#     --model_type heterogeneous_gnn_k4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml

# Main Experiment 2: Regression (COM)
python research/train_regression-com_msgn.py\
    --seed 42\
    --batch_size 64\
    --num_layers 8\
    --hidden_size 128\
    --lr 0.0012\
    --epochs 60\
    --logger_project_name com_debug\
    --model_type heterogeneous_gnn_s4_com\
    --wandb_api_key 9a342aac9d095ea22ea22bfd71b4b56ff4175f9f

# Main Experiment 3: Sample Efficiency, Classification K4
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_k4\
#     --model_type heterogeneous_gnn_k4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --sample_ratio 0.45
