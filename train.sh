#!/bin/bash

# Run `conda activate ms-hgnn` before running this script

pip install .

# ============== Training ==============
# Main Experiment 1: Contact Classification C2
python research/train_classification_msgn.py\
    --seed 2\
    --logger_project_name main_cls_c2\
    --model_type heterogeneous_gnn_c2\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-c2.yaml

# Main Experiment 1: Contact Classification K4
python research/train_classification_msgn.py\
    --seed 2\
    --logger_project_name main_cls_k4\
    --model_type heterogeneous_gnn_k4\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-k4.yaml

# Main Experiment 2: GRF Regression C2
python research/train_regression-grf_msgn.py\
    --seed 10\
    --logger_project_name main_grf_c2_d=3\
    --grf_body_to_world_frame True\
    --grf_dimension 3

python research/train_regression-grf_msgn.py\
    --seed 42\
    --logger_project_name main_grf_c2_d=3\
    --grf_body_to_world_frame True\
    --grf_dimension 3

python research/train_regression-grf_msgn.py\
    --seed 3407\
    --logger_project_name main_grf_c2_d=3\
    --grf_body_to_world_frame True\
    --grf_dimension 3
   
# 1D GRF
# python research/train_regression-grf_msgn.py\
#     --seed 3407\
#     --grf_dimension 1\
#     --logger_project_name main_grf_c2_d=1

# Main Experiment 3: COM Regression S4
python research/train_regression_com_msgn.py\
    --seed 42\
    --batch_size 64\
    --num_layers 8\
    --hidden_size 128\
    --lr 0.0012\
    --epochs 60\
    --logger_project_name com_debug\
    --model_type heterogeneous_gnn_s4_com

# Main Experiment 1: Sample Efficiency, Classification K4
python research/train_classification_msgn.py\
    --seed 2\
    --logger_project_name main_cls_k4\
    --model_type heterogeneous_gnn_k4\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-k4.yaml\
    --sample_ratio 0.45

# ============== Testing ==============

# # Main Experiment 1: Classification C2
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_c2\
#     --model_type heterogeneous_gnn_c2\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-c2.yaml

# Main Experiment 1: Classification K4
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_k4\
#     --model_type heterogeneous_gnn_k4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml

# Main Experiment 2: Regression (GRF)
# python research/evaluator_regression-grf_c2.py

# python research/evaluator_classification_k4.py\
#     --MorphSym_version K4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --path_to_checkpoint models/main_cls_k4/avid-moon-13/epoch=6-val_CE_loss=0.21062-val_F1_Score_Leg_Avg=0.94563.ckpt

# python research/evaluator_classification_k4.py\
#     --MorphSym_version K4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --path_to_checkpoint models/main_cls_k4_sample_eff/zesty-water-5/epoch=4-val_CE_loss=0.37975-val_F1_Score_Leg_Avg=0.88835.ckpt

# python research/evaluator_classification_k4.py\
#     --MorphSym_version K4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --path_to_checkpoint models/main_cls_k4_sample_eff/clean-aardvark-7/epoch=6-val_CE_loss=0.37497-val_F1_Score_Leg_Avg=0.90216.ckpt

# python research/evaluator_classification_k4.py\
#     --MorphSym_version K4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --path_to_checkpoint models/main_cls_k4_sample_eff/worthy-mountain-4/epoch=9-val_CE_loss=0.34174-val_F1_Score_Leg_Avg=0.92367.ckpt

python research/evaluator_classification_k4.py\
    --MorphSym_version K4\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-k4.yaml\
    --path_to_checkpoint models/main_cls_k4_sample_eff/worthy-mountain-4/epoch=9-val_CE_loss=0.34174-val_F1_Score_Leg_Avg=0.92367.ckpt


# Main Experiment 1: Classification C2
# python research/evaluator_classification_k4.py\
#     --MorphSym_version C2\
#     --group_operator_path cfg/mini_cheetah-c2.yaml\
#     --symmetry_mode MorphSym\
#     --path_to_checkpoint models/main_cls_c2/fearless-bush-9/epoch=5-val_CE_loss=0.20732-val_F1_Score_Leg_Avg=0.94922.ckpt

# Main Experiment 2: Regression (GRF) - d=1
# python research/evaluator_regression-grf_c2.py\
#     --path_to_checkpoint models/main_grf_c2_d=1/olive-vortex-6/epoch=13-val_MSE_loss=72.31809-val_L1_loss=1.73057.ckpt\
#     --test_only_on_z 1\
#     --grf_dimension 1

# # Main Experiment 2: Regression (GRF) - d=3
# python research/evaluator_regression-grf_c2.py\
#     --path_to_checkpoint models/main_grf_c2_d=3/helpful-gorge-6/epoch=7-val_MSE_loss=54.72873-val_L1_loss=2.28810.ckpt\
#     --grf_body_to_world_frame 1\
#     --grf_dimension 3

# python research/evaluator_regression-grf_c2.py\
#     --path_to_checkpoint models/main_grf_c2_d=3/glowing-snow-7/epoch=16-val_MSE_loss=57.40248-val_L1_loss=2.41434.ckpt\
#     --grf_body_to_world_frame 1\
#     --grf_dimension 3

# Main Experiment 2: Regression (COM)
# python research/train_regression_com_msgn.py\
#     --seed 42\
#     --batch_size 64\
#     --num_layers 8\
#     --hidden_size 128\
#     --lr 0.0012\
#     --epochs 60\
#     --logger_project_name com_debug\
#     --model_type heterogeneous_gnn_s4_com

# Main Experiment 3: Sample Efficiency, Classification K4
# python research/train_classification_msgn.py\
#     --seed 2\
#     --logger_project_name main_cls_k4\
#     --model_type heterogeneous_gnn_k4\
#     --symmetry_mode MorphSym\
#     --group_operator_path cfg/mini_cheetah-k4.yaml\
#     --sample_ratio 0.45

# Test Baseline
python research/evaluator_classification_k4.py\
    --MorphSym_version C2\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-c2.yaml\
    --path_to_checkpoint models/main_cls_c2_sample_eff/summer-donkey-9/epoch=10-val_CE_loss=0.51589-val_F1_Score_Leg_Avg=0.83822.ckpt

# Test GRF Baseline - d=3
python research/evaluator_regression-grf.py\
    --path_to_checkpoint models/grf_baseline_mihgnn_d=3/denim-elevator-5/epoch=4-val_MSE_loss=59.96093-val_L1_loss=2.49272.ckpt