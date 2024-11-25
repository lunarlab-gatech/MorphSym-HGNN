#! /bin/bash

conda activate mi-hgnn
pip install .

python research/train_classification_msgn.py\
    --seed 2\
    --logger_project_name main_cls_c2\
    --model_type heterogeneous_gnn_c2\
    --symmetry_mode MorphSym\
    --group_operator_path cfg/mini_cheetah-c2.yaml