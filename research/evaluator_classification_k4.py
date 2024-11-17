import os
import glob
from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import evaluate_model
import torch
import numpy as np
import pandas
from datetime import datetime

def main(K4_version=True, 
         path_to_checkpoint=None,
         symmetry_operator_list=None,
         symmetry_mode=None,
         group_operator_path=None):
    # ================================= CHANGE THIS ====================================
    if K4_version:
        import mi_hgnn.datasets_py.LinTzuYaunDataset_K4 as linData
        model_type = 'heterogeneous_gnn_k4'
    else:
        import mi_hgnn.datasets_py.LinTzuYaunDataset as linData
        model_type = 'heterogeneous_gnn' # 'heterogeneous_gnn_k4'

    # Swap legs to evaluate the model on the opposite leg
    # swap_legs_list = [(1, 3), (1, 0), (1, 2), None, ((1, 0), (3, 2)), ((1, 3), (2, 0))] # Swap tuple: FR: 0, FL: 1, RR: 2, RL: 3, None: no swap
    # swap_legs_list = [((1, 0), (3, 2)), ((1, 3), (2, 0))] # Debugging
    # legs_dict = {0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}
    # ==================================================================================
    
    # print the path to checkpoint
    print("===> Starting Evaluation on: ", path_to_checkpoint.split('/')[-2:])

    # Set parameters
    history_length = 150
    path_to_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()

    # Check that the user filled in the necessary parameters
    if path_to_checkpoint is None:
        raise ValueError("Please provide a checkpoint path by editing this file!")
    
    # path to save csv
    path_to_save_csv = path_to_checkpoint.replace('.ckpt', '.csv') # csv save location and file name
    ckpt_name = path_to_checkpoint.split('/')[-1].replace('.ckpt', '')
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # add the swap legs and leg swap mode to the path
    path_to_save_csv = path_to_save_csv.replace(ckpt_name, ckpt_name + '-symmetry_operator_list={}-symmetry_mode={}-{}'.format(symmetry_operator_list, symmetry_mode, time_stamp))
    print("===> Results saving to: ", path_to_save_csv)

    # Initialize DataFrame outside the loop
    columns = ["Symmetry Operator", "Model Accuracy", "Rear-Left", "Front-Left", "Rear-Right", "Front-Right", "F1 Avg"]
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each symmetry
    # for swap_legs in swap_legs_list:
    for symmetry_operator in symmetry_operator_list:
        swap_legs = None
        print("================================================")
        # print(f"Leg Swap Mode: {leg_swap_mode}")
        # if swap_legs:
        #     if isinstance(swap_legs[0], int):
        #         print(f"Swapping legs: {legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}")
        #         swap_str = f"{legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}"
        #     elif isinstance(swap_legs[0], tuple):
        #         print(f"Swapping legs: {legs_dict[swap_legs[0][0]]} and {legs_dict[swap_legs[0][1]]}, {legs_dict[swap_legs[1][0]]} and {legs_dict[swap_legs[1][1]]}")
        #         swap_str = f"{legs_dict[swap_legs[0][0]]} and {legs_dict[swap_legs[0][1]]}, {legs_dict[swap_legs[1][0]]} and {legs_dict[swap_legs[1][1]]}"
        #     else:
        #         raise ValueError("Invalid swap legs")
        # else:
        #     print("No legs swapped")
        #     swap_str = "None"
        print(f"Symmetry Operator: {symmetry_operator}")
        print(f"Model Type: {model_type}")
        print(f"Path to Checkpoint: {path_to_checkpoint}")
        print("================================================")

        test_dataset = prepare_test_dataset(linData, path_to_urdf, model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)

        # Evaluate with model
        pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(path_to_checkpoint, test_dataset, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        # Save to DataFrame
        df = pandas.concat([df, pandas.DataFrame([[symmetry_operator, acc.item(), f1_leg_0.item(), f1_leg_1.item(), f1_leg_2.item(), f1_leg_3.item(), f1_avg_legs.item()]], columns=columns)], ignore_index=True)

        # Print the results
        print_results(acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs)

    # Save to csv
    df.to_csv(path_to_save_csv, index=False)
    print("===> Evaluation Finished! Results saved to: ", path_to_save_csv)

def prepare_test_dataset(linData, path_to_urdf, model_type, history_length, normalize=True, swap_legs=None, symmetry_operator=None, symmetry_mode=None, group_operator_path=None):
    """Prepare the test dataset"""
    datasets = []
    dataset_classes = [
        linData.LinTzuYaunDataset_air_jumping_gait,
        linData.LinTzuYaunDataset_concrete_pronking,
        linData.LinTzuYaunDataset_concrete_right_circle,
        linData.LinTzuYaunDataset_forest,
        linData.LinTzuYaunDataset_small_pebble
    ]
    
    dataset_paths = [
        'LinTzuYaun-AJG',
        'LinTzuYaun-CP',
        'LinTzuYaun-CRC',
        'LinTzuYaun-F',
        'LinTzuYaun-SP'
    ]
    
    for dataset_class, dataset_path in zip(dataset_classes, dataset_paths):
        dataset = dataset_class(
            Path(Path('.').parent, 'datasets', dataset_path).absolute(),
            path_to_urdf,
            'package://yobotics_description/',
            'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description',
            model_type,
            history_length,
            normalize=normalize,
            swap_legs=swap_legs,
            symmetry_operator=symmetry_operator,
            symmetry_mode=symmetry_mode,
            group_operator_path=group_operator_path
        )
        datasets.append(dataset)
    
    test_dataset = torch.utils.data.ConcatDataset(datasets)
    return torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

def print_results(acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs):
    """Print the results"""
    print("Model Accuracy: ", acc.item())
    print("F1-Score Leg 0: ", f1_leg_0.item())
    print("F1-Score Leg 1: ", f1_leg_1.item())
    print("F1-Score Leg 2: ", f1_leg_2.item())
    print("F1-Score Leg 3: ", f1_leg_3.item())
    print("F1-Score Legs Avg: ", f1_avg_legs.item())

if __name__ == "__main__":
    K4_version = True
    path_to_checkpoint = "/home/swei303/Documents/proj/MorphSym-HGNN/models/ethereal-microwave-12/epoch=6-val_Accuracy=0.89402.ckpt"
    symmetry_operator_list = [None]  # Can be 'gs' or 'gt' or 'gr' or None
    symmetry_mode = 'MorphSym' # Can be 'Euclidean' or 'MorphSym' or None
    group_operator_path = '/home/swei303/Documents/proj/MorphSym-HGNN/cfg/mini_cheetah-k4.yaml'

    if os.path.isdir(path_to_checkpoint):
        checkpoint_files = glob.glob(os.path.join(path_to_checkpoint, "*.ckpt"))
        for file in checkpoint_files:
            main(K4_version=K4_version, path_to_checkpoint=file, symmetry_operator_list=symmetry_operator_list, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
    else:
        main(K4_version=K4_version, path_to_checkpoint=path_to_checkpoint, symmetry_operator_list=symmetry_operator_list, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)