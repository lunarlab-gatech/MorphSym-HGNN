from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import evaluate_model
import torch
import numpy as np
import pandas
from datetime import datetime

def main():
    # ================================= CHANGE THIS ====================================
    K4_version = True

    if K4_version:
        import mi_hgnn.datasets_py.LinTzuYaunDataset_K4 as linData
        model_type = 'heterogeneous_gnn_k4'
        path_to_checkpoint = "/home/swei303/Documents/proj/MorphSym-HGNN/models/splendid-armadillo-4/epoch=10-val_CE_loss=0.33491.ckpt" # Path to specific checkpoint file
    else:
        import mi_hgnn.datasets_py.LinTzuYaunDataset as linData
        model_type = 'heterogeneous_gnn' # 'heterogeneous_gnn_k4'
        path_to_checkpoint = "ckpts/Classification Experiment/Main Experiment/leafy-totem-5/epoch=10-val_CE_loss=0.30258.ckpt" # Path to specific checkpoint file

    # Swap legs to evaluate the model on the opposite leg
    # swap_legs_list = [(1, 3), (1, 0), (1, 2), None, ((1, 0), (3, 2)), ((1, 3), (2, 0))] # Swap tuple: FR: 0, FL: 1, RR: 2, RL: 3, None: no swap
    # swap_legs_list = [((1, 0), (3, 2)), ((1, 3), (2, 0))] # Debugging
    symmetry_operator_list = ['gs', 'gt', 'gr']  # Can be 'gs' or 'gt' or 'gr' or None
    symmetry_mode = 'MorphSym' # Can be 'Euclidean' or 'MorphSym' or None
    group_operator_path = '/home/swei303/Documents/proj/MorphSym-HGNN/cfg/mini_cheetah-k4.yaml'
    # ==================================================================================
    legs_dict = {0: 'FR', 1: 'FL', 2: 'RR', 3: 'RL'}
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
    print("Results saving to: ", path_to_save_csv)

    # Initialize DataFrame outside the loop
    columns = ["Symmetry Operator", "Model Accuracy", "Rear-Left", "Front-Left",	"Rear-Right", "Front-Right", "F1 Avg"]
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

        air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        forest = linData.LinTzuYaunDataset_forest(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        small_pebble = linData.LinTzuYaunDataset_small_pebble(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])

        # Convert them to subsets
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

        # Evaluate with model
        pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(path_to_checkpoint, test_dataset)
        # Save to DataFrame
        df = pandas.concat([df, pandas.DataFrame([[symmetry_operator, acc.item(), f1_leg_0.item(), f1_leg_1.item(), f1_leg_2.item(), f1_leg_3.item(), f1_avg_legs.item()]], columns=columns)], ignore_index=True)

        # Print the results
        print("Model Accuracy: ", acc.item())
        print("F1-Score Leg 0: ", f1_leg_0.item())
        print("F1-Score Leg 1: ", f1_leg_1.item())
        print("F1-Score Leg 2: ", f1_leg_2.item())
        print("F1-Score Leg 3: ", f1_leg_3.item())
        print("F1-Score Legs Avg: ", f1_avg_legs.item())

    # Save to csv
    if path_to_save_csv is not None:
        df.to_csv(path_to_save_csv, index=False)
    else:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_to_save_csv = path_to_checkpoint.replace('.ckpt', f'_classification_results_{time_stamp}.csv') # csv save location and file name
        print("Results saving to: ", path_to_save_csv)
        df.to_csv(path_to_save_csv, index=False)

if __name__ == "__main__":
    main()
