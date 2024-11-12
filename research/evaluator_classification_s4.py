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
    swap_legs_list = [((1, 0), (3, 2)), ((1, 3), (2, 0))] # Debugging
    leg_swap_mode = 'MorphSym'
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
    path_to_save_csv = path_to_save_csv.replace(ckpt_name, ckpt_name + '-swap_legs={}-leg_swap_mode={}-{}'.format(swap_legs_list, leg_swap_mode, time_stamp))
    print("Results saving to: ", path_to_save_csv)

    # Initialize DataFrame outside the loop
    columns = ["Swap", "Model Accuracy", "Rear-Left", "Front-Left",	"Rear-Right", "Front-Right", "F1 Avg"]
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each symmetry
    for swap_legs in swap_legs_list:
        print("================================================")
        print(f"Leg Swap Mode: {leg_swap_mode}")
        if swap_legs:
            if isinstance(swap_legs[0], int):
                print(f"Swapping legs: {legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}")
                swap_str = f"{legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}"
            elif isinstance(swap_legs[0], tuple):
                print(f"Swapping legs: {legs_dict[swap_legs[0][0]]} and {legs_dict[swap_legs[0][1]]}, {legs_dict[swap_legs[1][0]]} and {legs_dict[swap_legs[1][1]]}")
                swap_str = f"{legs_dict[swap_legs[0][0]]} and {legs_dict[swap_legs[0][1]]}, {legs_dict[swap_legs[1][0]]} and {legs_dict[swap_legs[1][1]]}"
            else:
                raise ValueError("Invalid swap legs")
        else:
            print("No legs swapped")
            swap_str = "None"
        print(f"Model Type: {model_type}")
        print(f"Path to Checkpoint: {path_to_checkpoint}")
        print("================================================")

        air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, leg_swap_mode=leg_swap_mode)
        concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, leg_swap_mode=leg_swap_mode)
        concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, leg_swap_mode=leg_swap_mode)
        forest = linData.LinTzuYaunDataset_forest(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, leg_swap_mode=leg_swap_mode)
        small_pebble = linData.LinTzuYaunDataset_small_pebble(
            Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=True, swap_legs=swap_legs, leg_swap_mode=leg_swap_mode)
        test_dataset = torch.utils.data.ConcatDataset([air_jumping_gait, concrete_pronking, concrete_right_circle, forest, small_pebble])

        # Convert them to subsets
        test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

        # Evaluate with model
        pred, labels, acc, f1_leg_0, f1_leg_1, f1_leg_2, f1_leg_3, f1_avg_legs = evaluate_model(path_to_checkpoint, test_dataset)
        # Save to DataFrame
        df = pandas.concat([df, pandas.DataFrame([[swap_str, acc.item(), f1_leg_0.item(), f1_leg_1.item(), f1_leg_2.item(), f1_leg_3.item(), f1_avg_legs.item()]], columns=columns)], ignore_index=True)

        # Print the results
        print("Model Accuracy: ", acc.item())
        print("F1-Score Leg 0: ", f1_leg_0.item())
        print("F1-Score Leg 1: ", f1_leg_1.item())
        print("F1-Score Leg 2: ", f1_leg_2.item())
        print("F1-Score Leg 3: ", f1_leg_3.item())
        print("F1-Score Legs Avg: ", f1_avg_legs.item())

        if swap_legs is None and path_to_save_csv is not None:
            save_file_name = path_to_save_csv.split('/')[-1].replace('.csv', '')
            path_to_save_csv = path_to_save_csv.replace(save_file_name, save_file_name + '-acc={}-f1={}'.format(acc.item(), f1_avg_legs.item()))
            print("Note: Results saving path is changed to: ", path_to_save_csv)

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
