from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.visualization import visualize_model_outputs_regression
import pandas
import os
import numpy as np
from datetime import datetime
import glob

def init_test_datasets_debug(a1Dataset, path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, symmetry_operator, symmetry_mode, group_operator_path):
    # ======================= Initalize the test datasets =======================
    # Unseen All (Friction, Speed, and Terrain)
    uniform = a1Dataset.QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])

    # Combine into one test set
    test_dataset = [uniform]
    for i, dataset in enumerate(test_dataset):
        test_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)

    # Keep track of all to evaluate
    to_evaluate = [test_dataset]
    dataset_names = ["Full"]

    return to_evaluate, dataset_names


def init_test_datasets(a1Dataset, path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension):
    # ======================= Initalize the test datasets =======================
    
    # Unseen Friction
    alpha = a1Dataset.QuadSDKDataset_A1_Alpha(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Alpha').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    delta = a1Dataset.QuadSDKDataset_A1_Delta(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Delta').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    india = a1Dataset.QuadSDKDataset_A1_India(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-India').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    lima = a1Dataset.QuadSDKDataset_A1_Lima(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Lima').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    unseen_friction_dataset = [alpha, delta, india, lima]
    for i, dataset in enumerate(unseen_friction_dataset): # Remove last entry, as dynamics can't use it.
        unseen_friction_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_friction_dataset = torch.utils.data.ConcatDataset(unseen_friction_dataset)
    
    # Unseen Speed
    quebec = a1Dataset.QuadSDKDataset_A1_Quebec(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Quebec').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    romeo = a1Dataset.QuadSDKDataset_A1_Romeo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Romeo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    sierra = a1Dataset.QuadSDKDataset_A1_Sierra(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Sierra').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    tango = a1Dataset.QuadSDKDataset_A1_Tango(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Tango').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    unseen_speed_dataset = [quebec, romeo, sierra, tango]
    for i, dataset in enumerate(unseen_speed_dataset):
        unseen_speed_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_speed_dataset = torch.utils.data.ConcatDataset(unseen_speed_dataset)
    
    # Unseen Terrain
    golf = a1Dataset.QuadSDKDataset_A1_Golf(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Golf').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    hotel = a1Dataset.QuadSDKDataset_A1_Hotel(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Hotel').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    oscar = a1Dataset.QuadSDKDataset_A1_Oscar(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Oscar').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    papa = a1Dataset.QuadSDKDataset_A1_Papa(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Papa').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    unseen_terrain_dataset = [golf, hotel, oscar, papa]
    for i, dataset in enumerate(unseen_terrain_dataset):
        unseen_terrain_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_terrain_dataset = torch.utils.data.ConcatDataset(unseen_terrain_dataset)

    # Unseen All (Friction, Speed, and Terrain)
    uniform = a1Dataset.QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])

    # Combine into one test set
    test_dataset = [alpha, delta, golf, hotel,
                    india, lima, oscar, papa,
                    quebec, romeo, sierra, tango, uniform]
    # test_dataset = [uniform]
    for i, dataset in enumerate(test_dataset):
        test_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)

    # Keep track of all to evaluate
    to_evaluate = [unseen_friction_dataset, unseen_speed_dataset, unseen_terrain_dataset, unseen_all_dataset, test_dataset]
    dataset_names = ["F", "S", "T", "A", "Full"]
    # to_evaluate = [test_dataset]
    # dataset_names = ["Full"]

    return to_evaluate, dataset_names

def find_models_to_test(path_to_models):
    # ======================= Find Models =======================
    all_model_dirs = os.listdir(path_to_models)
    final_models_to_test = []
    final_model_names = []

    # For each model
    for model_dir in all_model_dirs:

        # Get all checkpoints
        model_ckpts = os.listdir(path_to_models + model_dir + "/")

        # Search for the correct checkpoint
        ckpt_to_use = None
        curr_max = -1
        for model_ckpt in model_ckpts:
            # If v1, is the last checkpoint
            if "-v1.ckpt" in model_ckpt:
                ckpt_to_use = model_ckpt
                break
            
            # Else, use checkpoint with highest epoch num
            model_epoch_num = int(model_ckpt[6:model_ckpt.find("-")])
            if model_epoch_num > curr_max:
                curr_max = model_epoch_num
                ckpt_to_use = model_ckpt
        
        # Save the checkpoint at the end of training
        final_models_to_test.append(path_to_models + model_dir + "/" + ckpt_to_use)
        final_model_names.append(model_dir)

    return final_models_to_test, final_model_names

def main(MorphSym_version: str, 
         path_to_checkpoint=None,
         symmetry_operator_list=None,
         symmetry_mode=None, # Can be 'Euclidean' or 'MorphSym' or None
         group_operator_path=None,
         grf_body_to_world_frame=True,
         grf_dimension=3,
         test_only_on_z=False,
         batch_size: int = 100):

    if MorphSym_version == 'S4':
        import mi_hgnn.datasets_py.quadSDKDataset as a1Dataset
        model_type = 'heterogeneous_gnn'
        symmetry_mode = 'Euclidean'
        legs_dict = {0: 'RL', 1: 'FL', 2: 'RR', 3: 'FR'}
    elif MorphSym_version == 'C2':
        import mi_hgnn.datasets_py.quadSDKDataset_Morph as a1Dataset
        model_type = 'heterogeneous_gnn_c2'
        symmetry_mode = 'MorphSym'
    else:
        raise ValueError(f"MorphSym_version {MorphSym_version} not supported.")

    # Check that the user filled in the necessary parameters
    if path_to_checkpoint is None or model_type is None:
        raise ValueError("Please provide necessary parameters by editing this file!")
    print("===> Starting Evaluation on: ", path_to_checkpoint.split('/')[-2:])
    path_to_save_csv = path_to_checkpoint.replace('.ckpt', '.csv') # csv save location and file name
    ckpt_name = path_to_checkpoint.split('/')[-1].replace('.ckpt', '')
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # add the swap legs and leg swap mode to the path
    path_to_save_csv = path_to_save_csv.replace(ckpt_name, ckpt_name + '-symmetry_operator_list={}-symmetry_mode={}-test_only_on_z={}-{}'.format(symmetry_operator_list, symmetry_mode, test_only_on_z, time_stamp))
    print("===> Results saving to: ", path_to_save_csv)

    # Define model type
    history_length = 150
    normalize = False # TODO: check the normalization
    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Initialize DataFrame outside the loop
    columns = ["Swap"]
    # Get dataset names from first iteration to set up columns
    _, dataset_names = init_test_datasets(a1Dataset, path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, None, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)
    for name in dataset_names:
        columns.append(name + "-MSE")
        columns.append(name + "-RMSE")
        columns.append(name + "-L1")
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each symmetry
    for idx, symmetry_operator in enumerate(symmetry_operator_list):
        print("================================================")
        if symmetry_mode == 'Euclidean':
            print(f"Swapping legs: {legs_dict[symmetry_operator[0]]} and {legs_dict[symmetry_operator[1]]}")
            swap_str = f"{legs_dict[symmetry_operator[0]]} and {legs_dict[symmetry_operator[1]]}"
        elif symmetry_mode == 'MorphSym':
            print(f"Using symmetry operator: {symmetry_operator}")
            swap_str = f"{symmetry_operator}"
        else:
            print("No symmetry operator")
            swap_str = "None"
        print("================================================")

        # Initialize test datasets
        to_evaluate, dataset_names = init_test_datasets(a1Dataset, path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, symmetry_operator, symmetry_mode, group_operator_path, grf_body_to_world_frame, grf_dimension)

        # ======================= Evaluation =======================    
        results = [swap_str]
        for idx, dataset in enumerate(to_evaluate):
            print(f"Evaluating {dataset_names[idx]} ...")
            pred, labels, mse, rmse, l1 = evaluate_model(
                path_to_checkpoint, 
                torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__())),
                symmetry_mode=symmetry_mode,
                group_operator_path=group_operator_path,
                grf_body_to_world_frame=grf_body_to_world_frame,
                grf_dimension=grf_dimension,
                batch_size=batch_size,
                test_only_on_z=test_only_on_z)
            results.append(mse.item())
            results.append(rmse.item())
            results.append(l1.item())
        df = pandas.concat([df, pandas.DataFrame([results], columns=df.columns)], ignore_index=True)
        print(f"Finished Evaluating: Swap Operation[{swap_str}], MSE: {mse.item()}, RMSE: {rmse.item()}, L1: {l1.item()}")

    # Save csv
    if path_to_save_csv is not None:
        df.to_csv(path_to_save_csv, index=False)
    else:
        df.to_csv("regression_results.csv", index=False)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--group_operator_path', type=str, default='cfg/a1-c2.yaml', help='Path to group operator')
    parser.add_argument('--symmetry_mode', type=str, default='MorphSym', help='Symmetry mode')
    parser.add_argument('--grf_body_to_world_frame', type=bool, default=False, help='GRF body to world frame')
    parser.add_argument('--test_only_on_z', type=bool, default=False, help='Test only on z')
    parser.add_argument('--grf_dimension', type=int, default=1, help='Dimension of GRF')

    args = parser.parse_args()

    print(args)

    batch_size = 100
    path_to_checkpoint = args.path_to_checkpoint
    group_operator_path = args.group_operator_path
    symmetry_mode = args.symmetry_mode
    grf_body_to_world_frame = args.grf_body_to_world_frame
    grf_dimension = args.grf_dimension
    test_only_on_z = args.test_only_on_z

    MorphSym_version = 'C2'
    symmetry_operator_list = [None]  # Can be 'gs' or 'gt' or 'gr' or None

    if os.path.isdir(path_to_checkpoint):
        checkpoint_files = glob.glob(os.path.join(path_to_checkpoint, "*.ckpt"))
        # Sort checkpoint files by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
        for file in checkpoint_files:
            main(MorphSym_version=MorphSym_version, 
                 path_to_checkpoint=file, 
                 symmetry_operator_list=symmetry_operator_list, 
                 symmetry_mode=symmetry_mode, 
                 group_operator_path=group_operator_path,
                 grf_body_to_world_frame=grf_body_to_world_frame,
                 grf_dimension=grf_dimension,
                 batch_size=batch_size,
                 test_only_on_z=test_only_on_z)
    else:
        main(MorphSym_version=MorphSym_version, 
             path_to_checkpoint=path_to_checkpoint, 
             symmetry_operator_list=symmetry_operator_list, 
             symmetry_mode=symmetry_mode, 
             group_operator_path=group_operator_path,
             grf_body_to_world_frame=grf_body_to_world_frame,
             grf_dimension=grf_dimension,
             batch_size=batch_size,
             test_only_on_z=test_only_on_z)