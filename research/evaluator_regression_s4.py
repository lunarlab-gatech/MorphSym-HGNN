from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression
import pandas
import os

def init_test_datasets(path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, swap_legs):
    # ======================= Initalize the test datasets =======================
    # Unseen Friction
    # alpha = QuadSDKDataset_A1_Alpha(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Alpha').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # delta = QuadSDKDataset_A1_Delta(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Delta').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # india = QuadSDKDataset_A1_India(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-India').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # lima = QuadSDKDataset_A1_Lima(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Lima').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # unseen_friction_dataset = [alpha, delta, india, lima]
    # for i, dataset in enumerate(unseen_friction_dataset): # Remove last entry, as dynamics can't use it.
    #     unseen_friction_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    # unseen_friction_dataset = torch.utils.data.ConcatDataset(unseen_friction_dataset)
    
    # Unseen Speed
    # quebec = QuadSDKDataset_A1_Quebec(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Quebec').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # romeo = QuadSDKDataset_A1_Romeo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Romeo').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # sierra = QuadSDKDataset_A1_Sierra(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Sierra').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # tango = QuadSDKDataset_A1_Tango(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Tango').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # unseen_speed_dataset = [quebec, romeo, sierra, tango]
    # for i, dataset in enumerate(unseen_speed_dataset):
    #     unseen_speed_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    # unseen_speed_dataset = torch.utils.data.ConcatDataset(unseen_speed_dataset)
    
    # Unseen Terrain
    # golf = QuadSDKDataset_A1_Golf(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Golf').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # hotel = QuadSDKDataset_A1_Hotel(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Hotel').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # oscar = QuadSDKDataset_A1_Oscar(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Oscar').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # papa = QuadSDKDataset_A1_Papa(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Papa').absolute(), path_to_urdf, 
    #             'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    # unseen_terrain_dataset = [golf, hotel, oscar, papa]
    # for i, dataset in enumerate(unseen_terrain_dataset):
    #     unseen_terrain_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    # unseen_terrain_dataset = torch.utils.data.ConcatDataset(unseen_terrain_dataset)

    # Unseen All (Friction, Speed, and Terrain)
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, swap_legs=swap_legs)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])

    # Combine into one test set
    # test_dataset = [alpha, delta, golf, hotel,
    #                 india, lima, oscar, papa,
    #                 quebec, romeo, sierra, tango, uniform]
    test_dataset = [uniform]
    for i, dataset in enumerate(test_dataset):
        test_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)

    # Keep track of all to evaluate
    # to_evaluate = [unseen_friction_dataset, unseen_speed_dataset, unseen_terrain_dataset, unseen_all_dataset, test_dataset]
    # dataset_names = ["F", "S", "T", "A", "Full"]
    to_evaluate = [test_dataset]
    dataset_names = ["Full"]

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

def main():
    # ================================= CHANGE THESE ===================================
    path_to_save_csv = '/home/swei303/Documents/proj/MorphSym-HGNN/paper/regression_results_hgnns_s4.csv' # csv save location and file name
    path_to_models = '/home/swei303/Documents/proj/MorphSym-HGNN/ckpts/Regression Experiment/hgnns/' # Folder with all models of same type (like the hgnns folder on Dropbox)
    model_type = 'heterogeneous_gnn' # 'heterogeneous_gnn' or 'mlp'
    verify_symmetry = True # Whether to verify symmetry of the model
    if verify_symmetry:
        final_models_to_test = []
        final_model_names = []
        model_path = "/home/swei303/Documents/proj/MorphSym-HGNN/ckpts/Regression Experiment/hgnns/twilight-bush-24/epoch=22-val_MSE_loss=78.51031.ckpt"
        final_models_to_test.append(model_path)
        final_model_names.append("twilight-bush-24")
        swap_legs_list = [None, (1, 3), (1, 0), (1, 2)] # Swap tuple: FR: 0, FL: 1, RR: 2, RL: 3, None: no swap
        legs_dict = {0: 'RL', 1: 'FL', 2: 'RR', 3: 'FR'}
    # ==================================================================================

    # Check that the user filled in the necessary parameters
    if path_to_models is None or model_type is None:
        raise ValueError("Please provide necessary parameters by editing this file!")

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Define model type
    history_length = 150
    normalize = False

    # Initialize DataFrame outside the loop
    columns = ["Swap", "Model"]
    # Get dataset names from first iteration to set up columns
    _, dataset_names = init_test_datasets(path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, swap_legs_list[0])
    for name in dataset_names:
        columns.append(name + "-MSE")
        columns.append(name + "-RMSE")
        columns.append(name + "-L1")
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each symmetry
    for swap_legs in swap_legs_list:
        print("================================================")
        if swap_legs:
            print(f"Swapping legs: {legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}")
        else:
            print("No legs swapped")
        print("================================================")

        # Initialize test datasets
        to_evaluate, _ = init_test_datasets(path_to_urdf, path_to_urdf_dynamics, model_type, history_length, normalize, swap_legs)

        # Find models to test
        if verify_symmetry: # Only test one model
            final_models_to_test = ["/home/swei303/Documents/proj/MorphSym-HGNN/ckpts/Regression Experiment/hgnns/twilight-bush-24/epoch=22-val_MSE_loss=78.51031.ckpt"]
            final_model_names = ['twilight-bush-24']
        else:
            final_models_to_test, final_model_names = find_models_to_test(path_to_models)

        # ======================= Evaluation =======================    
        # Evaluate each model
        for i, model in enumerate(final_models_to_test):
            # Evaluate and save to Dataframe
            if swap_legs:
                swap_str = f"{legs_dict[swap_legs[0]]} and {legs_dict[swap_legs[1]]}"
            else:
                swap_str = "None"
            results = [swap_str, final_model_names[i]]
            for dataset in to_evaluate:
                pred, labels, mse, rmse, l1 = evaluate_model(model, torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__())))
                results.append(mse.item())
                results.append(rmse.item())
                results.append(l1.item())
            df = pandas.concat([df, pandas.DataFrame([results], columns=df.columns)], ignore_index=True)
            print("Finished Evaluating ", final_model_names[i])

    # Save csv
    if path_to_save_csv is not None:
        df.to_csv(path_to_save_csv, index=False)
    else:
        df.to_csv("regression_results.csv", index=False)

if __name__ == '__main__':
     main()