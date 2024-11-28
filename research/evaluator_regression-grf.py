from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression
import pandas
import os
import glob
from datetime import datetime

def main(path_to_checkpoint: str,
         model_type: str = 'heterogeneous_gnn',
         grf_body_to_world_frame: bool = True,
         grf_dimension: int = 3):
    
    print("===> Starting Evaluation on: ", path_to_checkpoint.split('/')[-2:])
    path_to_save_csv = path_to_checkpoint.replace('.ckpt', '.csv') # csv save location and file name
    ckpt_name = path_to_checkpoint.split('/')[-1].replace('.ckpt', '')
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # add the swap legs and leg swap mode to the path
    path_to_save_csv = path_to_save_csv.replace(ckpt_name, ckpt_name + f'{time_stamp}')
    print("===> Results saving to: ", path_to_save_csv)

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Define model type
    history_length = 150
    normalize = False

    # ======================= Initalize the test datasets =======================
    # Unseen Friction
    alpha = QuadSDKDataset_A1_Alpha(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Alpha').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    delta = QuadSDKDataset_A1_Delta(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Delta').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    india = QuadSDKDataset_A1_India(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-India').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    lima = QuadSDKDataset_A1_Lima(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Lima').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    unseen_friction_dataset = [alpha, delta, india, lima]
    for i, dataset in enumerate(unseen_friction_dataset): # Remove last entry, as dynamics can't use it.
        unseen_friction_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_friction_dataset = torch.utils.data.ConcatDataset(unseen_friction_dataset)
    
    # Unseen Speed
    quebec = QuadSDKDataset_A1_Quebec(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Quebec').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    romeo = QuadSDKDataset_A1_Romeo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Romeo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    sierra = QuadSDKDataset_A1_Sierra(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Sierra').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    tango = QuadSDKDataset_A1_Tango(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Tango').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    unseen_speed_dataset = [quebec, romeo, sierra, tango]
    for i, dataset in enumerate(unseen_speed_dataset):
        unseen_speed_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_speed_dataset = torch.utils.data.ConcatDataset(unseen_speed_dataset)
    
    # Unseen Terrain
    golf = QuadSDKDataset_A1_Golf(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Golf').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    hotel = QuadSDKDataset_A1_Hotel(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Hotel').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    oscar = QuadSDKDataset_A1_Oscar(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Oscar').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    papa = QuadSDKDataset_A1_Papa(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Papa').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    unseen_terrain_dataset = [golf, hotel, oscar, papa]
    for i, dataset in enumerate(unseen_terrain_dataset):
        unseen_terrain_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    unseen_terrain_dataset = torch.utils.data.ConcatDataset(unseen_terrain_dataset)

    # Unseen All (Friction, Speed, and Terrain)
    uniform = QuadSDKDataset_A1_Uniform(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Uniform').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    unseen_all_dataset = torch.utils.data.ConcatDataset([torch.utils.data.Subset(uniform, np.arange(0, uniform.__len__() - 1))])

    # Combine into one test set
    test_dataset = [alpha, delta, golf, hotel,
                    india, lima, oscar, papa,
                    quebec, romeo, sierra, tango, uniform]
    for i, dataset in enumerate(test_dataset):
        test_dataset[i] = torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__() - 1))
    test_dataset = torch.utils.data.ConcatDataset(test_dataset)

    # Keep track of all to evaluate
    to_evaluate = [unseen_friction_dataset, unseen_speed_dataset, unseen_terrain_dataset, unseen_all_dataset, test_dataset]
    dataset_names = ["F", "S", "T", "A", "Full"]

    # New Code ======================================================================
    # ======================= Evaluation =======================
    # Create new Dataframe
    columns = ["Model"]
    for name in dataset_names:
        columns.append(name + "-MSE")
        columns.append(name + "-RMSE")
        columns.append(name + "-L1")
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each model
    # Evaluate and save to Dataframe
    results = [path_to_checkpoint.split('/')[-2:]]
    for dataset in to_evaluate:
        pred, labels, mse, rmse, l1 = evaluate_model(path_to_checkpoint, torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__())))
        results.append(mse.item())
        results.append(rmse.item())
        results.append(l1.item())
    df = pandas.concat([df, pandas.DataFrame([results], columns=df.columns)], ignore_index=True)
    print(f"Finished Evaluating:, Model: {path_to_checkpoint.split('/')[-2:]},  MSE: {mse.item()}, RMSE: {rmse.item()}, L1: {l1.item()}")

    # Save csv
    if path_to_save_csv is not None:
        df.to_csv(path_to_save_csv, index=False)
    else:
        df.to_csv("regression_results.csv", index=False)

if __name__ == '__main__':
    batch_size = 100
    path_to_checkpoint = "models/grf_baseline_mihgnn_d3/rose-firefly-1/"
    grf_body_to_world_frame = True
    grf_dimension = 3

    if os.path.isdir(path_to_checkpoint):
        checkpoint_files = glob.glob(os.path.join(path_to_checkpoint, "*.ckpt"))
        # Sort checkpoint files by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
        for file in checkpoint_files:
            main(path_to_checkpoint=file, 
                 grf_body_to_world_frame=grf_body_to_world_frame,
                 grf_dimension=grf_dimension)
    else:
        main(path_to_checkpoint=path_to_checkpoint, 
             grf_body_to_world_frame=grf_body_to_world_frame,
             grf_dimension=grf_dimension)


    # Old Code ======================================================================

    # # ======================= Find Models =======================
    # all_model_dirs = os.listdir(path_to_models)
    # final_models_to_test = []
    # final_model_names = []

    # For each model
    # for model_dir in all_model_dirs:

    #     # Get all checkpoints
    #     model_ckpts = os.listdir(path_to_models + model_dir + "/")

    #     # Search for the correct checkpoint
    #     ckpt_to_use = None
    #     curr_max = -1
    #     for model_ckpt in model_ckpts:
    #         # If v1, is the last checkpoint
    #         if "-v1.ckpt" in model_ckpt:
    #             ckpt_to_use = model_ckpt
    #             break
            
    #         # Else, use checkpoint with highest epoch num
    #         model_epoch_num = int(model_ckpt[6:model_ckpt.find("-")])
    #         if model_epoch_num > curr_max:
    #             curr_max = model_epoch_num
    #             ckpt_to_use = model_ckpt
        
    #     # Save the checkpoint at the end of training
    #     final_models_to_test.append(path_to_models + model_dir + "/" + ckpt_to_use)
    #     final_model_names.append(model_dir)

    # # ======================= Evaluation =======================
    # # Create new Dataframe
    # columns = ["Model"]
    # for name in dataset_names:
    #     columns.append(name + "-MSE")
    #     columns.append(name + "-RMSE")
    #     columns.append(name + "-L1")
    # df = pandas.DataFrame(None, columns=columns)

    # # Evaluate each model
    # for i, model in enumerate(final_models_to_test):
    #     # Evaluate and save to Dataframe
    #     results = [final_model_names[i]]
    #     for dataset in to_evaluate:
    #         pred, labels, mse, rmse, l1 = evaluate_model(model, torch.utils.data.Subset(dataset, np.arange(0, dataset.__len__())))
    #         results.append(mse.item())
    #         results.append(rmse.item())
    #         results.append(l1.item())
    #     df = pandas.concat([df, pandas.DataFrame([results], columns=df.columns)], ignore_index=True)
    #     print("Finished Evaluating ", final_model_names[i])

    # # Save csv
    # if path_to_save_csv is not None:
    #     df.to_csv(path_to_save_csv, index=False)
    # else:
    #     df.to_csv("regression_results.csv", index=False)
