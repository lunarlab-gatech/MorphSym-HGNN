import os
import glob
from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import evaluate_model
import torch
import numpy as np
import pandas
from datetime import datetime

def main(MorphSym_version: str, 
         path_to_checkpoint=None,
         symmetry_operator_list=None,
         symmetry_mode=None,
         group_operator_path=None):
    # ================================= CHANGE THIS ====================================
    if MorphSym_version == 'K4':
        from mi_hgnn.datasets_py.soloDataset import Solo12Dataset
        model_type = 'heterogeneous_gnn_k4_com'
    elif MorphSym_version == 'S4':
        from mi_hgnn.datasets_py.soloDataset import Solo12Dataset
        model_type = 'heterogeneous_gnn_s4_com'
    else:
        raise ValueError("Other MorphSym versions are not supported for this script yet!")
    
    # ==================================================================================
    
    # print the path to checkpoint
    print("===> Starting Evaluation on: ", path_to_checkpoint.split('/')[-2:])

    # Set parameters
    history_length = 1
    path_to_urdf = Path('urdf_files', 'Solo', 'solo12.urdf').absolute()

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
    columns = ["Symmetry Operator", "Test Loss", "Test Cos Sim Lin", "Test Cos Sim Ang"]
    df = pandas.DataFrame(None, columns=columns)

    # Evaluate each symmetry
    # for swap_legs in swap_legs_list:
    for symmetry_operator in symmetry_operator_list:
        swap_legs = None
        print("================================================")
        print(f"Symmetry Operator: {symmetry_operator}")
        print(f"Model Type: {model_type}")
        print(f"Path to Checkpoint: {path_to_checkpoint}")
        print("================================================")
        
        root = Path(Path('.').parent, 'datasets', 'Solo-12').absolute()
        test_dataset = prepare_test_dataset(Solo12Dataset, root, path_to_urdf, model_type, history_length, normalize=True, 
                                                symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
        
        # Evaluate with model
        pred, labels, loss, cos_sim_lin, cos_sim_ang = evaluate_model(path_to_checkpoint, test_dataset, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path, data_path = root)
        # Save to DataFrame
        df = pandas.concat([df, pandas.DataFrame([[symmetry_operator, loss.item(), cos_sim_lin.item(), cos_sim_ang.item()]], columns=columns)], ignore_index=True)

        # Print the results
        print_results(loss, cos_sim_lin, cos_sim_ang)

    # Save to csv
    df.to_csv(path_to_save_csv, index=False)
    print("===> Evaluation Finished! Results saved to: ", path_to_save_csv)

def prepare_test_dataset(Solo12Dataset, root, path_to_urdf, model_type, history_length, normalize=True, symmetry_operator=None, symmetry_mode=None, group_operator_path=None):
    """Prepare the test dataset"""
    # Define train and val sets
    solo12data_test = Solo12Dataset(root, path_to_urdf, 
                       'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize, stage='test', 
                       symmetry_operator=symmetry_operator, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)

    test_dataset = torch.utils.data.Subset(solo12data_test, np.arange(0, solo12data_test.__len__()))
    
    return test_dataset

def print_results(loss, cos_sim_lin, cos_sim_ang):
    """Print the results"""
    print("Test Loss: ", loss.item())
    print("Test Cos Sim Lin: ", cos_sim_lin.item())
    print("Test Cos Sim Ang: ", cos_sim_ang.item())

if __name__ == "__main__":
    # K4
    MorphSym_version = 'K4'
    path_to_checkpoint = "Your Checkpoint File Path"
    group_operator_path = "Your Config File Path"
    symmetry_operator_list = [None]  # Can be 'gs' or 'gt' or 'gr' or None
    symmetry_mode = 'MorphSym' # Can be 'Euclidean' or 'MorphSym' or None

    if os.path.isdir(path_to_checkpoint):
        checkpoint_files = glob.glob(os.path.join(path_to_checkpoint, "*.ckpt"))
        # Sort checkpoint files by epoch number
        checkpoint_files.sort(key=lambda x: int(x.split('epoch=')[1].split('-')[0]))
        for file in checkpoint_files:
            main(MorphSym_version=MorphSym_version, path_to_checkpoint=file, symmetry_operator_list=symmetry_operator_list, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)
    else:
        main(MorphSym_version=MorphSym_version, path_to_checkpoint=path_to_checkpoint, symmetry_operator_list=symmetry_operator_list, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path)