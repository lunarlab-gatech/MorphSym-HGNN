from pathlib import Path
from torch_geometric.loader import DataLoader
import numpy as np
import torch
from ms_hgnn.lightning_py.gnnLightning import train_model
import glob
import os

def find_latest_ckpt(ckpt_dir):
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
    if len(ckpt_files) == 0:
        return None
    return sorted(ckpt_files, key=lambda x: os.path.getmtime(x))[-1]

def main(train_percentage: float,
         val_percentage: float,
         seed: int = 42,
         batch_size: int = 64,
         num_layers: int = 8,
         hidden_size: int = 128,
         lr: float = 0.0001,
         epochs: int = 49,
         logger_project_name: str = 'main_cls_k4',
         model_type: str = 'heterogeneous_gnn_k4',
         symmetry_mode: str = 'MorphSym',
         group_operator_path: str = 'cfg/mini_cheetah-k4.yaml',
         ckpt_dir: str = None):
    """
    Duplicate the experiment found in Section VI-B of "On discrete symmetries 
    of robotics systems: A group-theoretic and data-driven analysis", but training
    on our HGNN instead.
    
    Here, we can vary the train and validation percentages to test sample efficiency.
    """
    
    print(f"model_type: {model_type}")
    wandb_api_key = "eed5fa86674230b63649180cc343f14e1f1ace78"

    # Set model parameters (so they all match)
    history_length = 150
    normalize = True

    if model_type == 'heterogeneous_gnn_k4' or model_type == 'heterogeneous_gnn_c2':
        import ms_hgnn.datasets_py.LinTzuYaunDataset_Morph as linData
    else:
        import ms_hgnn.datasets_py.LinTzuYaunDataset as linData

    # Initialize the Training/Validation datasets
    path_to_urdf = Path('urdf_files', 'MiniCheetah', 'miniCheetah.urdf').absolute()
    air_walking_gait = linData.LinTzuYaunDataset_air_walking_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AWG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    grass = linData.LinTzuYaunDataset_grass(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-G').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    middle_pebble = linData.LinTzuYaunDataset_middle_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-MP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_left_circle = linData.LinTzuYaunDataset_concrete_left_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CLC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_difficult_slippery = linData.LinTzuYaunDataset_concrete_difficult_slippery(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CDS').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    asphalt_road = linData.LinTzuYaunDataset_asphalt_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    old_asphalt_road = linData.LinTzuYaunDataset_old_asphalt_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-OAR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_galloping = linData.LinTzuYaunDataset_concrete_galloping(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    rock_road = linData.LinTzuYaunDataset_rock_road(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-RR').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    sidewalk = linData.LinTzuYaunDataset_sidewalk(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-S').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    train_val_datasets = [air_walking_gait, grass, middle_pebble, concrete_left_circle, concrete_difficult_slippery, asphalt_road, old_asphalt_road, concrete_galloping, rock_road, sidewalk]

    train_subsets = []
    val_subsets = []
    for dataset in train_val_datasets:
        train_index = int(np.round(dataset.__len__() * train_percentage)) # When value has .5, round to nearest-even
        val_index = dataset.len() - int(np.round(dataset.__len__() * val_percentage))
        train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, train_index)))
        val_subsets.append(torch.utils.data.Subset(dataset, np.arange(val_index, dataset.len())))
    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)

    # Initialize the Testing datasets
    concrete_pronking = linData.LinTzuYaunDataset_concrete_pronking(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    concrete_right_circle = linData.LinTzuYaunDataset_concrete_right_circle(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-CRC').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    small_pebble = linData.LinTzuYaunDataset_small_pebble(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-SP').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    air_jumping_gait = linData.LinTzuYaunDataset_air_jumping_gait(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-AJG').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    forest = linData.LinTzuYaunDataset_forest(
        Path(Path('.').parent, 'datasets', 'LinTzuYaun-F').absolute(), path_to_urdf, 'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize=normalize)
    test_dataset = torch.utils.data.ConcatDataset([concrete_pronking, concrete_right_circle, small_pebble, air_jumping_gait, forest])

    # Convert them to subsets
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, train_dataset.__len__()))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, val_dataset.__len__()))
    test_dataset = torch.utils.data.Subset(test_dataset, np.arange(0, test_dataset.__len__()))

	# Test cases to make sure this matches MorphoSymm-Replication can be found in the corresponding 
    # MorphoSymm-Replication release for this sample efficiency experiment.

    # Find the latest checkpoint
    if ckpt_dir is not None:
        ckpt_path = find_latest_ckpt(ckpt_dir)
    else:
        ckpt_path = None

    # Train the model
    train_model(train_dataset, val_dataset, test_dataset, normalize, num_layers=num_layers, hidden_size=hidden_size, 
                logger_project_name=logger_project_name, batch_size=batch_size, regression=False, lr=lr, epochs=epochs, 
                seed=seed, devices=1, early_stopping=True, train_percentage_to_log=train_percentage, symmetry_mode=symmetry_mode, group_operator_path=group_operator_path, subfoler_name=logger_project_name, wandb_api_key=wandb_api_key, ckpt_path=ckpt_path)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Sample efficiency parameters
    parser.add_argument('--train_percentage', type=float, default=0.15, help='Train percentage')
    parser.add_argument('--val_percentage', type=float, default=0.15, help='Validation percentage')
    # Model parameters
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=49, help='Number of epochs')
    # Logging parameters
    parser.add_argument('--logger_project_name', type=str, default='main_cls_k4_sample_eff', help='Logger project name')
	# Symmetry parameters
    parser.add_argument('--model_type', type=str, default='heterogeneous_gnn_k4', help='Model type, options: heterogeneous_gnn_k4, mlp, heterogeneous_gnn, heterogeneous_gnn_k4, heterogeneous_gnn_c2')
    parser.add_argument('--symmetry_mode', type=str, default='MorphSym', help='Symmetry mode, options: Euclidean, MorphSym, None')
    parser.add_argument('--group_operator_path', type=str, default='cfg/mini_cheetah-k4.yaml', help='Group operator path')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='Checkpoint directory')
    args = parser.parse_args()
    
    print(f"args: {args}")
    
    main(
        train_percentage=args.train_percentage,
        val_percentage=args.val_percentage,
        seed=args.seed, 
        batch_size=args.batch_size, 
        num_layers=args.num_layers, 
        hidden_size=args.hidden_size, 
        lr=args.lr, 
        epochs=args.epochs, 
        logger_project_name=args.logger_project_name, 
        model_type=args.model_type, 
        symmetry_mode=args.symmetry_mode, 
        group_operator_path=args.group_operator_path, 
        ckpt_dir=args.ckpt_dir)