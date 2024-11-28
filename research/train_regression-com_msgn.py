from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.soloDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main(seed,
         batch_size=64,
         num_layers=4,
         hidden_size=128,
         lr=0.0024,
         epochs=30,
         logger_project_name='com_debug',
         model_type='heterogeneous_gnn_s4_com',
         wandb_api_key = "eed5fa86674230b63649180cc343f14e1f1ace78"):
    # ================================= CHANGE THESE ===================================
#     wandb_api_key = "eed5fa86674230b63649180cc343f14e1f1ace78"
    # ==================================================================================

    # Define model information
    history_length = 1
    normalize = True

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'Solo', 'solo12.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'Solo-12', 'solo12.urdf').absolute()

    root = Path(Path('.').parent, 'datasets', 'Solo-12').absolute()
    # Initalize the Train datasets
    solo12data_train = Solo12Dataset(root, path_to_urdf, 
                       'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize, stage='train')
    
    solo12data_val = Solo12Dataset(root, path_to_urdf, 
                       'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize, stage='val')
    
    solo12data_test = Solo12Dataset(root, path_to_urdf, 
                       'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize, stage='test')
    
    # Define train and val sets
    # train_val_datasets = [solo12data]

    # Take first 70% for training, and 15% for validation, and 15% for testing
    # Also remove the last entries, as dynamics models can't use last entry due to derivative calculation
    # train_subsets = []
    # val_subsets = []
    # for dataset in train_val_datasets:
    #     data_len_minus_1 = dataset.__len__() - 1
    #     split_index_train_val = int(np.round(data_len_minus_1 * 0.70)) # When value has .5, round to nearest-even
    #     split_index_test = int(np.round(data_len_minus_1 * 0.85)) # When value has .5, round to nearest-even
    #     train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, split_index_train_val)))
    #     val_subsets.append(torch.utils.data.Subset(dataset, np.arange(split_index_train_val, split_index_test)))
    
    
    train_dataset = torch.utils.data.Subset(solo12data_train, np.arange(0, solo12data_train.__len__()))
    val_dataset = torch.utils.data.Subset(solo12data_val, np.arange(0, solo12data_val.__len__()))
    
    # Train the model (evaluate later, so no test set)
    train_model(train_dataset, val_dataset, None, normalize, 
                num_layers=num_layers, hidden_size=hidden_size, logger_project_name=logger_project_name, 
                batch_size=batch_size, regression=True, lr=lr, epochs=epochs, seed=seed, devices=1, early_stopping=True,
                disable_test=True, 
                data_path = root, 
                subfoler_name=logger_project_name,
                wandb_api_key=wandb_api_key)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--lr', type=float, default=0.012, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=60, help='Number of epochs')
    # Logging parameters
    parser.add_argument('--logger_project_name', type=str, default='com_debug', help='Logger project name')
    # Model parameters
    parser.add_argument('--model_type', type=str, default='heterogeneous_gnn_s4_com', help='Model type, options: heterogeneous_gnn_s4_com, heterogeneous_gnn_k4_com')
    parser.add_argument('--wandb_api_key', type=str, default='eed5fa86674230b63649180cc343f14e1f1ace78', help="Check your key at https://wandb.ai/authorize",)
    args = parser.parse_args()

    print(f"args: {args}")
    
    main(seed=args.seed, 
         batch_size=args.batch_size, 
         num_layers=args.num_layers, 
         hidden_size=args.hidden_size, 
         lr=args.lr, 
         epochs=args.epochs,
         logger_project_name=args.logger_project_name,
         model_type=args.model_type,
         wandb_api_key=args.wandb_api_key)