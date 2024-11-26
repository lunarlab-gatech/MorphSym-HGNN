from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.visualization import visualize_model_outputs_regression
import numpy as np

def main(seed,
         batch_size=32,
         num_layers=8,
         hidden_size=128,
         lr=0.0001,
         epochs=30,
         logger_project_name='grf_c2_debug',
         model_type='heterogeneous_gnn_c2'):
    # ================================= CHANGE THESE ===================================
    wandb_api_key = "eed5fa86674230b63649180cc343f14e1f1ace78"
    # ==================================================================================

    # Define model information
    history_length = 150
    normalize = False

    if model_type == 'heterogeneous_gnn_c2':
        import mi_hgnn.datasets_py.quadSDKDataset_Morph as QuadSDKDataset
    else:
        raise ValueError(f"model_type {model_type} not supported.")

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Initalize the Train datasets
    bravo = QuadSDKDataset.QuadSDKDataset_A1_Bravo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Bravo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    charlie = QuadSDKDataset.QuadSDKDataset_A1_Charlie(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Charlie').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    echo = QuadSDKDataset.QuadSDKDataset_A1_Echo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Echo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    foxtrot = QuadSDKDataset.QuadSDKDataset_A1_Foxtrot(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Foxtrot').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    juliett = QuadSDKDataset.QuadSDKDataset_A1_Juliett(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Juliett').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    kilo = QuadSDKDataset.QuadSDKDataset_A1_Kilo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Kilo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    mike = QuadSDKDataset.QuadSDKDataset_A1_Mike(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Mike').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    november = QuadSDKDataset.QuadSDKDataset_A1_November(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-November').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics)
    
    # Define train and val sets
    train_val_datasets = [bravo, charlie, echo, foxtrot, juliett, kilo, mike, november]

    # Take first 85% for training, and last 15% for validation
    # Also remove the last entries, as dynamics models can't use last entry due to derivative calculation
    train_subsets = []
    val_subsets = []
    for dataset in train_val_datasets:
        data_len_minus_1 = dataset.__len__() - 1
        split_index = int(np.round(data_len_minus_1 * 0.85)) # When value has .5, round to nearest-even
        train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, split_index)))
        val_subsets.append(torch.utils.data.Subset(dataset, np.arange(split_index, data_len_minus_1)))
    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)
    
    # Convert them to subsets
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, train_dataset.__len__()))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, val_dataset.__len__()))
    
    # Train the model (evaluate later, so no test set)
    train_model(train_dataset, val_dataset, None, normalize, 
                num_layers=num_layers, hidden_size=hidden_size, logger_project_name=logger_project_name, 
                batch_size=batch_size, regression=True, lr=lr, epochs=epochs, seed=seed, devices=1, early_stopping=True,
                disable_test=True,
                subfoler_name=logger_project_name,
                wandb_api_key=wandb_api_key)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    # Logging parameters
    parser.add_argument('--logger_project_name', type=str, default='grf_c2_debug', help='Logger project name')
    # Model parameters
    parser.add_argument('--model_type', type=str, default='heterogeneous_gnn_c2', help='Model type, options: heterogeneous_gnn_c2')
    args = parser.parse_args()

    print(f"args: {args}")
    
    main(seed=args.seed, 
         batch_size=args.batch_size, 
         num_layers=args.num_layers, 
         hidden_size=args.hidden_size, 
         lr=args.lr, 
         epochs=args.epochs,
         logger_project_name=args.logger_project_name,
         model_type=args.model_type)