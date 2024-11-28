from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.quadSDKDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main(seed,
         batch_size=32,
         num_layers=8,
         hidden_size=128,
         lr=0.0001,
         epochs=30,
         logger_project_name='grf_baseline_mihgnn_d3',
         model_type='heterogeneous_gnn',
         grf_body_to_world_frame=True,
         grf_dimension=3):
    # ==================================================================================

    # Define model information
    history_length = 150
    normalize = False

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'A1-Quad', 'a1_pruned.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'A1-Quad', 'a1.urdf').absolute()

    # Initalize the Train datasets
    bravo = QuadSDKDataset_A1_Bravo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Bravo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    charlie = QuadSDKDataset_A1_Charlie(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Charlie').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension   )
    echo = QuadSDKDataset_A1_Echo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Echo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    foxtrot = QuadSDKDataset_A1_Foxtrot(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Foxtrot').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    juliett = QuadSDKDataset_A1_Juliett(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Juliett').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    kilo = QuadSDKDataset_A1_Kilo(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Kilo').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    mike = QuadSDKDataset_A1_Mike(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-Mike').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    november = QuadSDKDataset_A1_November(Path(Path('.').parent, 'datasets', 'QuadSDK-A1-November').absolute(), path_to_urdf, 
                'package://a1_description/', '', model_type, history_length, normalize, path_to_urdf_dynamics, grf_body_to_world_frame, grf_dimension)
    
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
                grf_body_to_world_frame=grf_body_to_world_frame,
                grf_dimension=grf_dimension)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--hidden_size', type=int, default=128, help='Hidden size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    # Logging parameters
    parser.add_argument('--logger_project_name', type=str, default='grf_baseline_mihgnn_d3', help='Logger project name') # TODO: change
    # Model parameters
    parser.add_argument('--model_type', type=str, default='heterogeneous_gnn', help='Model type, options: heterogeneous_gnn')
    parser.add_argument('--grf_body_to_world_frame', type=bool, default=True, help='Whether to convert GRF to world frame') # TODO: change
    parser.add_argument('--grf_dimension', type=int, default=3, help='Dimension of GRF') # TODO: change
    args = parser.parse_args()
    
    main(seed=args.seed,
         batch_size=args.batch_size,
         num_layers=args.num_layers,
         hidden_size=args.hidden_size,
         lr=args.lr,
         epochs=args.epochs,
         logger_project_name=args.logger_project_name,
         model_type=args.model_type,
         grf_body_to_world_frame=args.grf_body_to_world_frame,
         grf_dimension=args.grf_dimension)