from pathlib import Path
from mi_hgnn.lightning_py.gnnLightning import train_model, evaluate_model
import torch
from mi_hgnn.datasets_py.soloDataset import *
from mi_hgnn.visualization import visualize_model_outputs_regression

def main():
    # ================================= CHANGE THESE ===================================
    model_type = 'heterogeneous_gnn_k4_com' # `mlp`
    seed = 0
    # ==================================================================================

    # Define model information
    history_length = 150
    normalize = False
    num_layers = 8
    hidden_size = 128

    # Set up the urdf paths
    path_to_urdf = Path('urdf_files', 'Solo', 'solo12.urdf').absolute()
    path_to_urdf_dynamics = Path('urdf_files', 'Solo-12', 'solo12.urdf').absolute()

    # Initalize the Train datasets
    solo12data = Solo12Dataset(Path(Path('.').parent, 'datasets', 'Solo-12').absolute(), path_to_urdf, 
                       'package://yobotics_description/', 'mini-cheetah-gazebo-urdf/yobo_model/yobotics_description', model_type, history_length, normalize)
    
    # Define train and val sets
    train_val_datasets = [solo12data]

    # Take first 85% for training, and last 15% for validation
    # Also remove the last entries, as dynamics models can't use last entry due to derivative calculation
    train_subsets = []
    val_subsets = []
    for dataset in train_val_datasets:
        data_len_minus_1 = dataset.__len__() - 1
        split_index = int(np.round(data_len_minus_1 * 0.70)) # When value has .5, round to nearest-even
        train_subsets.append(torch.utils.data.Subset(dataset, np.arange(0, split_index)))
        val_subsets.append(torch.utils.data.Subset(dataset, np.arange(split_index, data_len_minus_1)))
    train_dataset = torch.utils.data.ConcatDataset(train_subsets)
    val_dataset = torch.utils.data.ConcatDataset(val_subsets)
    
    # Convert them to subsets
    train_dataset = torch.utils.data.Subset(train_dataset, np.arange(0, train_dataset.__len__()))
    val_dataset = torch.utils.data.Subset(val_dataset, np.arange(0, val_dataset.__len__()))
    
    # Train the model (evaluate later, so no test set)
    train_model(train_dataset, val_dataset, None, normalize, 
                num_layers=num_layers, hidden_size=hidden_size, logger_project_name="regression", 
                batch_size=30, regression=True, lr=0.0001, epochs=30, seed=seed, devices=1, early_stopping=True,
                disable_test=True)

if __name__ == '__main__':
     main()