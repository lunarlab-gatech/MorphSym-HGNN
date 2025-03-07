import os
import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pathlib import Path
import names
from torch.utils.data import Subset
import numpy as np
import torchmetrics
import torchmetrics.classification
from .customMetrics import CrossEntropyLossMetric, BinaryF1Score, CosineSimilarityMetric
from .hgnn import COM_HGNN
from .hgnn_s4_com import COM_HGNN_S4
from .hgnn_k4_com import COM_HGNN_K4
from .hgnn_c2_com import COM_HGNN_C2
from torch_geometric.profile import count_parameters
from ..datasets_py.flexibleDataset import FlexibleDataset
from ..datasets_py.soloDataset import Standarizer
from scipy.spatial.transform import Rotation


class COM_Base_Lightning(L.LightningModule):
    """
    Define training, validation, test, and prediction 
    steps used by all models, in addition to the 
    optimizer.
    """

    def __init__(self, optimizer: str, lr: float, data_path: str):
        """
        Parameters:
            optimizer (str) - A string representing the optimizer to use. 
                Currently supports "adam" and "sgd".
            lr (float) - The learning rate of the optimizer.
            data_path (str) - dataset folder.
        """

        # Setup input parameters
        super().__init__()
        self.optimizer = optimizer
        self.lr = lr
        self.data_path = data_path
        self.regression = True # Only works for regression task

        # Data pre-processing
        stats = np.load(os.path.join(self.data_path, 'processed', 'rss_stats.npz'))
        x_means = stats['x_mean']
        x_stds = stats['x_std']
        y_means = stats['y_mean']
        y_stds = stats['y_std']
        self.standarizer = Standarizer(x_means, x_stds, y_means, y_stds, device='cuda:0')

        # ====== Setup the metrics ======
        self.metric_mse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=True)
        self.metric_rmse: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=False)
        self.metric_mse_lin: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=True)
        self.metric_mse_ang: torchmetrics.MeanSquaredError = torchmetrics.regression.MeanSquaredError(squared=True)
        self.metric_cos_sim_lin = CosineSimilarityMetric()
        self.metric_cos_sim_ang = CosineSimilarityMetric()
        # self.metric_l1: torchmetrics.MeanAbsoluteError = torchmetrics.regression.MeanAbsoluteError()

        # Setup variables to hold the losses
        self.mse_loss = None
        self.rmse_loss = None
        self.mse_loss_lin = None
        self.mse_loss_ang = None
        self.cos_sim_lin = None
        self.cos_sim_ang = None
        self.avg_cos_sim = None
        self.loss = None
        # self.l1_loss = None

    # ======================= Logging =======================
    def log_losses(self, step_name: str, on_step: bool):
        # Ensure one is enabled and one is disabled
        on_epoch = not on_step

        # Log the losses
        self.log(step_name + "_MSE_loss", self.mse_loss, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_RMSE_loss", self.rmse_loss, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_MSE_loss_lin", self.mse_loss_lin, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_MSE_loss_ang", self.mse_loss_ang, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_cos_sim_lin", self.cos_sim_lin, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_cos_sim_ang", self.cos_sim_ang, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + '_avg_cos_sim', self.avg_cos_sim, on_step=on_step, on_epoch=on_epoch)
        self.log(step_name + "_loss", self.loss, on_step=on_step, on_epoch=on_epoch)
        # self.log(step_name + "_L1_loss", self.l1_loss, on_step=on_step, on_epoch=on_epoch)

    # ======================= Loss Calculation =======================
    def calculate_losses_step(self, y: torch.Tensor, y_pred: torch.Tensor):
        self.mse_loss = self.metric_mse(y_pred.flatten(), y.flatten())
        self.rmse_loss = self.metric_rmse(y_pred.flatten(), y.flatten())
        y = y.view(y.shape[0], self.model.num_bases, self.model.num_dimensions_per_base)
        y_pred = y_pred.view(y_pred.shape[0], self.model.num_bases, self.model.num_dimensions_per_base)
        self.mse_loss_lin = self.metric_mse_lin(y_pred[:, :, :3].flatten(), y[:, :, :3].flatten())
        self.mse_loss_ang = self.metric_mse_ang(y_pred[:, :, 3:].flatten(), y[:, :, 3:].flatten())
        # self.l1_loss = self.metric_l1(y_pred, y)

        # Unstandarize the data for cos similarity metrics
        self.standarizer.to(y.device)
        y = self.standarizer.unstandarize(yn=y)
        y_pred = self.standarizer.unstandarize(yn=y_pred)

        y = y.view(y.shape[0], self.model.num_bases, self.model.num_dimensions_per_base)
        y_pred = y_pred.view(y_pred.shape[0], self.model.num_bases, self.model.num_dimensions_per_base)
        y_lin_vel = y[:, 0, :3].flatten(start_dim=1)
        y_ang_vel = y[:, 0, 3:].flatten(start_dim=1)
        y_pred_lin_vel = y_pred[:, 0, :3].flatten(start_dim=1)
        y_pred_ang_vel = y_pred[:, 0, 3:].flatten(start_dim=1)

        self.cos_sim_lin = self.metric_cos_sim_lin(y_pred_lin_vel, y_lin_vel)
        self.cos_sim_ang = self.metric_cos_sim_ang(y_pred_ang_vel, y_ang_vel)
        self.avg_cos_sim = (self.cos_sim_lin + self.cos_sim_ang) / 2

        self.loss = self.mse_loss


    def calculate_losses_epoch(self) -> None:
        self.mse_loss = self.metric_mse.compute()
        self.rmse_loss = self.metric_rmse.compute()
        self.mse_loss_lin = self.metric_mse_lin.compute()
        self.mse_loss_ang = self.metric_mse_ang.compute()
        self.cos_sim_lin = self.metric_cos_sim_lin.compute()
        self.cos_sim_ang = self.metric_cos_sim_ang.compute()
        self.avg_cos_sim = (self.cos_sim_lin + self.cos_sim_ang) / 2
        self.loss = self.metric_mse.compute()
        # self.l1_loss = self.metric_l1.compute()

    def reset_all_metrics(self) -> None:
        self.metric_mse.reset()
        self.metric_rmse.reset()
        self.metric_mse_lin.reset()
        self.metric_mse_ang.reset()
        self.metric_cos_sim_lin.reset()
        self.metric_cos_sim_ang.reset()
        # self.metric_l1.reset()

    # ======================= Training =======================
    def training_step(self, batch, batch_idx):
        y, y_pred = self.step_helper_function(batch)
        self.calculate_losses_step(y, y_pred)
        self.log_losses("train", on_step=True)
        return self.loss

    # ======================= Validation =======================
    def on_validation_epoch_start(self):
        self.reset_all_metrics()

    def validation_step(self, batch, batch_idx):
        y, y_pred = self.step_helper_function(batch)
        self.calculate_losses_step(y, y_pred)
        return self.mse_loss

    def on_validation_epoch_end(self):
        self.calculate_losses_epoch()
        self.log_losses("val", on_step=False)

    # ======================= Testing =======================
    def on_test_epoch_start(self):
        self.reset_all_metrics()

    def test_step(self, batch, batch_idx):
        y, y_pred = self.step_helper_function(batch)
        self.calculate_losses_step(y, y_pred)
        return self.mse_loss

    def on_test_epoch_end(self):
        self.calculate_losses_epoch()
        self.log_losses("test", on_step=False)

    # ======================= Prediction =======================
    # NOTE: These methods have not been fully tested. Use at 
    # your own risk.

    def on_predict_start(self):
        self.reset_all_metrics()

    def predict_step(self, batch, batch_idx):
        """
        Returns the predicted values from the model given a specific batch. 
        Currently only implemented for regression.

        Note, the HGNN models (Both regression and classification) directly
        predict with model, instead of using this built-in method.

        Returns:
            y (torch.Tensor) - Ground Truth labels per foot (GRF labels for
              regression, 16 class contact labels for classifiction)
            y_pred (torch.Tensor) - Predicted outputs (GRF labels per foot 
                for regression, 16 class predictions for classifications)
        """
        y, y_pred = self.step_helper_function(batch)
        self.calculate_losses_step(y, y_pred)
        return y, y_pred
        
    def on_predict_end(self):
        self.calculate_losses_epoch()

    # ======================= Optimizer =======================
    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Invalid optimizer setting")
        return optimizer

    # ======================= Helper Functions =======================
    def step_helper_function(self, batch):
        """
        Function that actually runs the model on the batch
        to get loss and model output.

        Returns:
            y: The ground truth labels with the shape (batch_size, 4).
            y_pred: The predicted model output. If self.regression, these are 
                just GRF values with the shape (batch_size, 4). If not 
                self.regression, then these are contact probabilty logits,
                two per foot, with shape (batch_size, 4). Foot order matches
                order of URDF file, and logit assumes first value is logit of
                no contact, and second value is logit of contact.

        """
        raise NotImplementedError


class COM_MLP_Lightning(COM_Base_Lightning):

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 batch_size: int, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU(),
                 data_path: Path = None):
        """
        Constructor for MLP_Lightning class. Pytorch Lightning
        wrapper around the Pytorch Torchvision MLP class.

        Parameters:
            in_channels (int): Number of input parameters to the MLP.
            hidden_channels (int): The hidden size.
            out_channels (int): The number of outputs from the MLP.
            num_layers (int): The number of layers in the model.
            batch_size (int): The size of the batches from the dataloaders.
            optimizer (str): String name of the optimizer that should
                be used.
            lr (float): The learning rate used by the model.
            regression (bool): True if the problem is regression, false if 
                classification. Mainly for tracking model usage using W&B.
            activation_fn (class): The activation function used between the layers.
        """

        super().__init__(optimizer, lr, data_path)
        self.batch_size = batch_size
        self.regression = regression

        # Create the proper number of layers
        modules = []
        if num_layers < 2:
            raise ValueError("num_layers must be 2 or greater")
        elif num_layers == 2:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))
        else:
            modules.append(nn.Linear(in_channels, hidden_channels))
            modules.append(activation_fn)
            for i in range(0, num_layers - 2):
                modules.append(nn.Linear(hidden_channels, hidden_channels))
                modules.append(activation_fn)
            modules.append(nn.Linear(hidden_channels, out_channels))

        self.model = nn.Sequential(*modules)
        self.model.num_bases = 1
        self.model.num_dimensions_per_base = 6
        self.save_hyperparameters()

    def step_helper_function(self, batch):
        x, y = batch
        y_pred = self.model(x)
        return y, y_pred


class COM_HGNN_Lightning(COM_Base_Lightning):

    def __init__(self, hidden_channels: int, num_layers: int, data_metadata,
                 dummy_batch, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU(), com_dimension: int = 6,
                 data_path: Path = None):
        """
        Constructor for Heterogeneous GNN.

        Parameters:
            dummy_batch: Used to initialize the lazy modules.
            optimizer (str): String name of the optimizer that should
                be used.
            lr (float): The learning rate used by the model.

            See hgnn.py for information on remaining parameters.
        """
        super().__init__(optimizer, lr, data_path)

        self.model = COM_HGNN(hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              data_metadata=data_metadata,
                              regression=regression,
                              activation_fn=activation_fn,
                              com_dimension=com_dimension)
        self.regression = regression

        # print(dummy_batch)
        # Initialize lazy modules
        with torch.no_grad():
            self.model(x_dict=dummy_batch.x_dict,
                       edge_index_dict=dummy_batch.edge_index_dict)
        self.save_hyperparameters()

    def step_helper_function(self, batch):
        # Get the raw foot output
        out_raw = self.model(x_dict=batch.x_dict,
                             edge_index_dict=batch.edge_index_dict)

        # Get the outputs from the foot nodes
        batch_size = None
        if hasattr(batch, "batch_size"):
            batch_size = batch.batch_size
        else:
            batch_size = 1
        y_pred = torch.reshape(out_raw.squeeze(), (batch_size, self.model.num_dimensions_per_base * self.model.num_bases))

        # Get the labels
        y = torch.reshape(batch.y, (batch_size, self.model.num_dimensions_per_base * self.model.num_bases))

        return y, y_pred
    

class COM_HGNN_SYM_Lightning(COM_Base_Lightning):
    def __init__(self, hidden_channels: int, num_layers: int, data_metadata,
                 dummy_batch, optimizer: str = "adam", lr: float = 0.003,
                 regression: bool = True, activation_fn = nn.ReLU(),
                 symmetry_mode: str = None, group_operator_path: str = None, model_type: str = 'heterogeneous_gnn_k4_com',
                 data_path: Path = None):
        """
        Constructor for Heterogeneous GNN with K4 structure.

        Parameters:
            dummy_batch: Used to initialize the lazy modules.
            optimizer (str): String name of the optimizer that should
                be used.
            lr (float): The learning rate used by the model.

            See hgnn_k4.py for information on remaining parameters.
        """
        super().__init__(optimizer, lr, data_path)

        if model_type == 'heterogeneous_gnn_k4_com':
            self.model = COM_HGNN_K4(hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              data_metadata=data_metadata,
                              regression=regression,
                              activation_fn=activation_fn,
                              symmetry_mode=symmetry_mode,
                              group_operator_path=group_operator_path)
        elif model_type == 'heterogeneous_gnn_c2_com':
            self.model = COM_HGNN_C2(hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              data_metadata=data_metadata,
                              regression=regression,
                              activation_fn=activation_fn,
                              symmetry_mode=symmetry_mode,
                              group_operator_path=group_operator_path)
        elif model_type == 'heterogeneous_gnn_s4_com':
            self.model = COM_HGNN_S4(hidden_channels=hidden_channels,
                              num_layers=num_layers,
                              data_metadata=data_metadata,
                              regression=regression,
                              activation_fn=activation_fn)
        self.regression = regression

        # Initialize lazy modules
        with torch.no_grad():
            self.model(x_dict=dummy_batch.x_dict,
                       edge_index_dict=dummy_batch.edge_index_dict)
        self.save_hyperparameters()

    # Rewrite the step helper function to match the base class
    # Same with Heterogeneous_GNN_Lightning()
    def step_helper_function(self, batch):
        # Get the raw foot output
        out_raw = self.model(x_dict=batch.x_dict,
                             edge_index_dict=batch.edge_index_dict)

        # Get the outputs from the foot nodes
        batch_size = None
        if hasattr(batch, "batch_size"):
            batch_size = batch.batch_size
        else:
            batch_size = 1
        y_pred = torch.reshape(out_raw.squeeze(), (batch_size, self.model.num_bases * self.model.num_dimensions_per_base))

        # Get the labels
        y = torch.reshape(batch.y, (batch_size, self.model.num_bases * self.model.num_dimensions_per_base))

        return y, y_pred