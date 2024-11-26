from .flexibleDataset import FlexibleDataset
import os
import torch
import numpy as np
import scipy.io as sio
import yaml
from scipy.spatial.transform import Rotation
from torch_geometric.data import HeteroData

# class Standarizer_Seperate:
#     """
#     Class to standardize the data.
#     """
#     def __init__(self, q_mean, q_std, qd_mean, qd_std, lin_mean, lin_std, ang_mean, ang_std, device):
#         self.q_mean, self.q_std = torch.tensor(q_mean).to(device), torch.tensor(q_std).to(device)
#         self.qd_mean, self.qd_std = torch.tensor(qd_mean).to(device), torch.tensor(qd_std).to(device)
#         self.lin_mean, self.lin_std = torch.tensor(lin_mean).to(device), torch.tensor(lin_std).to(device)
#         self.ang_mean, self.ang_std = torch.tensor(ang_mean).to(device), torch.tensor(ang_std).to(device)

#     def transform(self, q=None, qd=None, lin=None, ang=None):
#         if isinstance(q, np.ndarray):
#             q_mean, q_std = self.q_mean.cpu().numpy(), self.q_std.cpu().numpy()
#             qd_mean, qd_std = self.qd_mean.cpu().numpy(), self.qd_std.cpu().numpy()
#             lin_mean, lin_std = self.lin_mean.cpu().numpy(), self.lin_std.cpu().numpy()
#             ang_mean, ang_std = self.ang_mean.cpu().numpy(), self.ang_std.cpu().numpy()
#         else:
#             q_mean, q_std = self.q_mean, self.q_std
#             qd_mean, qd_std = self.qd_mean, self.qd_std
#             lin_mean, lin_std = self.lin_mean, self.lin_std
#             ang_mean, ang_std = self.ang_mean, self.ang_std

#         return (q - q_mean) / q_std, (qd - qd_mean) / qd_std, (lin - lin_mean) / lin_std, (ang - ang_mean) / ang_std

#     def unstandarize(self, qn=None, qdn=None, linn=None, angn=None):
#         if isinstance(qn, np.ndarray):
#             q_mean, q_std = self.q_mean.cpu().numpy(), self.q_std.cpu().numpy()
#             qd_mean, qd_std = self.qd_mean.cpu().numpy(), self.qd_std.cpu().numpy()
#             lin_mean, lin_std = self.lin_mean.cpu().numpy(), self.lin_std.cpu().numpy()
#             ang_mean, ang_std = self.ang_mean.cpu().numpy(), self.ang_std.cpu().numpy()
#         else:
#             q_mean, q_std = self.q_mean, self.q_std
#             qd_mean, qd_std = self.qd_mean, self.qd_std
#             lin_mean, lin_std = self.lin_mean, self.lin_std
#             ang_mean, ang_std = self.ang_mean, self.ang_std

#         return qn * q_std + q_mean, qdn * qd_std + qd_mean, linn * lin_std + lin_mean, angn * ang_std + ang_mean

class Standarizer:

    def __init__(self, X_mean, X_std, Y_mean, Y_std, device):
        self.X_mean, self.X_std = torch.tensor(X_mean).to(device), torch.tensor(X_std).to(device)
        self.Y_mean, self.Y_std = torch.tensor(Y_mean).to(device), torch.tensor(Y_std).to(device)

    def transform(self, x=None, y=None):
        if isinstance(x, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if x is not None and y is not None:
            return (x - X_mean) / X_std, (y - Y_mean) / Y_std
        elif x is not None:
            return (x - X_mean) / X_std
        elif y is not None:
            return (y - Y_mean) / Y_std

    def unstandarize(self, xn=None, yn=None):
        if isinstance(xn, np.ndarray):
            X_mean, X_std = self.X_mean.cpu().numpy(), self.X_std.cpu().numpy()
            Y_mean, Y_std = self.Y_mean.cpu().numpy(), self.Y_std.cpu().numpy()
        else:
            X_mean, X_std = self.X_mean, self.X_std
            Y_mean, Y_std = self.Y_mean, self.Y_std

        if xn is not None and yn is not None:
            return xn * X_std + X_mean, yn * Y_std + Y_mean
        elif xn is not None:
            return xn * X_std + X_mean
        elif yn is not None:
            return yn * Y_std + Y_mean
        
    def to(self, device):
        self.X_mean = self.X_mean.to(device)
        self.X_std = self.X_std.to(device)
        self.Y_mean = self.Y_mean.to(device)
        self.Y_std = self.Y_std.to(device)

class Solo12Dataset(FlexibleDataset):
    """
    Dataset class for the Solo12 robot data.
    Each leg has 3 joints: HAA (Hip Abduction/Adduction),
    HFE (Hip Flexion/Extension), and KFE (Knee Flexion/Extension)
    """

    def __init__(self, 
                 root, 
                 path_to_urdf, 
                 urdf_package_name, 
                 urdf_package_relative_path, 
                 model_type, 
                 history_length, 
                 normalize=True, 
                 swap_legs=None,
                 symmetry_operator=None,  # Can be 'gs' or 'gt' or 'gr' or None
                 symmetry_mode=None, # Can be 'Euclidean' or 'MorphSym' or None
                 group_operator_path=None,
                 stage='train'):
        
        if swap_legs is not None and symmetry_operator is not None:
            raise ValueError("swap_legs is not None and symmetry_operator is not None is not supported.")
        
        # Check for symmetry_mode is specified correctly
        if symmetry_operator is not None and (symmetry_mode != 'MorphSym' and symmetry_mode != 'Euclidean' or group_operator_path is None):
            raise ValueError("symmetry_mode must be 'MorphSym' or 'Euclidean' when symmetry_operator is not None.")
        
        self.swap_legs = swap_legs
        # Set the swap legs parameter
        if swap_legs is not None:
            # sort each tuple in swap_legs
            if isinstance(swap_legs[0], tuple):
                sorted_swap_legs = tuple(tuple(sorted(swap_pair)) for swap_pair in swap_legs)
            else:
                sorted_swap_legs = tuple([tuple(sorted(swap_legs))])
            self.swap_legs = sorted_swap_legs

        # Set the symmetry parameters
        self.symmetry_operator = symmetry_operator
        self.symmetry_mode = symmetry_mode
        self.group_operator_path = group_operator_path 

        if self.symmetry_operator is not None:
            try:
                with open(self.group_operator_path, 'r') as file:
                    group_data = yaml.safe_load(file)
                self.permutation_Q_js = group_data.get('permutation_Q_js', [])
                self.reflection_Q_js = group_data.get('reflection_Q_js', [])

                # TODO: check if use base symmetry
                self.permutation_Q_bs = group_data.get('permutation_Q_bs', [])
                self.reflection_Q_bs_lin = group_data.get('reflection_Q_bs_lin', [])
                self.reflection_Q_bs_ang = group_data.get('reflection_Q_bs_ang', [])

                if self.symmetry_mode == 'MorphSym':
                    self.joint_coefficients = self.create_morphsym_coefficients(self.reflection_Q_js)
                    self.base_coefficients_lin = self.create_morphsym_coefficients(self.reflection_Q_bs_lin)
                    self.base_coefficients_ang = self.create_morphsym_coefficients(self.reflection_Q_bs_ang)
                    self.label_coefficients = self.create_morphsym_coefficients(self.reflection_Q_ls)
                elif self.symmetry_mode == 'Euclidean': # TODO: uncomment this
                    self.joint_coefficients = self.create_coefficient_dict(self.reflection_Q_js)
                    self.base_coefficients_lin = self.create_coefficient_dict(self.reflection_Q_bs_lin)
                    self.base_coefficients_ang = self.create_coefficient_dict(self.reflection_Q_bs_ang)
                    self.label_coefficients = self.create_coefficient_dict(self.reflection_Q_ls)
            except Exception as e:
                raise ValueError(f"Error loading group operator from {self.group_operator_path}: {e}")
        
        if model_type == 'heterogeneous_gnn_k4_com':
            self.load_data_sorted = self.load_data_sorted_k4
        elif model_type == 'heterogeneous_gnn_s4_com':
            self.load_data_sorted = self.load_data_sorted
        else:
            raise ValueError(f"Model type {model_type} is not implemented for Solo12 dataset.")
        
        self.normalize = normalize
        
        path_to_npz = os.path.join(root, 'processed', f'{stage}.npz')
        raw_data_npz = np.load(path_to_npz)
        self.X = raw_data_npz['X']
        self.Y = raw_data_npz['Y']
        if self.normalize:
            # self.X_mean, self.X_std, self.Y_mean, self.Y_std = self.compute_normalization(self.X, self.Y)
            path_to_stats = os.path.join(root, 'processed', 'rss_stats.npz')
            stat_data = np.load(path_to_stats)
            self.X_mean, self.X_std, self.Y_mean, self.Y_std = stat_data['x_mean'], stat_data['x_std'], stat_data['y_mean'], stat_data['y_std']
            self.standarizer = Standarizer(self.X_mean, self.X_std, self.Y_mean, self.Y_std, 'cpu')
            self.X, self.Y = self.standarizer.transform(self.X, self.Y)

        super().__init__(root, path_to_urdf, urdf_package_name, 
                         urdf_package_relative_path, data_format=model_type, 
                         history_length=history_length, normalize=self.normalize)
        
        self.model_type = model_type
        if self.model_type == 'heterogeneous_gnn_k4_com':
            # modify the number of base nodes
            self.hgnn_number_nodes = (4, self.hgnn_number_nodes[1], self.hgnn_number_nodes[2])
            # initialize new edges
            self._init_new_edges_k4()
        elif self.model_type == 'heterogeneous_gnn_s4_com':
            # initialize new edges
            print(f"Model type: {self.model_type} is initialized.")
        else:
            raise ValueError(f"Model type: {model_type} is not implemented yet for Solo12 dataset.")
        
        if self.normalize:
            unsorted_joint_stat_list = [self.X_mean[:12], self.X_mean[12:], self.X_std[:12], self.X_std[12:]]
            sorted_joint_stat_list = []
            for unsorted_array in unsorted_joint_stat_list:
                sorted_joint_stat_list.append(unsorted_array[self.joint_node_indices_sorted])
            self.X_mean = np.concatenate([sorted_joint_stat_list[0], sorted_joint_stat_list[1]])
            self.X_std = np.concatenate([sorted_joint_stat_list[2], sorted_joint_stat_list[3]])
            
            # Save statistics to processed/stats.npz
            processed_dir = os.path.join(self.root, 'processed')
            np.savez(os.path.join(processed_dir, f'{stage}_stats.npz'),
                    x_mean=self.X_mean,
                    x_std=self.X_std, 
                    y_mean=self.Y_mean,
                    y_std=self.Y_std)

        self.length = self.Y.shape[0]
        print(f"{stage} Dataset length: {self.length}")
    # def compute_normalization(self, q, qd, lin, ang):
    #     """
    #     Compute the mean and standard deviation for the joint and base data.
    #     Adapted from MorphSymm's repo: datasets/com_momentum/com_momentum.py
    #     """
    #     q_mean, qd_mean, lin_mean, ang_mean = 0., 0., 0., 0.
    #     q_std, qd_std, lin_std, ang_std = 1., 1., 1., 1.
        
    #     q_mean = np.mean(q, axis=0)
    #     qd_mean = np.mean(qd, axis=0)
    #     lin_mean = np.mean(lin, axis=0)
    #     ang_mean = np.mean(ang, axis=0)
    #     q_std = np.std(q, axis=0)
    #     qd_std = np.std(qd, axis=0)
    #     lin_std = np.std(lin, axis=0)
    #     ang_std = np.std(ang, axis=0)

    #     return q_mean, q_std, qd_mean, qd_std, lin_mean, lin_std, ang_mean, ang_std

    def compute_normalization(self, X, Y):
        """
        Compute the mean and standard deviation for the joint and base data.
        Adapted from MorphSymm's repo: datasets/com_momentum/com_momentum.py
        """
        X_mean, Y_mean, X_std, Y_std = 0., 0., 1., 1.
        
        # TODO: Obtain analytic formula for mean and std along orbit of discrete and continuous groups.
        X_mean = np.mean(X, axis=0)
        Y_mean = np.mean(Y, axis=0)
        X_std = np.std(X, axis=0)
        Y_std = np.std(Y, axis=0)

        return X_mean, X_std, Y_mean, Y_std

    def get_data_metadata(self):
        """Return metadata for the new graph structure"""
        # define node types
        node_types = ['base', 'joint']
        
        # define edge types (including new connections)
        if self.model_type == 'heterogeneous_gnn_k4_com':
            edge_types = [
                ('base', 'connect', 'joint'),
                ('joint', 'connect', 'base'),
                ('joint', 'connect', 'joint'),
                ('base', 'gt', 'base'),    # new transverse connections
                ('base', 'gs', 'base')     # new sagittal connections
            ]
        elif self.model_type == 'heterogeneous_gnn_s4_com':
            edge_types = [
                ('base', 'connect', 'joint'),
                ('joint', 'connect', 'base'),
                ('joint', 'connect', 'joint')
            ]
        else:
            raise ValueError(f"Model type: {self.model_type} is not implemented for Solo12 dataset.")
        
        return node_types, edge_types
    
    def get_helper_heterogeneous_gnn(self, idx):
        """Extended data retrieval method, including new graph structure"""

        # Create the Heterogeneous Data objects
        data = HeteroData()

        # set edge connections
        data['base', 'connect', 'joint'].edge_index = self.bj
        data['joint', 'connect', 'base'].edge_index = self.jb
        data['joint', 'connect', 'joint'].edge_index = self.jj
        if self.model_type == 'heterogeneous_gnn_k4_com':
            data['base', 'gt', 'base'].edge_index = self.gt
            data['base', 'gs', 'base'].edge_index = self.gs

        # create node feature matrices
        base_x = torch.ones((self.hgnn_number_nodes[0], self.base_width), dtype=torch.float64)
        joint_x = torch.ones((self.hgnn_number_nodes[1], self.joint_width), dtype=torch.float64)

        # get original data
        if self.model_type == 'heterogeneous_gnn_k4_com':
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted_k4(idx)
        elif self.model_type == 'heterogeneous_gnn_s4_com':
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted(idx)

        # For each base specified
        base_data = [lin_acc, ang_vel] # lin_acc, ang_vel shape: [history_length, 3*4]
        for i in range(self.hgnn_number_nodes[0]):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_base:
                final_input = torch.cat((final_input, torch.tensor(base_data[k][:,i*3:(i+1)*3].flatten('F'), dtype=torch.float64)), axis=0)
            base_x[i] = final_input

        # process joint and foot features (same as original implementation)
        # refer to the joint and foot feature processing part in the FlexibleDataset.get_helper_heterogeneous_gnn method L570-L596
        # For each joint specified
        joint_data = [j_p, j_v, j_T]
        for i, urdf_node_name in enumerate(self.urdf_name_to_graph_index_joint.keys()):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_joint:
                final_input = torch.cat((final_input, torch.tensor(joint_data[k][:,i].flatten('F'), dtype=torch.float64)), axis=0)

            joint_x[i] = final_input

        # set labels and node features
        data.y = torch.tensor(np.array(labels).reshape(-1), dtype=torch.float64)
        data['base'].x = base_x
        data['joint'].x = joint_x
        data.num_nodes = sum(self.hgnn_number_nodes)

        return data

    # ======================== DATA LOADING ==========================
    def load_data_sorted(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the joint and feet are sorted so that they match 
        the order in the URDF file. Additionally, the foot labels 
        are sorted so it matches the order in the URDF file.

        Next, labels are checked to make sure they aren't None. 
        Finally, normalize the data if self.normalize was set as True.
        We calculate the standard deviation for this normalization 
        using Bessel's correction (n-1 used instead of n).

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same values as load_data_at_dataset_seq(), but order of
            values inside arrays have been sorted (and potentially
            normalized).
            lin_vel: shape: Zeros(history_length, 4*3) 4 bases linear acceleration
            ang_vel: shape: Zeros(history_length, 4*3) 4 bases angular velocity
            j_p: shape: [history_length, 12] 12 joints position
            j_v: shape: [history_length, 12] 12 joints velocity
            j_T: None
            f_p: None
            f_v: None
            labels: shape: [6] 6 labels (base linear and angular velocity)
            r_p: None
            r_o: None
            timestamps: shape: [history_length]
        """
        lin_vel, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq(seq_num)

        sorted_base_list = [lin_vel, ang_vel]
        
        # Sort the joint information
        unsorted_joint_list = [j_p, j_v, j_T]
        sorted_joint_list = []
        for unsorted_array in unsorted_joint_list:
            if unsorted_array is not None:
                sorted_joint_list.append(unsorted_array[:,self.joint_node_indices_sorted])
            else:
                sorted_joint_list.append(None)

        # Sort the ground truth labels
        labels_sorted = labels # [batch_size=64, 6]

        return sorted_base_list[0], sorted_base_list[1], sorted_joint_list[0], sorted_joint_list[1], sorted_joint_list[2], None, None, labels_sorted, None, None, timestamps
                    
    def load_data_at_dataset_seq(self, seq_num: int):
        """Load data for a specified sequence"""
        # Load joint data
        # j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        # j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_p = self.X[:, :12][seq_num:seq_num+self.history_length].reshape(self.history_length, 12)
        j_v = self.X[:, 12:][seq_num:seq_num+self.history_length].reshape(self.history_length, 12)
        
        # Load base velocity data (as labels)
        # base_lin_vel = np.array(self.mat_data['base_lin_vel'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        # base_ang_vel = np.array(self.mat_data['base_ang_vel'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        base_lin_vel = self.Y[:, :3][seq_num:seq_num+self.history_length].reshape(self.history_length, 3)
        base_ang_vel = self.Y[:, 3:][seq_num:seq_num+self.history_length].reshape(self.history_length, 3)
        
        # Concatenate base velocities into a 6D vector as labels
        labels = np.concatenate([base_lin_vel[-1], base_ang_vel[-1]])  # Only use the last frame of the sequence as prediction target

        # Initilize lin_vel, ang_vel to zero
        lin_vel = np.zeros((self.history_length, 3))
        ang_vel = np.zeros((self.history_length, 3))
        
        # Fields returned as None are used in original QuadSDKDataset but not needed here
        return lin_vel, ang_vel, j_p, j_v, None, None, None, labels, None, None, None

    # ======================== DATA PROCESSING =======================
    def process(self):
        """
        Process raw Solo-12 data and save it in .mat format
        Data format:
        - X: (N, 24) contains positions and velocities of 12 joints
        - Y: (N, 6) contains base linear and angular velocities
        """
        # Read the npz file
        path_to_npz = os.path.join(self.root, 'raw', 'data.npz')
        raw_data = np.load(path_to_npz)
        
        # Get the data length
        dataset_entries = len(raw_data['X'])

        X = raw_data['X']
        Y = raw_data['Y']

        if self.normalize:
            self.X_mean, self.X_std, self.Y_mean, self.Y_std = self.compute_normalization(X, Y)
            self.standarizer = Standarizer(self.X_mean, self.X_std, self.Y_mean, self.Y_std, 'cpu')
            X_n, Y_n = self.standarizer.transform(X, Y)
        else:
            X_n, Y_n = X, Y
        
        # Separate the joint positions and velocities
        joint_positions = X_n[:, :12]    # The first 12 columns are joint positions
        joint_velocities = X_n[:, 12:]   # The last 12 columns are joint velocities
        
        # Separate the base linear and angular velocities
        base_lin_vel = Y_n[:, :3]        # The first 3 columns are linear velocities
        base_ang_vel = Y_n[:, 3:]        # The last 3 columns are angular velocities
        
        # Create the data dictionary
        data_dict = {
            'q': joint_positions,           # Joint positions (N, 12)
            'qd': joint_velocities,         # Joint velocities (N, 12)
            'base_lin_vel': base_lin_vel,   # Base linear velocities (N, 3)
            'base_ang_vel': base_ang_vel,   # Base angular velocities (N, 3)
            'timestamps': np.arange(dataset_entries),  # Use the index as timestamps
        }
        
        # Save as a mat file
        sio.savemat(os.path.join(self.processed_dir, "data.mat"), 
                    data_dict, 
                    do_compression=True)
        
        # Save the dataset information
        with open(os.path.join(self.processed_dir, "info.txt"), "w") as f:
            file_id, loc = self.get_file_id_and_loc()
            f.write(str(dataset_entries) + " " + file_id)

    # ======================== MorphoSymm DATA LOADING ========================
    def _init_new_edges_k4(self):
        """Initialize the new edge structure for K4"""
        # original edges
        bj, jb, jj, fj, jf = self.robotGraph.get_edge_index_matrices()

        # create new base-joint connections
        new_bj = []
        new_jb = []
        
        # Assign corresponding joints to each base node
        # RL base (idx 0) -> RL joints (0)
        # FL base (idx 1) -> FL joints (3)
        # RR base (idx 2) -> RR joints (6)
        # FR base (idx 3) -> FR joints (9)
        for base_idx in range(self.hgnn_number_nodes[0]):
            joint_hip = base_idx * 3
            new_bj.extend([[base_idx, joint_hip]])
            new_jb.extend([[joint_hip, base_idx]])

        # create new base-base connections
        # gt_edge: connections between front bases (0-1) and between back bases (2-3)
        gs_edges = [[0, 1], [1, 0], [2, 3], [3, 2]]
        
        # gs_edge: connections between left bases (1-3) and between right bases (0-2)
        gt_edges = [[0, 2], [2, 0], [1, 3], [3, 1]]

        # convert to PyTorch tensors
        self.bj = torch.tensor(new_bj, dtype=torch.long).t()
        self.jb = torch.tensor(new_jb, dtype=torch.long).t()
        self.jj = torch.tensor(jj, dtype=torch.long)
        self.fj = torch.tensor(fj, dtype=torch.long)
        self.jf = torch.tensor(jf, dtype=torch.long)
        self.gt = torch.tensor(gt_edges, dtype=torch.long).t()
        self.gs = torch.tensor(gs_edges, dtype=torch.long).t()

    def create_coefficient_dict(self, reflection_array):
        return {
            'gs': np.ones_like(reflection_array[0], dtype=np.float64),
            'gt': np.ones_like(reflection_array[0], dtype=np.float64),
            'gr': np.ones_like(reflection_array[0], dtype=np.float64)
        }
    
    def create_morphsym_coefficients(self, reflection_array):
        """Create coefficient dictionary for MorphSym mode.
        
        Args:
            reflection_array: Array containing sagittal and transversal reflection coefficients
        Returns:
            Dictionary containing gs, gt, and gr coefficients
        """
        gs = np.array(reflection_array[0], dtype=np.float64)
        gt = np.array(reflection_array[1], dtype=np.float64)
        return {
            'gs': gs,  # Sagittal symmetry
            'gt': gt,  # Transversal symmetry
            'gr': gs * gt  # Rotational symmetry (element-wise multiplication)
        }

    def load_data_sorted_k4(self, seq_num: int):
        """
        Loads data from the dataset at the provided sequence number.
        However, the joint and feet are sorted so that they match 
        the order in the URDF file. Additionally, the foot labels 
        are sorted so it matches the order in the URDF file.

        Next, labels are checked to make sure they aren't None. 
        Finally, normalize the data if self.normalize was set as True.
        We calculate the standard deviation for this normalization 
        using Bessel's correction (n-1 used instead of n).

        Parameters:
            seq_num (int): The sequence number of the txt file
                whose data should be loaded.

        Returns:
            Same values as load_data_at_dataset_seq(), but order of
            values inside arrays have been sorted (and potentially
            normalized).
            lin_vel: shape: Zeros(history_length, 4*3) 4 bases linear acceleration
            ang_vel: shape: Zeros(history_length, 4*3) 4 bases angular velocity
            j_p: shape: [history_length, 12] 12 joints position
            j_v: shape: [history_length, 12] 12 joints velocity
            j_T: None
            f_p: shape: [history_length, 4*3] 4 feet position
            f_v: shape: [history_length, 4*3] 4 feet velocity
            labels: shape: [4 * 6] 4 base nodes * 6 labels (base linear and angular velocity)
            r_p: None
            r_o: None
            timestamps: shape: [history_length]
        """
        lin_vel, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq(seq_num)

        # duplicate the base information for each base node, no need to sort
        # lin_vel, ang_vel shape: [history_length, 3] ==> [history_length, 4*3]
        lin_vel = np.tile(lin_vel, (1, 4))
        ang_vel = np.tile(ang_vel, (1, 4))
        sorted_base_list = [lin_vel, ang_vel]

        # if self.symmetry_operator is not None:
        #     sorted_base_list = self.apply_symmetry(sorted_base_list, part='base')
        
        # Sort the joint information
        unsorted_joint_list = [j_p, j_v, j_T]
        sorted_joint_list = []
        for unsorted_array in unsorted_joint_list:
            if unsorted_array is not None:
                sorted_joint_list.append(unsorted_array[:,self.joint_node_indices_sorted])
            else:
                sorted_joint_list.append(None)

        if self.symmetry_operator is not None:
            sorted_joint_list = self.apply_symmetry(sorted_joint_list, part='joint')

        # Sort the foot information
        unsorted_foot_list = [f_p, f_v]
        sorted_foot_list = []
        for unsorted_array in unsorted_foot_list:
            if unsorted_array is not None:
                sorted_indices = []
                for index in self.foot_node_indices_sorted:
                    for i in range(0, 3):
                        sorted_indices.append(int(index*3+i))
                sorted_foot_list.append(unsorted_array[:,sorted_indices])
            else:
                sorted_foot_list.append(None)
        
        # if self.symmetry_operator is not None:
        #     sorted_foot_list = self.apply_symmetry(sorted_foot_list, part='foot')

        # Sort the ground truth labels
        # Duplicate labels for each base node: labels [6] -> [4*6]
        labels_lin_vel = np.tile(labels[:3], 4)
        labels_ang_vel = np.tile(labels[3:], 4)
        labels_sorted = [labels_lin_vel, labels_ang_vel]
        if self.symmetry_operator is not None:
            labels_sorted = self.apply_symmetry([labels_sorted], part='label')[0]

        return sorted_base_list[0], sorted_base_list[1], sorted_joint_list[0], sorted_joint_list[1], sorted_joint_list[2], sorted_foot_list[0], sorted_foot_list[1], labels_sorted, r_p, r_o, timestamps

    def apply_symmetry(self, data_list, part='joint'):
        """Apply the symmetry operator to the data"""
        if part == 'label':
            permutation_Q = self.permutation_Q_bs
            coefficients_lin = self.base_coefficients_lin
            coefficients_ang = self.base_coefficients_ang
            new_data_list = []
            for i, data in enumerate(data_list):
                if i==0: # linear acceleration
                    coefficients = coefficients_lin
                elif i==1: # angular velocity
                    coefficients = coefficients_ang
                if self.symmetry_operator == 'gs':
                    data = data[permutation_Q[0]].copy() * coefficients['gs']
                elif self.symmetry_operator == 'gt':
                    data = data[permutation_Q[1]].copy() * coefficients['gt']
                elif self.symmetry_operator == 'gr':
                    data = data[permutation_Q[0]].copy()
                    data = data[permutation_Q[1]].copy() * coefficients['gr']
                new_data_list.append(data)
        else:
            if part == 'joint':
                permutation_Q = self.permutation_Q_js
                coefficients = self.joint_coefficients
            elif part == 'foot':
                permutation_Q = self.permutation_Q_fs
                coefficients = self.foot_coefficients
            elif part == 'label': 
                # for labels, data shape: (4,), for classification task, 
                # we need to apply Euclidean symmetry to labels
                permutation_Q = self.permutation_Q_ls
                coefficients = self.label_coefficients
            else:
                raise ValueError(f"Invalid part: {part}")

            new_data_list = []
            # data_list: [j_p, j_v, j_T], each shape: [history_length, 12]
            for data in data_list:
                if data is None:
                    new_data_list.append(None)
                    continue
                # for labels, data shape: (4,)
                if data.ndim == 1: 
                    if self.symmetry_operator == 'gs':
                        data = data[permutation_Q[0]].copy() * coefficients['gs']
                    elif self.symmetry_operator == 'gt':
                        data = data[permutation_Q[1]].copy() * coefficients['gt']
                    elif self.symmetry_operator == 'gr':
                        data = data[permutation_Q[0]].copy()
                        data = data[permutation_Q[1]].copy() * coefficients['gr']
                else: # for j_p, j_v, j_T, each shape: [history_length, 12] ... 
                    if self.symmetry_operator == 'gs':
                        data = data[:, permutation_Q[0]].copy() * coefficients['gs']
                    elif self.symmetry_operator == 'gt':
                        data = data[:, permutation_Q[1]].copy() * coefficients['gt']
                    elif self.symmetry_operator == 'gr':
                        data = data[:, permutation_Q[0]].copy()
                        data = data[:, permutation_Q[1]].copy() * coefficients['gr']
                new_data_list.append(data)
        return new_data_list
    
    def swap_legs_data(self, data_dict, leg1_idx, leg2_idx):
        """
        Swap the data of two legs
        leg_idx mapping: FR=0, FL=1, HR=2, HL=3
        Each leg has 3 joints
        """
        # Calculate the index range of the joints
        leg1_start = leg1_idx * 3
        leg1_end = leg1_start + 3
        leg2_start = leg2_idx * 3
        leg2_end = leg2_start + 3
        
        # Swap the joint position data
        if 'j_p' in data_dict and data_dict['j_p'] is not None:
            data_dict['j_p'][:, leg1_start:leg1_end], data_dict['j_p'][:, leg2_start:leg2_end] = \
                data_dict['j_p'][:, leg2_start:leg2_end].copy(), data_dict['j_p'][:, leg1_start:leg1_end].copy()
        
        # Swap the joint velocity data
        if 'j_v' in data_dict and data_dict['j_v'] is not None:
            data_dict['j_v'][:, leg1_start:leg1_end], data_dict['j_v'][:, leg2_start:leg2_end] = \
                data_dict['j_v'][:, leg2_start:leg2_end].copy(), data_dict['j_v'][:, leg1_start:leg1_end].copy()
        
        # Swap the joint torque data
        if 'j_T' in data_dict and data_dict['j_T'] is not None:
            data_dict['j_T'][:, leg1_start:leg1_end], data_dict['j_T'][:, leg2_start:leg2_end] = \
                data_dict['j_T'][:, leg2_start:leg2_end].copy(), data_dict['j_T'][:, leg1_start:leg1_end].copy()
        
        return data_dict
    
    # ========================= DOWNLOADING ==========================
    def get_downloaded_dataset_file_name(self):
        """Specify the name of the raw data file"""
        return "data.npz"
    
    def get_file_id_and_loc(self):
        """Specify the source location of the data file"""
        # If loading data from local, return local path
        return "local_file", "local"
        # If data needs to be downloaded, return download link
        # return "https://path/to/your/data_100000.npz", "url"
    
    def get_expected_urdf_name(self):
        return "solo"
    
    # ======================== DATA LOADING ==========================
    def get_urdf_name_to_dataset_array_index(self):
        """Joint mapping for Solo12"""
        '''
            # The data structure in npz file should be:
            X = [
                # Joint positions (first 12 columns)
                'FL_HAA', 'FL_HFE', 'FL_KFE',  # Front Left leg (0-2)
                'FR_HAA', 'FR_HFE', 'FR_KFE',  # Front Right leg (3-5)
                'HL_HAA', 'HL_HFE', 'HL_KFE',  # Hind Left leg (6-8)
                'HR_HAA', 'HR_HFE', 'HR_KFE',  # Hind Right leg (9-11)
                
                # Joint velocities (last 12 columns)
                'FL_HAA_vel', 'FL_HFE_vel', 'FL_KFE_vel',
                'FR_HAA_vel', 'FR_HFE_vel', 'FR_KFE_vel',
                'HL_HAA_vel', 'HL_HFE_vel', 'HL_KFE_vel',
                'HR_HAA_vel', 'HR_HFE_vel', 'HR_KFE_vel'
            ]

            Y = [
                'base_lin_vel_x', 'base_lin_vel_y', 'base_lin_vel_z',  # Base linear velocity (0-2)
                'base_ang_vel_x', 'base_ang_vel_y', 'base_ang_vel_z'   # Base angular velocity (3-5)
            ]
        '''
        return {
            'floating_base': 0,

            'FL_hip_joint': 0,   # Front Left Hip Abduction/Adduction
            'FL_thigh_joint': 1,   # Front Left Hip Flexion/Extension
            'FL_calf_joint': 2,   # Front Left Knee Flexion/Extension
            'FR_hip_joint': 3,   # Front Right Hip
            'FR_thigh_joint': 4,   # Front Right Hip
            'FR_calf_joint': 5,   # Front Right Knee
            'RL_hip_joint': 6,   # Hind Left Hip
            'RL_thigh_joint': 7,   # Hind Left Hip
            'RL_calf_joint': 8,   # Hind Left Knee
            'RR_hip_joint': 9,   # Hind Right Hip
            'RR_thigh_joint': 10,  # Hind Right Hip
            'RR_calf_joint': 11,   # Hind Right Knee

            'FL_foot_fixed': 0,
            'FR_foot_fixed': 1,
            'RL_foot_fixed': 2,
            'RR_foot_fixed': 3
        }
