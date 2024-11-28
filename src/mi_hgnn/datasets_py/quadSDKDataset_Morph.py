from rosbags.highlevel import AnyReader
from pathlib import Path
from torchvision.datasets.utils import download_file_from_google_drive
import os
from .flexibleDataset import FlexibleDataset
import numpy as np
import scipy.io as sio
from scipy.spatial.transform import Rotation
from .quadSDKDataset import QuadSDKDataset
import yaml
import torch
from torch_geometric.data import HeteroData


class QuadSDKDataset_NewGraph(QuadSDKDataset):
    """
    Dataset class for the simulated data provided through
    Quad-SDK and the Gazebo simulator.
    """
    def __init__(self, 
                 root, 
                 path_to_urdf, 
                 urdf_package_name, 
                 urdf_package_relative_path, 
                 model_type, 
                 history_length, 
                 normalize=True,
                 path_to_urdf_dynamics=None,
                 symmetry_operator=None,  # Can be 'gs' or 'gt' or 'gr' or None
                 symmetry_mode=None, # Can be 'Euclidean' or 'MorphSym' or None
                 group_operator_path=None,
                 grf_body_to_world_frame=False,
                 grf_dimension=1):

        # Set the symmetry parameters
        self.symmetry_operator = symmetry_operator
        self.symmetry_mode = symmetry_mode
        self.group_operator_path = group_operator_path 
        self.grf_body_to_world_frame = grf_body_to_world_frame
        self.grf_dimension = grf_dimension

        # TODO: should be checked, how to support c2 symmetry
        if self.symmetry_operator is not None:
            # Load the group operator from yaml file
            try:
                with open(self.group_operator_path, 'r') as file:
                    group_data = yaml.safe_load(file)
                    
                # Extract permutation and reflection data
                self.permutation_Q_js = group_data.get('permutation_Q_js', []) # np array, shape [2, 12]
                self.reflection_Q_js = group_data.get('reflection_Q_js', []) # np array, shape [2, 12]

                self.permutation_Q_fs = group_data.get('permutation_Q_fs', []) # np array, shape [2, 4*3]
                self.reflection_Q_fs = group_data.get('reflection_Q_fs', []) # np array, shape [2, 4*3]

                self.permutation_Q_bs = group_data.get('permutation_Q_bs', []) # np array, shape [2, 4*3]
                self.reflection_Q_bs_lin = group_data.get('reflection_Q_bs_lin', []) # np array, shape [2, 4*3]
                self.reflection_Q_bs_ang = group_data.get('reflection_Q_bs_ang', []) # np array, shape [2, 4*3]

                self.permutation_Q_ls = group_data.get('permutation_Q_ls', []) # np array, shape [2, 4]
                self.reflection_Q_ls = group_data.get('reflection_Q_ls', []) # np array, shape [2, 4]

                # Create symmetry coefficients dictionary based on the yaml data
                if self.symmetry_mode == 'MorphSym':
                    self.joint_coefficients = self.create_morphsym_coefficients(self.reflection_Q_js)
                    self.foot_coefficients = self.create_morphsym_coefficients(self.reflection_Q_fs)
                    self.base_coefficients_lin = self.create_morphsym_coefficients(self.reflection_Q_bs_lin)
                    self.base_coefficients_ang = self.create_morphsym_coefficients(self.reflection_Q_bs_ang)
                    self.label_coefficients = self.create_morphsym_coefficients(self.reflection_Q_ls)
                elif self.symmetry_mode == 'Euclidean': # TODO: uncomment this
                    self.joint_coefficients = self.create_coefficient_dict(self.reflection_Q_js)
                    self.foot_coefficients = self.create_coefficient_dict(self.reflection_Q_fs)
                    self.base_coefficients_lin = self.create_coefficient_dict(self.reflection_Q_bs_lin)
                    self.base_coefficients_ang = self.create_coefficient_dict(self.reflection_Q_bs_ang)
                    self.label_coefficients = self.create_coefficient_dict(self.reflection_Q_ls)
            except FileNotFoundError:
                raise ValueError(f"Group operator file not found at {self.group_operator_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

        if model_type == 'heterogeneous_gnn_c2':
            self.load_data_sorted = self.load_data_sorted_c2
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        super().__init__(root, path_to_urdf, urdf_package_name, 
                            urdf_package_relative_path, data_format=model_type, 
                            history_length=history_length, normalize=normalize,
                            urdf_path_dynamics=path_to_urdf_dynamics)
        
        self.model_type = model_type
        if self.model_type == 'heterogeneous_gnn_c2':
            # modify the number of base nodes
            self.hgnn_number_nodes = (2, self.hgnn_number_nodes[1], self.hgnn_number_nodes[2])
            # initialize new edges
            self._init_new_edges_c2()

    # ===================== DATA LOADING ==========================
    def load_data_sorted_c2(self, seq_num: int):
        if self.grf_dimension == 1:
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq(seq_num)
        elif self.grf_dimension == 3:
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq_3d(seq_num)
        else:
            raise ValueError(f"Invalid grf_dimension: {self.grf_dimension}")
        
        # duplicate the base information for each base node, no need to sort
        # lin_acc, ang_vel shape: [history_length, 3] ==> [history_length, 2*3]
        lin_acc = np.tile(lin_acc, (1, 2))
        ang_vel = np.tile(ang_vel, (1, 2))
        sorted_base_list = [lin_acc, ang_vel]

        if self.symmetry_operator is not None:
            sorted_base_list = self.apply_symmetry(sorted_base_list, part='base')
        
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
        labels_sorted = None
        if labels is None:
            raise ValueError("Dataset must provide labels.")
        elif self.grf_dimension == 1:
            labels_sorted = labels[self.foot_node_indices_sorted]
        elif self.grf_dimension == 3:
            unsorted_grfs = labels
            sorted_grfs = []
            for index in self.foot_node_indices_sorted:
                for i in range(0, 3):
                    sorted_grfs.append(int(index*3+i))
            labels_sorted = unsorted_grfs[sorted_grfs]

        if self.symmetry_operator is not None:
            labels_sorted = self.apply_symmetry([labels_sorted], part='label')[0]

        # Normalize the data if desired
        # TODO: check the normalization
        norm_arrs = [None, None, None, None, None, None, None, None, None]
        if self.normalize:
            # Normalize all data except the labels
            to_normalize_array = [sorted_base_list[0], sorted_base_list[1], sorted_joint_list[0], sorted_joint_list[1], sorted_joint_list[2], sorted_foot_list[0], sorted_foot_list[1], r_p, r_o]
            for i, array in enumerate(to_normalize_array):
                if (array is not None) and (array.shape[0] > 1):
                    array_tensor = torch.from_numpy(array)
                    norm_arrs[i] = np.nan_to_num((array_tensor-torch.mean(array_tensor,axis=0))/torch.std(array_tensor, axis=0, correction=1).numpy(), copy=False, nan=0.0)

            return norm_arrs[0], norm_arrs[1], norm_arrs[2], norm_arrs[3], norm_arrs[4], norm_arrs[5], norm_arrs[6], labels_sorted, norm_arrs[7], norm_arrs[8], timestamps
        else:
            return sorted_base_list[0], sorted_base_list[1], sorted_joint_list[0], sorted_joint_list[1], sorted_joint_list[2], sorted_foot_list[0], sorted_foot_list[1], labels_sorted, r_p, r_o, timestamps

    # ===================== MorphSym Tools ===============================
    def apply_symmetry(self, data_list, part='joint'):
        """Apply the symmetry operator to the data"""
        if part == 'base':
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
                    data = data[:, permutation_Q[0]].copy() * coefficients['gs']
                elif self.symmetry_operator == 'gt':
                    data = data[:, permutation_Q[1]].copy() * coefficients['gt']
                elif self.symmetry_operator == 'gr':
                    data = data[:, permutation_Q[0]].copy()
                    data = data[:, permutation_Q[1]].copy() * coefficients['gr']
                new_data_list.append(data)
        else:
            if part == 'joint':
                permutation_Q = self.permutation_Q_js
                coefficients = self.joint_coefficients
            elif part == 'foot':
                permutation_Q = self.permutation_Q_fs
                coefficients = self.foot_coefficients
            elif part == 'label' and self.grf_dimension == 1: 
                # for labels, data shape: (4,) for 1D GRF
                permutation_Q = self.permutation_Q_ls
                coefficients = self.label_coefficients
            elif part == 'label' and self.grf_dimension == 3:
                # for labels, data shape: (3*4, ) for 3D GRF
                permutation_Q = self.permutation_Q_fs
                coefficients = self.foot_coefficients
            else:
                raise ValueError(f"Invalid part: {part}")

            new_data_list = []
            # data_list: [j_p, j_v, j_T], each shape: [history_length, 12]
            for data in data_list:
                if data is None:
                    new_data_list.append(None)
                    continue
                # for labels, data shape: (4,) or (3*4, )
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
    
    def _init_new_edges_c2(self):
        """Initialize the new edge structure for C2"""
        # original edges
        bj, jb, jj, fj, jf = self.robotGraph.get_edge_index_matrices()

        # create new base-joint connections
        new_bj_front = []
        new_jb_front = []
        new_bj_back = []
        new_jb_back = []
        # L base (idx 0) -> FL joints (0)
        # R base (idx 1) -> FR joints (6)
        new_bj_front.extend([[0, 0], [1, 6]])
        new_jb_front.extend([[0, 0], [6, 1]])
        # L base (idx 0) -> RL joints (3)
        # R base (idx 1) -> RR joints (9)
        new_bj_back.extend([[0, 3], [1, 9]])
        new_jb_back.extend([[3, 0], [9, 1]])

        # create new base-base connections
        # gt_edge: connections between front bases (0-1) and between back bases (2-3)
        bb_edges = [[0, 1], [1, 0]]

        # convert to PyTorch tensors
        self.bb = torch.tensor(bb_edges, dtype=torch.long).t()
        self.bj_front = torch.tensor(new_bj_front, dtype=torch.long).t()
        self.jb_front = torch.tensor(new_jb_front, dtype=torch.long).t()
        self.bj_back = torch.tensor(new_bj_back, dtype=torch.long).t()
        self.jb_back = torch.tensor(new_jb_back, dtype=torch.long).t()
        self.jj = torch.tensor(jj, dtype=torch.long)
        self.fj = torch.tensor(fj, dtype=torch.long)
        self.jf = torch.tensor(jf, dtype=torch.long)

    def get_data_metadata(self):
        """Return metadata for the new graph structure"""
        # define node types
        node_types = ['base', 'joint', 'foot']
        
        # define edge types (including new connections)
        if self.model_type == 'heterogeneous_gnn_k4':
            edge_types = [
                ('base', 'connect', 'joint'),
                ('joint', 'connect', 'base'),
                ('joint', 'connect', 'joint'),
                ('foot', 'connect', 'joint'),
                ('joint', 'connect', 'foot'),
                ('base', 'gt', 'base'),    # new transverse connections
                ('base', 'gs', 'base'),    # new sagittal connections
            ]
        elif self.model_type == 'heterogeneous_gnn_c2':
            edge_types = [
                ('base', 'front_bj', 'joint'),
                ('joint', 'front_bj', 'base'),
                ('base', 'back_bj', 'joint'),
                ('joint', 'back_bj', 'base'),
                ('joint', 'connect', 'joint'),
                ('foot', 'connect', 'joint'),
                ('joint', 'connect', 'foot'),
                ('base', 'center_bb', 'base')
            ]
        
        return node_types, edge_types
    
    def get_helper_heterogeneous_gnn_c2(self, idx):
        """Extended data retrieval method, including new graph structure for C2"""
        # Create the Heterogeneous Data objects
        data = HeteroData()

        # set edge connections
        data['base', 'front_bj', 'joint'].edge_index = self.bj_front
        data['joint', 'front_bj', 'base'].edge_index = self.jb_front
        data['base', 'back_bj', 'joint'].edge_index = self.bj_back
        data['joint', 'back_bj', 'base'].edge_index = self.jb_back
        data['joint', 'connect', 'joint'].edge_index = self.jj
        data['foot', 'connect', 'joint'].edge_index = self.fj
        data['joint', 'connect', 'foot'].edge_index = self.jf
        data['base', 'center_bb', 'base'].edge_index = self.bb

         # create node feature matrices
        base_x = torch.ones((self.hgnn_number_nodes[0], self.base_width), dtype=torch.float64)
        joint_x = torch.ones((self.hgnn_number_nodes[1], self.joint_width), dtype=torch.float64)
        foot_x = torch.ones((self.hgnn_number_nodes[2], self.foot_width), dtype=torch.float64)

        # get original data
        lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_sorted_c2(idx)

        # For each base specified
        base_data = [lin_acc, ang_vel] # lin_acc, ang_vel shape: [history_length, 2*3]
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

        # For each foot specified
        foot_data = [f_p, f_v] # f_p, f_v shape: [history_length, 3*4]
        for i, urdf_node_name in enumerate(self.urdf_name_to_graph_index_foot.keys()):
            # For each variable to use
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_foot:
                final_input = torch.cat((final_input, torch.tensor(foot_data[k][:,(3*i):(3*i)+3].flatten('F'), dtype=torch.float64)), axis=0)
            if final_input.shape[0] != 0:
                foot_x[i] = final_input

        # set labels and node features
        data.y = torch.tensor(labels, dtype=torch.float64)
        data['base'].x = base_x
        data['joint'].x = joint_x
        data['foot'].x = foot_x
        data.num_nodes = sum(self.hgnn_number_nodes)

        if self.grf_body_to_world_frame:    
            # Use the world frame GRF as the target
            data.r_o = torch.tensor(r_o[-1], dtype=torch.float64) # shape:(4,)

        return data

    # ===================== Tools ===============================
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

# ================================================================
# ======================== SUBCLASSES ============================
# ================================================================
class QuadSDKDataset_A1(QuadSDKDataset_NewGraph):
    # ===================== DATASET PROPERTIES =======================
    def get_expected_urdf_name(self):
        return "a1_description"
    
    # ============= DATA SORTING ORDER AND MAPPINGS ==================    
    def get_urdf_name_to_dataset_array_index(self):
        # Our URDF order for the Go2 is FR, FL, RR, RL.

        return {
            'floating_base': 0,
            
            # Joint Order is specified here: https://github.com/robomechanics/quad-sdk/wiki/FAQ .
            # It goes abd->hip->knee, which in our framing, is hip->thigh->calf, and this matches our URDF order.
            # Leg order goes FL, RL, FR, RR, which matches our URDF order of FL, RL, FR, RR.
            # Further proof for this can be found in quad_sdk_fork/quad_simulator/gazebo_scripts/src/estimator_plugin.cpp.

            # FL Leg
            '8': 0,
            '0': 1,
            '1': 2,

            # RL Leg
            '9': 3,
            '2': 4,
            '3': 5,

            # FR Leg
            '10': 6,
            '4': 7,
            '5': 8,

            # RR Leg
            '11': 9,
            '6': 10,
            '7': 11,

            # Feet orders (which matches leg orders)
            # Manually verified by reviewing scripts in quad_sdk_fork/quad_simulator/gazebo_scripts/src.
            'jtoe0': 0,
            'jtoe1': 1,
            'jtoe2': 2,
            'jtoe3': 3,
        }

    # ======================== DATA LOADING ==========================
    def load_data_at_dataset_seq_3d(self, seq_num: int):
        """
        The units of the return values are as follows:
        - lin_acc (meters/sec^2, represented in the world frame (converted from body frame))
        - ang_vel (rad/sec, represented in the world frame (converted from body frame))
        - j_p (rad)
        - j_v (rad/sec)
        - j_T (Newton-meters)
        - z_grfs (Newtons, represented in the world frame)
        - r_p (meters, represented in the world frame)
        - r_quat (N/A (Quaternion), represented in the world frame) (x, y, z, w)
        - timestamps (secs)

        This dataset does have the following values available, but doesn't load
        them because they aren't employed by the Dynamics model, so we don't
        want them included in the learning model automatically.

        - f_p (meters, represented in robot's body frame) 
        - f_v (meters/sec, represented in robot's body frame)
        """
        # Convert them all to numpy arrays
        lin_acc = np.array(self.mat_data['imu_acc'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        ang_vel = np.array(self.mat_data['imu_omega'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        j_p = np.array(self.mat_data['q'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_v = np.array(self.mat_data['qd'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        j_T = np.array(self.mat_data['tau'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12)
        grfs = np.squeeze(np.array(self.mat_data['F'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 12))[-1]
        r_p = np.array(self.mat_data['r_p'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)
        r_quat = np.array(self.mat_data['r_o'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 4)
        timestamps = np.array(self.mat_data['timestamps'][seq_num:seq_num+self.history_length]).reshape(self.history_length, 3)

        world_to_body_R = Rotation.from_quat(r_quat[-1])
        grfs_T = np.array(grfs.reshape(4, 3), dtype=np.double).T
        grfs_body = (world_to_body_R.as_matrix() @ grfs_T).T
        grfs = grfs_body.flatten()

        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, grfs, r_p, r_quat, timestamps

    def load_data_at_dataset_seq(self, seq_num: int):
        z_indices = [2, 5, 8, 11]
        lin_acc, ang_vel, j_p, j_v, j_T, grfs, r_p, r_quat, timestamps = self.load_data_at_dataset_seq_3d(seq_num)
        z_grfs = grfs[z_indices]

        return lin_acc, ang_vel, j_p, j_v, j_T, None, None, z_grfs, r_p, r_quat, timestamps

# A1
class QuadSDKDataset_A1_Alpha(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/w2b7uk3sm3u42uxxnzm0o/robot_1_a1_0p5mps_flat_mu1_50_mu2_50_trial1_2024-09-09-14-22-23.bag?rlkey=435hyxui9wc2qt93sr1lq58az&st=yn0n3aol&dl=1", "Dropbox"

class QuadSDKDataset_A1_Bravo(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/j7y2ukxjv99c6ge357g39/robot_1_a1_0p5mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-37-30.bag?rlkey=osklprqilkxoxixes7xlflqgq&st=u3tyw691&dl=1", "Dropbox"

class QuadSDKDataset_A1_Charlie(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/3swmvf4zmqhqjtdi37e0i/robot_1_a1_0p5mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-04-36.bag?rlkey=pekugjjnexobbpb772qtoolky&st=tzxksa3k&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Delta(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/4syqkizbm9qthg7e35soh/robot_1_a1_0p5mps_slope_mu1_50_mu2_50_trial1_2024-09-09-15-29-14.bag?rlkey=rpztfeh8sufxmryb67rqgkzbx&st=0q0vsf76&dl=1", "Dropbox"

class QuadSDKDataset_A1_Echo(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/bkgasvttt5v7szcpjm350/robot_1_a1_0p5mps_slope_mu1_75_mu2_75_trial1_2024-09-09-15-46-45.bag?rlkey=eg33dr3brs03uvvsgmkkj0tn1&st=3n9ugtpc&dl=1", "Dropbox"

class QuadSDKDataset_A1_Foxtrot(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/i2rik324b5byqpedloaew/robot_1_a1_0p5mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-02-56.bag?rlkey=pm3z8wm2rxw7ubvtx88z1g827&st=022deyoe&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Golf(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Rough, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/2k2v2bydphna2qx00e4oz/robot_1_a1_0p5mps_rough_mu1_75_mu2_75_trial1_2024-09-10-15-57-56.bag?rlkey=lc4e7826ro0qo4z9ri6dnokox&st=5j01yfle&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Hotel(QuadSDKDataset_A1):
    # Speed: 0.5 mps, Terrain: Rough, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/luyqyl11x2nj3p52k0ddc/robot_1_a1_0p5mps_rough_mu1_100_mu2_100_trial1_2024-09-10-16-37-10.bag?rlkey=1j4hiy8f5qbfrf5ojwdyxvo81&st=woilhuqo&dl=1", "Dropbox"

class QuadSDKDataset_A1_India(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/f615giqzrgrt6rt8rxwn8/robot_1_a1_0p75mps_flat_mu1_50_mu2_50_trial1_2024-09-09-14-30-18.bag?rlkey=q4e35770xjjrm4c9p8dx85j5u&st=2ikap4wu&dl=1", "Dropbox"

class QuadSDKDataset_A1_Juliett(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/u2hlxjbb9dgg9ttq9kwl2/robot_1_a1_0p75mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-45-02.bag?rlkey=nyevs573wf273zefbvyvwvkq4&st=vlcau1i3&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Kilo(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/sp90ohnu0y2hrwv5hykol/robot_1_a1_0p75mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-11-33.bag?rlkey=y11sftmemzpymwfy2syijvt7y&st=kvu4ut66&dl=1", "Dropbox"

class QuadSDKDataset_A1_Lima(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/zbf9sxb5c6192xgtubb42/robot_1_a1_0p75mps_slope_mu1_50_mu2_50_trial1_2024-09-09-15-39-03.bag?rlkey=949w060ptq7dajgphly58ebou&st=o5eijrf5&dl=1", "Dropbox"    

class QuadSDKDataset_A1_Mike(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/7qwiwicd1x9h051rxyla0/robot_1_a1_0p75mps_slope_mu1_75_mu2_75_trial1_2024-09-09-16-11-33.bag?rlkey=7lmasurdpzlleojqltmikkin4&st=tj4hmxcx&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_November(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/50rwpce47tushv8gb0oew/robot_1_a1_0p75mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-10-33.bag?rlkey=ajgb069ku2pp2uua49nv54c9q&st=rrd9u414&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Oscar(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Rough, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/l2c5w2lccaaut1k77nm3i/robot_1_a1_0p75mps_rough_mu1_75_mu2_75_trial1_2024-09-10-16-11-05.bag?rlkey=m5j3ya1yuc9hjrx2axfjcw47d&st=hpf2lg6p&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Papa(QuadSDKDataset_A1):
    # Speed: 0.75 mps, Terrain: Rough, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/wz9gn4zps1781eyrt8huz/robot_1_a1_0p75mps_rough_mu1_100_mu2_100_trial1_2024-09-10-16-49-42.bag?rlkey=vcebsg42ggxeemaagodgsb1yr&st=openvcsi&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Quebec(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Flat, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/1lv5jxavljoep4dvlfwlb/robot_1_a1_1p0mps_flat_mu1_75_mu2_75_trial1_2024-09-09-14-57-16.bag?rlkey=9jryyynd345ad8lf4cmhdj247&st=7pm05bgq&dl=1", "Dropbox"

class QuadSDKDataset_A1_Romeo(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Flat, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/sds8jnzftbk6f0vreyl0n/robot_1_a1_1p0mps_flat_mu1_100_mu2_100_trial1_2024-09-09-15-21-52.bag?rlkey=wxfyc8npy2jg0u7ws66ekece8&st=ipnftg46&dl=1", "Dropbox"

class QuadSDKDataset_A1_Sierra(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Slope, Mu: 75
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/hbx5gsdkp845c060uor6z/robot_1_a1_1p0mps_slope_mu1_75_mu2_75_trial1_2024-09-09-18-54-59.bag?rlkey=xtuooopimk4cq27nxqjz4wdlw&st=3jkrpqdr&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Tango(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Slope, Mu: 100
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/84lv882akicv37hmab0d7/robot_1_a1_1p0mps_slope_mu1_100_mu2_100_trial1_2024-09-09-19-24-12.bag?rlkey=761zudvru1ctdsde51ipx1onn&st=esumu7ot&dl=1", "Dropbox"
    
class QuadSDKDataset_A1_Uniform(QuadSDKDataset_A1):
    # Speed: 1.00 mps, Terrain: Rough, Mu: 50
    def get_file_id_and_loc(self):
        return "https://www.dropbox.com/scl/fi/fswpwmltqgwxa8pwb1rif/robot_1_a1_1p0mps_rough_mu1_50_mu2_50_trial1_2024-09-10-17-19-24.bag?rlkey=gqjn6458dt5rbmhv1klymm6xl&st=f9obhple&dl=1", "Dropbox"
