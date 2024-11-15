from .LinTzuYaunDataset import LinTzuYaunDataset
import torch
from torch_geometric.data import HeteroData
import numpy as np
import yaml
class LinTzuYaunDataset_NewGraph(LinTzuYaunDataset):
    """
    Extended LinTzuYaunDataset with new graph structure:
    - Split base node into 4 nodes (FL, FR, BL, BR base)
    - Add gt_edge between front bases and between back bases
    - Add gs_edge between left bases and between right bases

    Parameters (NEW):
        swap_legs (tuple | None): Either None, a tuple of (leg1_idx, leg2_idx), or a tuple of tuples
            e.g. None means no swap, (0,1) means swap FR and FL legs,
            ((0,1), (2,3)) means swap FR-FL and RR-RL legs
        symmetry_operator (str): The symmetry operator to use for the MorphSym symmetry mode.
            Can be 'gs' or 'gt' or 'gr' or None.
        symmetry_mode (str): The mode to use for symmetry. Either 'Euclidean' or 'MorphSym'.
        group_operator_path (Path): The path to the group operator file. Only used if symmetry_operator is not None.

        NOTE: 
            * If symmetry_operator is not None, then symmetry_mode must be 'MorphSym' or 'Euclidean'.
            * swap_legs is not None and symmetry_operator is not None is NOT supported.
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
                 group_operator_path=None):
        
        # Check for swap legs and symmetry operator compatibility
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
                elif self.symmetry_mode == 'Euclidean':
                    self.joint_coefficients = self.create_coefficient_dict(self.reflection_Q_js)
                    self.foot_coefficients = self.create_coefficient_dict(self.reflection_Q_fs)
                    self.base_coefficients_lin = self.create_coefficient_dict(self.reflection_Q_bs_lin)
                    self.base_coefficients_ang = self.create_coefficient_dict(self.reflection_Q_bs_ang)
                    self.label_coefficients = self.create_coefficient_dict(self.reflection_Q_ls)
            except FileNotFoundError:
                raise ValueError(f"Group operator file not found at {self.group_operator_path}")
            except yaml.YAMLError as e:
                raise ValueError(f"Error parsing YAML file: {e}")

        super().__init__(root, path_to_urdf, urdf_package_name, urdf_package_relative_path,
                        model_type, history_length, normalize=normalize)
        
        self.model_type = model_type
        if self.model_type == 'heterogeneous_gnn_k4':
            # modify the number of base nodes
            self.hgnn_number_nodes = (4, self.hgnn_number_nodes[1], self.hgnn_number_nodes[2])
            # initialize new edges
            self._init_new_edges()

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
            lin_acc: shape: [history_length, 4*3] 4 bases linear acceleration
            ang_vel: shape: [history_length, 4*3] 4 bases angular velocity
            j_p: shape: [history_length, 12] 12 joints position
            j_v: shape: [history_length, 12] 12 joints velocity
            j_T: None
            f_p: shape: [history_length, 4*3] 4 feet position
            f_v: shape: [history_length, 4*3] 4 feet velocity
            labels: shape: [4] 4 labels
            r_p: None
            r_o: None
            timestamps: shape: [history_length]
        """
        if self.swap_legs is not None:
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq_with_swap(seq_num, self.swap_legs)
        else:
            lin_acc, ang_vel, j_p, j_v, j_T, f_p, f_v, labels, r_p, r_o, timestamps = self.load_data_at_dataset_seq(seq_num)

        # duplicate the base information for each base node, no need to sort
        # lin_acc, ang_vel shape: [history_length, 3] ==> [history_length, 4*3]
        lin_acc = np.tile(lin_acc, (1, 4))
        ang_vel = np.tile(ang_vel, (1, 4))
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
        
        if self.symmetry_operator is not None:
            sorted_foot_list = self.apply_symmetry(sorted_foot_list, part='foot')

        # Sort the ground truth labels
        labels_sorted = None
        if labels is None:
            raise ValueError("Dataset must provide labels.")
        else:
            labels_sorted = labels[self.foot_node_indices_sorted]

        if self.symmetry_operator is not None:
            labels_sorted = self.apply_symmetry([labels_sorted], part='label')[0]

        # Normalize the data if desired
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
                # NOTE: debug use, only apply Euclidean symmetry to foot
                # old_gs = self.joint_coefficients['gs']
                # coefficients = {
                #     'gs': np.ones_like(old_gs, dtype=np.float64),
                #     'gt': np.ones_like(old_gs, dtype=np.float64),
                #     'gr': np.ones_like(old_gs, dtype=np.float64)
                # }
            elif part == 'foot':
                permutation_Q = self.permutation_Q_fs
                coefficients = self.foot_coefficients
                # NOTE: debug use, only apply Euclidean symmetry to foot
                # old_gs = self.foot_coefficients['gs']
                # coefficients = {
                #     'gs': np.ones_like(old_gs, dtype=np.float64),
                #     'gt': np.ones_like(old_gs, dtype=np.float64),
                #     'gr': np.ones_like(old_gs, dtype=np.float64)
                # }
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
    
    def _init_new_edges(self):
        """Initialize the new edge structure"""
        # original edges
        bj, jb, jj, fj, jf = self.robotGraph.get_edge_index_matrices()
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.robotGraph.get_edge_attr_matrices()

        # create new base-joint connections
        new_bj = []
        new_jb = []
        
        # Assign corresponding joints to each base node
        # FR base (idx 0) -> FR joints (0)
        # FL base (idx 1) -> FL joints (3)
        # BR base (idx 2) -> BR joints (6)
        # BL base (idx 3) -> BL joints (9)
        for base_idx in range(self.hgnn_number_nodes[0]):
            joint_hip = base_idx * 3
            new_bj.extend([[base_idx, joint_hip]])
            new_jb.extend([[joint_hip, base_idx]])

        # create new base-base connections
        # gt_edge: connections between front bases (0-1) and between back bases (2-3)
        gt_edges = [[0, 1], [1, 0], [2, 3], [3, 2]]
        
        # gs_edge: connections between left bases (1-3) and between right bases (0-2)
        gs_edges = [[0, 2], [2, 0], [1, 3], [3, 1]]

        # convert to PyTorch tensors
        self.bj = torch.tensor(new_bj, dtype=torch.long).t()
        self.jb = torch.tensor(new_jb, dtype=torch.long).t()
        self.jj = torch.tensor(jj, dtype=torch.long)
        self.fj = torch.tensor(fj, dtype=torch.long)
        self.jf = torch.tensor(jf, dtype=torch.long)
        self.gt = torch.tensor(gt_edges, dtype=torch.long).t()
        self.gs = torch.tensor(gs_edges, dtype=torch.long).t()

        # edge attributes, shape: [2, E], where E is the number of edges
        bj_attr, jb_attr, jj_attr, fj_attr, jf_attr = self.robotGraph.get_edge_attr_matrices()
        self.bj_attr = torch.tensor(bj_attr, dtype=torch.float64)
        self.jb_attr = torch.tensor(jb_attr, dtype=torch.float64)
        self.jj_attr = torch.tensor(jj_attr, dtype=torch.float64)
        self.fj_attr = torch.tensor(fj_attr, dtype=torch.float64)
        self.jf_attr = torch.tensor(jf_attr, dtype=torch.float64)

        N = 7
        self.gt_attr = np.zeros((self.gt.size(1), N))
        self.gs_attr = np.zeros((self.gs.size(1), N))
        
        # Assume we know the robot's dimensions #TODO: load from urdf
        robot_width = 0.4   # robot width
        robot_length = 0.6  # robot length
        
        # TODO: check if the following is correct
        # transverse edges - based on width
        for i in range(self.gt.size(1)):
            distance_factor = 1.0 / robot_width
            self.gt_attr[i] = torch.tensor([
                distance_factor,  # mass scaling
                distance_factor,  # Ixx
                0.0,            # Ixy
                0.0,            # Ixz
                distance_factor, # Iyy
                0.0,            # Iyz
                distance_factor  # Izz
            ])
        
        # sagittal edges - based on length
        for i in range(self.gs.size(1)):
            distance_factor = 1.0 / robot_length
            self.gs_attr[i] = torch.tensor([
                distance_factor,  # mass scaling
                distance_factor,  # Ixx
                0.0,            # Ixy
                0.0,            # Ixz
                distance_factor, # Iyy
                0.0,            # Iyz
                distance_factor  # Izz
            ])

        self.gt_attr = torch.tensor(self.gt_attr, dtype=torch.float64)
        self.gs_attr = torch.tensor(self.gs_attr, dtype=torch.float64)

    def get_data_metadata(self):
        """Return metadata for the new graph structure"""
        # define node types
        node_types = ['base', 'joint', 'foot']
        
        # define edge types (including new connections)
        edge_types = [
            ('base', 'connect', 'joint'),
            ('joint', 'connect', 'base'),
            ('joint', 'connect', 'joint'),
            ('foot', 'connect', 'joint'),
            ('joint', 'connect', 'foot'),
            ('base', 'gt', 'base'),    # new transverse connections
            ('base', 'gs', 'base'),    # new sagittal connections
        ]
        
        return node_types, edge_types

    def get_helper_heterogeneous_gnn(self, idx):
        """Extended data retrieval method, including new graph structure"""

        # Create the Heterogeneous Data objects
        data = HeteroData()

        # set edge connections
        data['base', 'connect', 'joint'].edge_index = self.bj
        data['joint', 'connect', 'base'].edge_index = self.jb
        data['joint', 'connect', 'joint'].edge_index = self.jj
        data['foot', 'connect', 'joint'].edge_index = self.fj
        data['joint', 'connect', 'foot'].edge_index = self.jf
        data['base', 'gt', 'base'].edge_index = self.gt
        data['base', 'gs', 'base'].edge_index = self.gs

        # set edge attributes
        data['base', 'connect', 'joint'].edge_attr = self.bj_attr
        data['joint', 'connect', 'base'].edge_attr = self.jb_attr
        data['joint', 'connect', 'joint'].edge_attr = self.jj_attr
        data['foot', 'connect', 'joint'].edge_attr = self.fj_attr
        data['joint', 'connect', 'foot'].edge_attr = self.jf_attr
        data['base', 'gt', 'base'].edge_attr = self.gt_attr
        data['base', 'gs', 'base'].edge_attr = self.gs_attr

        # create node feature matrices
        base_x = torch.ones((self.hgnn_number_nodes[0], self.base_width), dtype=torch.float64)
        joint_x = torch.ones((self.hgnn_number_nodes[1], self.joint_width), dtype=torch.float64)
        foot_x = torch.ones((self.hgnn_number_nodes[2], self.foot_width), dtype=torch.float64)

        # get original data
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

        return data

    def visualize_graph_structure(self, data):
        """
        Visualize the heterogeneous graph structure using networkx
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        
        # Create a new graph
        G = nx.Graph()
        
        # Add nodes with different colors for different types
        # Base nodes (4 nodes)
        base_positions = {
            1: (-1, 1),   # FL
            3: (1, 1),    # FR
            0: (-1, -1),  # BL
            2: (1, -1)    # BR
        }
        for i in range(4):
            G.add_node(f'base_{i}', color='lightblue', pos=base_positions[i])
        
        # Joint nodes (12 nodes, 3 for each leg)
        joint_positions = {}
        for i in range(12):
            leg_idx = i // 3
            joint_idx = i % 3
            base_pos = base_positions[leg_idx]
            # Position joints between base and foot
            x = base_pos[0] * (0.7 - joint_idx * 0.2)
            y = base_pos[1] * (0.7 - joint_idx * 0.2)
            joint_positions[i] = (x, y)
            G.add_node(f'joint_{i}', color='lightgreen', pos=joint_positions[i])
        
        # Foot nodes (4 nodes)
        foot_positions = {}
        for i in range(4):
            base_pos = base_positions[i]
            foot_positions[i] = (base_pos[0] * 0.3, base_pos[1] * 0.3)
            G.add_node(f'foot_{i}', color='pink', pos=foot_positions[i])
        
        # Add edges
        # Connect edges
        for i, j in data['base', 'connect', 'joint'].edge_index.t():
            G.add_edge(f'base_{i.item()}', f'joint_{j.item()}', color='gray')
        for i, j in data['joint', 'connect', 'foot'].edge_index.t():
            G.add_edge(f'joint_{i.item()}', f'foot_{j.item()}', color='gray')
        
        # GT edges (between front/back bases)
        for i, j in data['base', 'gt', 'base'].edge_index.t():
            G.add_edge(f'base_{i.item()}', f'base_{j.item()}', color='red')
        
        # GS edges (between left/right bases)
        for i, j in data['base', 'gs', 'base'].edge_index.t():
            G.add_edge(f'base_{i.item()}', f'base_{j.item()}', color='blue')
        
        # Draw the graph
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
        
        # Draw edges with different colors
        edge_colors = [G[u][v]['color'] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)
        
        # Add labels
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Heterogeneous Graph Structure")
        plt.axis('off')
        plt.show()


class LinTzuYaunDataset_air_jumping_gait(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "17h4kMUKMymG_GzTZTMHPgj-ImDKZMg3R", "Google"
    
class LinTzuYaunDataset_air_walking_gait(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "17c_E-S_yTeeV_DCmcgVT7_J90cRIwg0z", "Google"
    
class LinTzuYaunDataset_asphalt_road(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1jty0yqd7gywNJEkS_V2hivZ-79BGuCgA", "Google"
    
class LinTzuYaunDataset_old_asphalt_road(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1Y4SHVLqQKQ14leBdpfEQv1Tq5uQUIEK8", "Google"
    
class LinTzuYaunDataset_concrete_right_circle(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1NnEnd0PFFT6XozErUNi3ORGVSuFkyjeJ", "Google"
    
class LinTzuYaunDataset_concrete_pronking(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1XWdEIKUtFKmZd9W5M7636-HVdusqglhd", "Google"

class LinTzuYaunDataset_concrete_left_circle(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1K9hUMqc0oBCv6VtgS0rYXbRjq9XiFOv5", "Google"
    
class LinTzuYaunDataset_concrete_galloping(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1oInoPLowARNsL0h_qPVgjLCLICR7zw7W", "Google"
    
class LinTzuYaunDataset_concrete_difficult_slippery(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1i7MNbJNCBkIfW5TOU94YHnb5G0jXkSAf", "Google"
    
class LinTzuYaunDataset_forest(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1qMriGIWAUXFN3a-ewfdVAZlDsi_jZRNi", "Google"
    
class LinTzuYaunDataset_grass(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1yVRmhPZN6wpKhsT947Jkr8mlels8WM7m", "Google"
    
class LinTzuYaunDataset_middle_pebble(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "128UAFroCGekx-Ibk-zEAGYlq8mekdzOI", "Google"
    
class LinTzuYaunDataset_rock_road(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1Hyo9UQkmAGrA0r49jZgVTAOe40SgnlfU", "Google"
    
class LinTzuYaunDataset_sidewalk(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1D1vAmruuZE5KQH8gA_pDhfETHPMhiu2c", "Google"
    
class LinTzuYaunDataset_small_pebble(LinTzuYaunDataset_NewGraph):
    def get_file_id_and_loc(self):
        return "1cmjzHD9CKAXmKxZkDbPsEPKGvDI5Grec", "Google"