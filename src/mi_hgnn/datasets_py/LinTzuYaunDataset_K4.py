from .LinTzuYaunDataset import LinTzuYaunDataset
import torch
from torch_geometric.data import HeteroData
import numpy as np

class LinTzuYaunDataset_NewGraph(LinTzuYaunDataset):
    """
    Extended LinTzuYaunDataset with new graph structure:
    - Split base node into 4 nodes (FL, FR, BL, BR base)
    - Add gt_edge between front bases and between back bases
    - Add gs_edge between left bases and between right bases
    """
    def __init__(self, root, path_to_urdf, urdf_package_name, urdf_package_relative_path, 
                 model_type, history_length, normalize=True, swap_legs=None):
        super().__init__(root, path_to_urdf, urdf_package_name, urdf_package_relative_path,
                        model_type, history_length, normalize, swap_legs)
        
        self.model_type = model_type
        if self.model_type == 'heterogeneous_gnn_k4':
            # modify the number of base nodes
            self.hgnn_number_nodes = (4, self.hgnn_number_nodes[1], self.hgnn_number_nodes[2])
            # initialize new edges
            self._init_new_edges()
    
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

        # copy base features to four base nodes
        base_data = [lin_acc, ang_vel]
        for i in range(self.hgnn_number_nodes[0]):
            final_input = torch.ones((0), dtype=torch.float64)
            for k in self.variables_to_use_base:
                final_input = torch.cat((final_input, torch.tensor(base_data[k][:,0:3].flatten('F'), dtype=torch.float64)), axis=0)
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
        foot_data = [f_p, f_v]
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