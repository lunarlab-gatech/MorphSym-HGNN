import torch
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, HeteroDictLinear, GraphConv
import yaml

class COM_HGNN_K4(torch.nn.Module):
    """
    Modified GRF_HGNN for the K4 graph structure with 4 base nodes and new edge types
    """
    def __init__(self, hidden_channels: int, num_layers: int, data_metadata, 
                 regression: bool = True, activation_fn = nn.ReLU(), symmetry_mode: str = None, group_operator_path: str = None):
        """
        Implementation of the modified MI-HGNN model for K4 structure.

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            data_metadata (tuple): Contains information on the node and edge types.
            regression (bool): True if regression, false if classification.
            activation_fn (class): The activation function used between layers.
            symmetry_mode (str): The symmetry mode used for the model.
            group_operator_path (str): The path to the group operator file.
        """
        super().__init__()
        self.regression = regression
        self.activation = activation_fn

        # NOTE: hardcoded for Solo CoM Exp
        self.num_timesteps = 1
        num_joints_per_leg = 3
        self.num_legs = 4
        self.num_bases = 4
        self.num_joints = self.num_legs * num_joints_per_leg
        self.num_dimensions_per_base = 6

        # Initialize the joint coefficients based on the symmetry mode and group operator path
        if symmetry_mode and group_operator_path:
            with open(group_operator_path, 'r') as file:
                group_data = yaml.safe_load(file)

            reflection_Q_js = group_data.get('reflection_Q_js', [])
            j_gs_coeffs = torch.tensor(reflection_Q_js[0][:num_joints_per_leg], dtype=torch.float64) # sagittal symmetry, shape [num_joints_per_leg]
            j_gt_coeffs = torch.tensor(reflection_Q_js[1][:num_joints_per_leg], dtype=torch.float64) # transversal symmetry, shape [num_joints_per_leg]
            j_e_coeffs = torch.ones_like(j_gs_coeffs, dtype=torch.float64) # rotational symmetry, shape [num_joints_per_leg]
            j_gr_coeffs = j_gs_coeffs * j_gt_coeffs

            reflection_Q_bs_lin = group_data.get('reflection_Q_bs_lin', [])
            b_gs_coeffs_lin = torch.tensor(reflection_Q_bs_lin[0][:self.num_dimensions_per_base // 2], dtype=torch.float64) # sagittal symmetry, shape [3]
            b_gt_coeffs_lin = torch.tensor(reflection_Q_bs_lin[1][:self.num_dimensions_per_base // 2], dtype=torch.float64) # transversal symmetry, shape [3]
            b_e_coeffs_lin = torch.ones_like(b_gs_coeffs_lin, dtype=torch.float64) # rotational symmetry, shape [3]
            b_gr_coeffs_lin = b_gs_coeffs_lin * b_gt_coeffs_lin

            reflection_Q_bs_ang = group_data.get('reflection_Q_bs_ang', [])
            b_gs_coeffs_ang = torch.tensor(reflection_Q_bs_ang[0][:self.num_dimensions_per_base // 2], dtype=torch.float64) # sagittal symmetry, shape [3]
            b_gt_coeffs_ang = torch.tensor(reflection_Q_bs_ang[1][:self.num_dimensions_per_base // 2], dtype=torch.float64) # transversal symmetry, shape [3]
            b_e_coeffs_ang = torch.ones_like(b_gs_coeffs_ang, dtype=torch.float64) # rotational symmetry, shape [3]
            b_gr_coeffs_ang = b_gs_coeffs_ang * b_gt_coeffs_ang
        else:
            j_gs_coeffs = torch.ones(num_joints_per_leg, dtype=torch.float64)
            j_gt_coeffs = torch.ones(num_joints_per_leg, dtype=torch.float64)
            j_gr_coeffs = torch.ones(num_joints_per_leg, dtype=torch.float64)
            j_e_coeffs = torch.ones(num_joints_per_leg, dtype=torch.float64)

            b_gs_coeffs_lin = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_gt_coeffs_lin = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_gr_coeffs_lin = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_e_coeffs_lin = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)

            b_gs_coeffs_ang = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_gt_coeffs_ang = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_gr_coeffs_ang = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
            b_e_coeffs_ang = torch.ones(self.num_dimensions_per_base // 2, dtype=torch.float64)
        # joints = [Back_Left, Frong_Left, Back_Right, Front_Right]
        joint_weights_array = torch.cat((j_e_coeffs, j_gt_coeffs, j_gs_coeffs, j_gr_coeffs), dim=0)
        self.joints_linear_weights = joint_weights_array
        print(f'===> self.joints_linear_weights: {self.joints_linear_weights}')

        base_weights_lin_array = torch.cat((b_e_coeffs_lin, b_gt_coeffs_lin, b_gs_coeffs_lin, b_gr_coeffs_lin), dim=0)
        self.base_coefficients_lin = base_weights_lin_array
        base_weights_ang_array = torch.cat((b_e_coeffs_ang, b_gt_coeffs_ang, b_gs_coeffs_ang, b_gr_coeffs_ang), dim=0)
        self.base_coefficients_ang = base_weights_ang_array
        print(f'===> self.base_coefficients_lin: {self.base_coefficients_lin}')
        print(f'===> self.base_coefficients_ang: {self.base_coefficients_ang}')

        # Create the first layer encoder to convert features into embeddings
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create convolutions for each layer with special handling for gt and gs edges
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in data_metadata[1]:
                source_node, edge_name, target_node = edge_type
                
                if edge_name in ['gt', 'gs']:
                    # create independent convolution layers for each leg pair
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='mean' # 'mean' or 'add'
                    )
                else:
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='add'
                    )
            
            # Use sum to aggregate information from all edge types
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Add an extra processing layer for base nodes (optional)
        self.base_transform = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Output layer remains the same
        self.decoder = Linear(hidden_channels, self.num_dimensions_per_base)
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with special handling for the K4 structure
        """
        x_dict = self.apply_symmetry(x_dict)

        # Initial feature encoding
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # Message passing layers, with the base transformation and residual connections
        for conv in self.convs:
            # Apply convolution
            x_dict_new = conv(x_dict, edge_index_dict)
            
            # Apply activation and optional transformations
            x_dict_new = {
                key: (self.base_transform(x) if key == 'base' else self.activation(x))
                for key, x in x_dict_new.items()
            }
            
            # Optional: Add residual connections
            x_dict = {
                key: x_dict_new[key] + x_dict[key] 
                if key in x_dict and x_dict[key].shape == x_dict_new[key].shape
                else x_dict_new[key]
                for key in x_dict_new
            }

        out = self.decoder(x_dict['base'])
        return self.morphological_symmetry_decoder(out)
    
    def morphological_symmetry_decoder(self, x):
        batch_size = x.shape[0] // self.num_bases
        x = x.reshape(batch_size, self.num_bases, self.num_dimensions_per_base)
        x_lin = x[:, :, :3].flatten(start_dim=1) * self.base_coefficients_lin.to(x.device).view(1, -1)
        x_lin = x_lin.reshape(batch_size, self.num_bases, 3)
        x_ang = x[:, :, 3:].flatten(start_dim=1) * self.base_coefficients_ang.to(x.device).view(1, -1)
        x_ang = x_ang.reshape(batch_size, self.num_bases, 3)
        x = torch.cat((x_lin, x_ang), dim=-1)
        return x
    
    def apply_symmetry(self, x_dict):
        """
        Apply the symmetry to the node features
        """
        # Apply morphological symmetry to the joint features
        joint_x = x_dict['joint']  # shape: [batch_size * num_joints, num_timesteps * num_variables]
        joint_x = joint_x.view(-1, self.num_joints, 2)
        weights_j = self.joints_linear_weights.to(joint_x.device).view(1, -1, 1)
        joint_x = joint_x * weights_j
        x_dict['joint'] = joint_x.reshape(-1, 2)
        return x_dict


    def reset_parameters(self):
        """Reset all learnable parameters"""
        self.encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.base_transform[0].reset_parameters()
        self.base_transform[2].reset_parameters()
        self.decoder.reset_parameters()

    def check_parameter_sharing(self):
        # check encoder parameters
        print("Encoder parameters for different node types:")
        for node_type in self.encoder.lins:
            print(f"{node_type}: {self.encoder.lins[node_type].weight}")
    
        # check convolution parameters
        print("\nConvolution parameters for different edge types:")
        for i, conv in enumerate(self.convs):
            print(f"Layer {i}:")
            for edge_type in conv.convs:
                print(f"{edge_type}: {conv.convs[edge_type].lin_rel.weight}")

    def visualize_gnn_structure(self, save_dir=None):
        """
        Visualize the GNN architecture including all layers and transformations
        
        Parameters:
            save_dir (str or Path, optional): Directory to save the figure. If None, only displays the figure.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        G = nx.DiGraph()
        pos = {}
        layer_spacing = 3
        node_spacing = 1
        
        # Add the input layer nodes
        input_types = ['base', 'joint', 'foot']
        for i, node_type in enumerate(input_types):
            node_name = f"input_{node_type}"
            G.add_node(node_name, color='lightblue', layer='input')
            pos[node_name] = (0, (len(input_types)-1)/2 - i)
        
        # Add the encoder layer nodes
        for i, node_type in enumerate(input_types):
            node_name = f"encoded_{node_type}"
            G.add_node(node_name, color='lightgreen', layer='encoder')
            pos[node_name] = (layer_spacing, (len(input_types)-1)/2 - i)
            G.add_edge(f"input_{node_type}", node_name, 
                    label=f"Linear\n{self.encoder.lins[node_type].weight.shape}")
        
        # Add the message passing layer nodes
        for layer_idx in range(len(self.convs)):
            x_pos = layer_spacing * (2 + layer_idx)
            
            # Add a node for each node type
            for i, node_type in enumerate(input_types):
                node_name = f"layer{layer_idx}_{node_type}"
                G.add_node(node_name, color='pink', layer=f'conv{layer_idx}')
                pos[node_name] = (x_pos, (len(input_types)-1)/2 - i)
                
                # Add edges from the previous layer
                prev_layer = 'encoded' if layer_idx == 0 else f'layer{layer_idx-1}'
                G.add_edge(f"{prev_layer}_{node_type}", node_name)
            
            # Add the message passing edges
            for edge_type in self.convs[layer_idx].convs:
                source, rel, target = edge_type
                edge_name = f"layer{layer_idx}_{source}_{rel}_{target}"
                G.add_edge(f"layer{layer_idx}_{source}", 
                        f"layer{layer_idx}_{target}",
                        label=rel)
        
        # Add the decoder layer (only for foot nodes)
        output_node = "output"
        G.add_node(output_node, color='lightblue', layer='output')
        pos[output_node] = (layer_spacing * (2 + len(self.convs)), 0)
        last_layer = f"layer{len(self.convs)-1}"
        G.add_edge(f"{last_layer}_foot", output_node, 
                label=f"Linear\n{self.decoder.weight.shape}")
        
        # Draw the graph
        plt.figure(figsize=(20, 10))
        
        # Draw the nodes
        node_colors = [G.nodes[node]['color'] for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1000)
        
        # Draw the edges
        nx.draw_networkx_edges(G, pos)
        
        # Add the node labels
        nx.draw_networkx_labels(G, pos)
        
        # Add the edge labels
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        # Add the layer labels
        layers = ['Input', 'Encoder'] + [f'Conv {i}' for i in range(len(self.convs))] + ['Output']
        for i, layer in enumerate(layers):
            plt.text(layer_spacing * i, 2, layer, 
                    horizontalalignment='center', fontsize=12)
        
        plt.title("GRF_HGNN_K4 Architecture")
        plt.axis('off')
        plt.tight_layout()
        
        if save_dir is not None:
            # Convert save_dir to Path object if it's a string
            save_dir = Path(save_dir)
            # Create directory if it doesn't exist
            save_dir.mkdir(parents=True, exist_ok=True)
            # Save figure
            plt.savefig(save_dir / 'gnn_structure.png', bbox_inches='tight', dpi=300)
        
        plt.show()

    def visualize_message_passing(self, x_dict, edge_index_dict):
        """
        Debug use, visualize the message passing process for a specific input.
        Use this function at the beginning of the forward pass to visualize the message passing process.
        """
        print("\nMessage Passing Process:")
        
        # Print the input features
        print("\nInput Features:")
        for node_type, features in x_dict.items():
            print(f"{node_type}: shape {features.shape}")
        
        # Print the encoded features
        encoded = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        print("\nEncoded Features:")
        for node_type, features in encoded.items():
            print(f"{node_type}: shape {features.shape}")
        
        # Print the message passing process
        x = encoded
        for i, conv in enumerate(self.convs):
            print(f"\nLayer {i} Message Passing:")
            
            # Print the message passing process for each edge type
            for edge_type in edge_index_dict:
                source, rel, target = edge_type
                print(f"\nEdge type: {source}->{rel}->{target}")
                print(f"Number of edges: {edge_index_dict[edge_type].shape[1]}")
                
                # Calculate the message for this edge type
                # conv_layer = conv.convs[edge_type]
                # source_features = x[source]
                # messages = conv_layer(source_features, edge_index_dict[edge_type])
                # print(f"Message shape: {messages.shape}")
                # print(f"Message stats: mean={messages.mean().item():.4f}, std={messages.std().item():.4f}")
            
            # Update the node features
            x = conv(x, edge_index_dict)
            print("\nUpdated node features:")
            for node_type, features in x.items():
                print(f"{node_type}: shape {features.shape}")