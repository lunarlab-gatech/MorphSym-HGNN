import torch
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, HeteroDictLinear, GraphConv
import yaml

class COM_HGNN_S4(torch.nn.Module):
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

        # NOTE: hardcoded for mini_cheetah
        self.num_timesteps = 150
        num_joints_per_leg = 3
        self.num_legs = 4
        self.num_bases = 1
        self.num_joints = self.num_legs * num_joints_per_leg
        self.num_dimensions_per_foot = 3
        self.num_dimensions_per_base = 6

        # Create the first layer encoder to convert features into embeddings
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create convolutions for each layer with special handling for gt and gs edges
        self.convs = torch.nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in data_metadata[1]:
                conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='add' # 'mean' or 'add'
                    )
            
            # Use sum to aggregate information from all edge types
            conv = HeteroConv(conv_dict, aggr='sum')
            self.convs.append(conv)

        # Output layer remains the same
        self.decoder = Linear(hidden_channels, self.num_dimensions_per_base)
        
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with special handling for the K4 structure
        """
        # x_dict = self.apply_symmetry(x_dict)

        # Initial feature encoding
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}
            
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.activation(x) for key, x in x_dict.items()}
        
        return self.decoder(x_dict['base'].view(x_dict['base'].shape[0], -1))
    
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