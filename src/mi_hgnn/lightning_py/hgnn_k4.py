import torch
from torch import nn
from torch_geometric.nn import Linear, HeteroConv, HeteroDictLinear, GraphConv

class GRF_HGNN_K4(torch.nn.Module):
    """
    Modified GRF_HGNN for the K4 graph structure with 4 base nodes and new edge types
    """
    def __init__(self, hidden_channels: int, num_layers: int, data_metadata, 
                 regression: bool = True, activation_fn = nn.ReLU()):
        """
        Implementation of the modified MI-HGNN model for K4 structure.

        Parameters:
            hidden_channels (int): Size of the node embeddings in the graph.
            num_layers (int): Number of message-passing layers.
            data_metadata (tuple): Contains information on the node and edge types.
            regression (bool): True if regression, false if classification.
            activation_fn (class): The activation function used between layers.
        """
        super().__init__()
        self.regression = regression
        self.activation = activation_fn

        # Create the first layer encoder to convert features into embeddings
        self.encoder = HeteroDictLinear(-1, hidden_channels, data_metadata[0])

        # Create convolutions for each layer with special handling for gt and gs edges
        self.convs = torch.nn.ModuleList()
        '''
        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in data_metadata[1]:
                # Use different aggregation methods for different edge types
                if edge_type[1] in ['gt', 'gs']:
                    # For new edges between base nodes, use a special aggregation method
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='mean',  # Use mean aggregation to handle symmetry
                    )
                else:
                    # For existing edges, keep the original processing
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='add'
                    )
        '''

        for _ in range(num_layers):
            conv_dict = {}
            for edge_type in data_metadata[1]:
                source_node, edge_name, target_node = edge_type
                
                if edge_name == 'gt':
                    # create independent convolution layers for each leg pair
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='mean'
                    )
                    
                elif edge_name == 'gs':
                    # create independent convolution layers for each leg pair
                    conv_dict[edge_type] = GraphConv(
                        hidden_channels,
                        hidden_channels,
                        aggr='mean'
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
        if self.regression:
            self.out_channels_per_foot = 1
        else:
            self.out_channels_per_foot = 2
        self.decoder = Linear(hidden_channels, self.out_channels_per_foot)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with special handling for the K4 structure
        """
        # Initial feature encoding
        x_dict = self.encoder(x_dict)
        x_dict = {key: self.activation(x) for key, x in x_dict.items()}

        # Message passing layers
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

        # Final prediction for foot nodes
        return self.decoder(x_dict['foot'])

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
        for node_type in self.encoder.weight_dict:
            print(f"{node_type}: {self.encoder.weight_dict[node_type].shape}")
    
        # check convolution parameters
        print("\nConvolution parameters for different edge types:")
        for i, conv in enumerate(self.convs):
            print(f"Layer {i}:")
            for edge_type in conv.convs:
                print(f"{edge_type}: {conv.convs[edge_type].lin.weight.shape}")