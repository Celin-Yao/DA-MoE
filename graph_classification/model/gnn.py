import torch
from torch_geometric.nn import MLP, GINConv,GCNConv,GatedGraphConv, global_add_pool,global_mean_pool,global_max_pool,GlobalAttention,Set2Set
from model.conv_ogb import GNN_node,GNN_node_Virtualnode

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout,fin_mlp_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels],num_layers=2)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None,num_layers=fin_mlp_layers, dropout=dropout)
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,num_layers,dropout,fin_mlp_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_channels,hidden_channels))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],num_layers=fin_mlp_layers,
                       norm=None, dropout=dropout)
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return self.mlp(x)


class GatedGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,dropout,fin_mlp_layers):
        super().__init__()

        self.conv = GatedGraphConv(in_channels, num_layers)

        self.mlp = MLP([in_channels, hidden_channels, out_channels],
                       norm=None, dropout=dropout,num_layers=fin_mlp_layers)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = global_add_pool(x, batch)
        return self.mlp(x)



class GNN_OGB(torch.nn.Module):

    def __init__(self, num_tasks, num_layer = 5, emb_dim = 300, 
                    gnn_type = 'gin',virtual_node = True, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GNN_OGB, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        

        ### GNN to generate node embeddings
        if virtual_node:
            self.gnn_node = GNN_node_Virtualnode(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
        else:
            self.gnn_node = GNN_node(num_layer, emb_dim, JK = JK, drop_ratio = drop_ratio, residual = residual, gnn_type = gnn_type)
            

        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "attention":
            self.pool = GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, 1)))
        elif self.graph_pooling == "set2set":
            self.pool = Set2Set(emb_dim, processing_steps = 2)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return self.graph_pred_linear(h_graph)

