import torch
from torch_geometric.nn import MLP
from model.moe import MoE
from model.moe_ogb import MoE_OGB

class GNN_Moe(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,topK,model_type,gate_type,min_layers,coef,fin_mlp_layers,dropout,device):
        super().__init__()
        self.moe = MoE(model_type,gate_type,in_channels,hidden_channels,num_layers,min_layers,device,k=topK,coef=coef)
        if model_type == 'GatedGCN':
            self.mlp = MLP([in_channels, hidden_channels, out_channels],num_layers=fin_mlp_layers,
                norm=None, dropout=dropout)
        else:
            self.mlp = MLP([hidden_channels, hidden_channels, out_channels],num_layers=fin_mlp_layers,
                        norm=None, dropout=dropout)

    def forward(self, x, edge_index, batch):

        x,self.load_balance_loss=self.moe(x,batch,edge_index)
        # x = global_add_pool(x, batch)
        return self.mlp(x)


class GNN_Moe_OGB(torch.nn.Module):

    def __init__(self, device,num_tasks, num_layer = 5, emb_dim = 300, topK=4,min_layers=2,coef=0.001,
                    gnn_type = 'gin',gate_type='GIN', drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):

        super(GNN_Moe_OGB, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.graph_pooling = graph_pooling


        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node embeddings
        self.gnn_node_moe = MoE_OGB(gnn_type,gate_type,emb_dim,num_layer,min_layers,device,k=topK, dropout = drop_ratio)
            

        if graph_pooling == "set2set":
            self.graph_pred_linear = torch.nn.Linear(2*self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def forward(self, batched_data):
        h_graph,self.load_balance_loss = self.gnn_node_moe(batched_data)

        return self.graph_pred_linear(h_graph)
