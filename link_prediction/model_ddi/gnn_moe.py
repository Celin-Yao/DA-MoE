import torch
from model_ddi.moe import MoE_OGB

    
class GNN_Moe(torch.nn.Module):
    def __init__(self,gnn_type,gate_type, in_channels, hidden_channels, out_channels, num_layers,min_layers,topK,coef,
                 dropout):
        super().__init__()
        self.gnn_node_moe = MoE_OGB(gnn_type,gate_type,in_channels,hidden_channels,out_channels,num_layers,min_layers,k=topK, coef=coef,dropout = dropout)

    def forward(self, x, adj_t):
        x,self.load_balance_loss = self.gnn_node_moe(x,adj_t)
        return x
