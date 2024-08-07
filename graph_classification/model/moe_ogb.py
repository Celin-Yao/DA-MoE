import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch_geometric.nn import MLP, GINEConv,GATConv,GCNConv,global_mean_pool,global_add_pool,global_max_pool,GlobalAttention,Set2Set
from model.conv_ogb import GNN_node,GNN_node_Virtualnode
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

class GNN(torch.nn.Module):

    def __init__(self, emb_dim = 300,num_layer = 3, gnn_type ='gin',
                    virtual_node = False, residual = False, drop_ratio = 0.5, JK = "last", graph_pooling = "mean"):

        super(GNN, self).__init__()

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
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


    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)

        h_graph = self.pool(h_node, batched_data.batch)

        return h_graph



class MoE_OGB(nn.Module):
    def __init__(self,model_type,gate_type,input_size,num_experts,min_layers,device, noisy_gating=True, k=4 , coef=1e-3,dropout=0.5,gate_dropout=0.2,heads=4):
        super(MoE_OGB, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        self.device=device
        self.gate_dropout=gate_dropout
        self.gate_type=gate_type
        # instantiate experts
        self.experts = torch.nn.ModuleList()
        self.atom_encoder = AtomEncoder(emb_dim = input_size)
        self.bond_encoder = BondEncoder(emb_dim = input_size)

        # model        
        for i in range(min_layers,num_experts+min_layers):
            input_channel = input_size
            self.experts.append(GNN(input_channel,i,model_type,drop_ratio=dropout))


        #gate
        if gate_type == 'liner':
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        elif gate_type =='GIN':
            self.gate_model=nn.ModuleList()
            input_channel=input_size
            for _ in range(2):
                mlp = MLP([input_channel, num_experts, num_experts],num_layers=2)
                self.gate_model.append(GINEConv(nn=mlp, train_eps=False,edge_dim=input_size))
                input_channel = num_experts
        elif gate_type =='GCN':
            self.gate_model=nn.ModuleList()
            input_channel=input_size
            for _ in range(2):
                self.gate_model.append(GCNConv(input_channel,num_experts))
                input_channel = num_experts
        elif gate_type =='GAT':
            self.gate_model=nn.ModuleList()
            input_channel=input_size
            for _ in range(2):
                self.gate_model.append(GATConv(input_channel, num_experts, heads,edge_dim=input_size ,dropout=0.6))
                input_channel = num_experts *heads
            self.gate_model.append(MLP([input_channel,input_channel,num_experts],norm=None, dropout=0.5))
           
        self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert(self.k <= self.num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        """
        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x,batch,edge_index,edge_attr, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        """
        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)
        if self.gate_type == 'liner':
            clean_logits = x @ self.w_gate
            clean_logits = global_mean_pool(clean_logits,batch)
        elif self.gate_type == 'GAT':
            input_x=x
            for conv in self.gate_model[:-1]:
                input_x = conv(input_x,edge_index,edge_attr)
            input_x=global_mean_pool(input_x,batch)
            input_x = self.gate_model[-1](input_x)
            clean_logits = input_x
        elif self.gate_type == 'GCN':
            input_x=x
            for conv in self.gate_model:
                input_x = conv(input_x,edge_index).relu()
            clean_logits = input_x
            clean_logits=global_mean_pool(clean_logits,batch)
        elif self.gate_type == 'GIN':
            input_x=x
            for conv in self.gate_model:
                input_x = conv(input_x,edge_index,edge_attr).relu()
            clean_logits = input_x
            clean_logits=global_mean_pool(clean_logits,batch)
        
        x=global_mean_pool(x,batch)
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices,top_k_gates )

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates,load

    def forward(self,batch_data):  
        gates,load = self.noisy_top_k_gating(batch_data.x,batch_data.batch,batch_data.edge_index,batch_data.edge_attr, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
          input_x = batch_data.x
          output=self.experts[i](batch_data)
          expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=1)
        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.sum(dim=1)
        return y,loss
    
    

