import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch_geometric.nn import MLP, GINConv,GINEConv,GATConv,GCNConv,GatedGraphConv,global_add_pool


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels,num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(in_channels,hidden_channels))
            in_channels = hidden_channels
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index,batch):
        for conv in self.convs:
          x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return x
    
class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels],num_layers=2)
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return x
    
class GatedGCN(torch.nn.Module):
    def __init__(self, in_channels,num_layers):
        super().__init__()
        self.conv = GatedGraphConv(in_channels, num_layers)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        x = global_add_pool(x, batch)
        return x

class MoE(nn.Module):
    def __init__(self, model_type,gate_type,input_size, output_size, num_experts,min_layers,device, noisy_gating=True, k=4, coef=1e-3,dropout=0.2,heads=4):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        self.device=device
        self.gate_dropout=dropout
        self.gate_type=gate_type
        # instantiate experts
        self.experts = torch.nn.ModuleList()
        
        # model
        if model_type=='GIN':
            for i in range(min_layers,num_experts+min_layers):
                input_channel = input_size
                self.experts.append(GIN(input_channel,output_size,i))
        elif model_type=='GCN':
            for i in range(min_layers,num_experts+min_layers):
                input_channel = input_size
                self.experts.append(GCN(input_channel,output_size,i))
        elif model_type=='GatedGCN':
            for i in range(min_layers,num_experts+min_layers):
                input_channel = input_size
                self.experts.append(GatedGCN(input_channel,i))

        #gate
        if gate_type == 'liner':
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        elif gate_type =='GIN':
            self.gate_model=nn.ModuleList()
            mlp = MLP([input_channel, output_size, output_size],num_layers=2)
            self.gate_model.append(GINConv(nn=mlp, train_eps=False))
            mlp = MLP([output_size, output_size, num_experts],num_layers=2)
            self.gate_model.append(GINConv(nn=mlp, train_eps=False))            
        elif gate_type =='GINE':
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
                self.gate_model.append(GATConv(input_channel, num_experts, heads, dropout=0.6))
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

    def noisy_top_k_gating(self, x,batch,edge_index, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        """
        if self.gate_type == 'liner':
            clean_logits = x @ self.w_gate
            clean_logits = global_add_pool(clean_logits,batch)
        elif self.gate_type == 'GAT':
            input_x=x
            for conv in self.gate_model[:-1]:
                input_x = conv(input_x,edge_index)
            input_x = self.gate_model[-1](input_x)
            input_x=global_add_pool(input_x,batch)
            clean_logits = input_x
        elif self.gate_type == 'GCN' or 'GIN':
            input_x=x
            for conv in self.gate_model:
                input_x = conv(input_x,edge_index).relu()
            clean_logits = input_x
            clean_logits=global_add_pool(clean_logits,batch)
        
        x=global_add_pool(x,batch)
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

    def forward(self,x,batch,edge_index):
        gates,load = self.noisy_top_k_gating(x,batch,edge_index, self.training)
        # calculate importance loss
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
          input_x = x
          output=self.experts[i](input_x, edge_index,batch)
          expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.sum(dim=1)
        return y,loss
    
    

