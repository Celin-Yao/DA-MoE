import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from model_proteins.gnn import GCN_OGB,SAGE_OGB


class MoE_OGB(nn.Module):
    def __init__(self,gnn_type,gate_type,input_size,hidden_size,out_size,num_experts,min_layers, noisy_gating=True, k=4 , coef=1e-3,dropout=0.5,gate_dropout=0.2,heads=4):
        super(MoE_OGB, self).__init__()
        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.k = k
        self.loss_coef = coef
        # self.device=device
        self.gate_dropout=gate_dropout
        self.gate_type=gate_type
        # instantiate experts
        self.experts = torch.nn.ModuleList()

        # model   
        if gnn_type == 'GCN':     
            for i in range(min_layers,num_experts+min_layers):
                self.experts.append(GCN_OGB(input_size,hidden_size,out_size,i,dropout=dropout))      

        #gate
        if gate_type == 'liner':
            self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        elif gate_type =='GCN':
            self.gate_model=GCN_OGB(input_size,hidden_size,num_experts,2,dropout)

        elif gate_type =='SAGE':
            self.gate_model=SAGE_OGB(input_size,hidden_size,num_experts,2,dropout)

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

    def noisy_top_k_gating(self, x,adj_t, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
        """
        # clean_logits = x @ self.w_gate
        if self.gate_type == 'liner':
            clean_logits = x @ self.w_gate
        elif self.gate_type == 'GCN' or 'SAGE':
            clean_logits = self.gate_model(x,adj_t)
        
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

    def forward(self,x,adj_t):
        gates,load = self.noisy_top_k_gating(x,adj_t, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= self.loss_coef

        expert_outputs = []
        for i in range(self.num_experts):
          input_x = x
          output=self.experts[i](input_x,adj_t)
          expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=1)

        y = gates.unsqueeze(dim=-1) * expert_outputs
        y = y.sum(dim=1)
        return y,loss
    
    

