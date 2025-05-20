import torch
import torch.nn as nn
import torch.nn.functional as F


class MoE_predictor(nn.Module):
    def __init__(self, num_experts, text_dim, speech_dim, hidden_size=256, k=2, drop=0., forward_l2=True, forward_cl=True):
        super(MoE_predictor, self).__init__()
        self.num_experts = num_experts
        self.k = k

        ## layer norm
        self.l2_ln = nn.LayerNorm(hidden_size)
        self.cl_ln = nn.LayerNorm(hidden_size)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(hidden_size, hidden_size),
                nn.Dropout(drop)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(hidden_size, num_experts)
        
        self.txt_predictor_embed = nn.Linear(text_dim, hidden_size, bias=True)
        self.voe_predictor_embed = nn.Linear(speech_dim, hidden_size, bias=True)

        self.predictor_voe2txt = nn.Linear(hidden_size, text_dim, bias=True)
        self.predictor_txt2voe = nn.Linear(hidden_size, speech_dim, bias=True)

        ## contrastive learning
        self.CL_voe2txt = nn.Linear(hidden_size, hidden_size, bias=True)
        self.CL_txt2voe = nn.Linear(hidden_size, hidden_size, bias=True)
        
        ## task embedding
        self.cl_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.l2_embedding = nn.Parameter(torch.empty(1, 1, hidden_size))

        # to do....控制任务的超参数，比如只进行cl或者l2 待实现
        forward_l2 = forward_l2
        forward_cl = forward_cl

        self._init_w()
    
    def _init_w(self):
        for expert in self.experts:
            for layer in expert:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.gate.weight, nonlinearity='relu')
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

        nn.init.kaiming_uniform_(self.txt_predictor_embed.weight, nonlinearity='relu')
        if self.txt_predictor_embed.bias is not None:
            nn.init.zeros_(self.txt_predictor_embed.bias)
        nn.init.kaiming_uniform_(self.voe_predictor_embed.weight, nonlinearity='relu')
        if self.voe_predictor_embed.bias is not None:
            nn.init.zeros_(self.voe_predictor_embed.bias)


        nn.init.kaiming_uniform_(self.predictor_txt2voe.weight, nonlinearity='relu')
        if self.predictor_txt2voe.bias is not None:
            nn.init.zeros_(self.predictor_txt2voe.bias)
        nn.init.kaiming_uniform_(self.predictor_voe2txt.weight, nonlinearity='relu')
        if self.predictor_voe2txt.bias is not None:
            nn.init.zeros_(self.predictor_voe2txt.bias)

        nn.init.kaiming_uniform_(self.CL_txt2voe.weight, nonlinearity='relu')
        if self.CL_txt2voe.bias is not None:
            nn.init.zeros_(self.CL_txt2voe.bias)
        nn.init.kaiming_uniform_(self.CL_voe2txt.weight, nonlinearity='relu')
        if self.CL_voe2txt.bias is not None:
            nn.init.zeros_(self.CL_voe2txt.bias)    

        nn.init.kaiming_uniform_(self.cl_embedding, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.l2_embedding, nonlinearity='relu')

    def forward(self, x, task, forward_l2=False, forward_cl=False):
        x = F.gelu(self.apply_task_specific_embedding(x, task))

        l2_m_t_embedding = self.l2_embedding.expand(x.shape[0], x.shape[1], -1)
        cl_m_t_embedding = self.cl_embedding.expand(x.shape[0], x.shape[1], -1)
        x_1 = x + l2_m_t_embedding
        x_2 = x + cl_m_t_embedding
        
        l2_result = self.forward_moe(x_1)
        l2_result = self.l2_ln(l2_result)
        l2_result = F.gelu(l2_result)
        l2_result = l2_result + x_1

        cl_result = self.forward_moe(x_2)
        cl_result = self.cl_ln(cl_result)
        cl_result = F.gelu(cl_result)
        cl_result = cl_result + x_2

        l2_result, cl_result = self.apply_task_specific_output(l2_result, cl_result, task)
        return l2_result, cl_result

    def forward_moe(self, x):

        gate_output = self.gate(x)
        
        topk_values, topk_indices = torch.topk(F.softmax(gate_output, dim=-1), self.k, dim=-1)
        # Shape of expert_outputs: (batch_size, seq_len, num_experts, output_size)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        
        # Reshape tensors to match dimensions
        batch_size, seq_len, _ = x.size()
        topk_indices = topk_indices.unsqueeze(-1).expand(batch_size, seq_len, self.k, expert_outputs.size(-1))
        selected_experts = torch.gather(expert_outputs, 2, topk_indices)
        
        # Shape of weighted_expert_output: (batch_size, seq_len, k, output_size)
        weighted_expert_output = selected_experts * topk_values.unsqueeze(-1).expand_as(selected_experts)
        
        # Sum over the k dimension to combine expert outputs
        result = weighted_expert_output.sum(dim=2)
        
        return result

    def apply_task_specific_embedding(self, x, task):
        if task == 'text2voice':
            return self.txt_predictor_embed(x)
        elif task == 'voice2text':
            return self.voe_predictor_embed(x)
        else:
            raise ValueError("Unknown task type")

    def apply_task_specific_output(self, x_1, x_2, task):
        if task == 'text2voice':
            return self.predictor_txt2voe(x_1), self.CL_txt2voe(x_2)
        elif task == 'voice2text':
            return self.predictor_voe2txt(x_1), self.CL_voe2txt(x_2)
        else:
            raise ValueError("Unknown task type")

def create_moe_predictor(num_experts, text_dim, speech_dim, hidden_size, k, drop=0.):
    moe = MoE_predictor(num_experts, text_dim, speech_dim, hidden_size=hidden_size, k=k, drop=drop)
    return moe
