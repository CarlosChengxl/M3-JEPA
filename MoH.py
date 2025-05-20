import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.multihead_attention import init_multihead_attn

class MoE(nn.Module):
    def __init__(self, num_experts, input_size, output_size, hidden_size=256, k=2, num_heads=32, drop=0.):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.heads = num_heads
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(hidden_size, output_size),
                nn.Dropout(drop)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(input_size, num_experts)
        self.atten = init_multihead_attn(hidden_size, n_heads=self.heads, dropout=0.1)

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

    def forward(self, x):
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

class ExpertHead(nn.Module):
    def __init__(self, dim):
        super(ExpertHead, self).__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.atten_norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, dim)
        self. _init_w()
    
    def _init_w(self):
        nn.init.kaiming_uniform_(self.query.weight, nonlinearity='relu')
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        nn.init.kaiming_uniform_(self.key.weight, nonlinearity='relu')
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        nn.init.kaiming_uniform_(self.value.weight, nonlinearity='relu')
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        nn.init.kaiming_uniform_(self.out.weight, nonlinearity='relu')
        if self.out.bias is not None:
            nn.init.zeros_(self.out.bias)
        

    def forward(self, x):
        # x: (B, seq, dim)
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        output = self.atten_norm(output)
        output = F.gelu(output)
        output = self.out(output)
        return output   

class MoE_predictor(nn.Module):
    def __init__(self, num_experts, hidden_size=256, k=2, num_head=32, num_layers=2, drop=0., forward_l2=True, forward_cl=True, class_num=None):
        super(MoE_predictor, self).__init__()
        self.num_experts = num_experts
        self.k = k
        self.num_layer = num_layers
        self.cls_proj_in = nn.Linear(hidden_size, hidden_size, bias=True)

        self.cls_proj_in_1 = nn.Linear(hidden_size, hidden_size, bias=True)

        self.cls_fusion = nn.Linear(2*hidden_size, hidden_size, bias=True)
        ## layer norm

        self.cls_ln = nn.LayerNorm(hidden_size)

        self.cls_ln_out = nn.LayerNorm(hidden_size)

        self.cls_ln_1 = nn.LayerNorm(hidden_size)

        self.cls_ln_out_1 = nn.LayerNorm(hidden_size)

        self.cls_ln_out2 = nn.LayerNorm(2*hidden_size)
        self.after_atten = nn.LayerNorm(2*hidden_size)
        self.cls_ln_out3 = nn.LayerNorm(hidden_size)

        if num_layers == 1:
            self.experts = nn.ModuleList([ExpertHead(dim=hidden_size) for _ in range (num_experts)])
        else:
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Dropout(drop)
                ) for _ in range(num_experts)
            ])

        self.gate = nn.Linear(hidden_size, num_experts)
        # self.atten = init_multihead_attn(hidden_size, n_heads=num_head, num_layers=num_layers, dropout=0.1)
        if num_layers==2:
            self.atten = nn.ModuleList([nn.MultiheadAttention(50, 5, dropout=drop) for _ in range(4)])
            self.post_atten = nn.ModuleList([
                nn.Sequential(
                    nn.GELU(),
                    nn.Dropout(drop),
                    nn.Linear(50, 50),
                    nn.GELU(),
                    nn.Dropout(drop)
                ) for _ in range(4)
            ])
        

        ## cls proj
        # self.cls_prj = nn.Linear(hidden_size, text_dim, bias=True)

        # to do....控制任务的超参数，比如只进行cl或者l2 待实现
        forward_l2 = forward_l2
        forward_cl = forward_cl

        self._init_w()
    
    def _init_w(self):
        if self.num_layer != 1:
            for expert in self.experts:
                for layer in expert:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

        nn.init.kaiming_uniform_(self.gate.weight, nonlinearity='relu')
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

        nn.init.kaiming_uniform_(self.cls_proj_in.weight, nonlinearity='relu')
        if self.cls_proj_in.bias is not None:
            nn.init.zeros_(self.cls_proj_in.bias)



        nn.init.kaiming_uniform_(self.cls_proj_in_1.weight, nonlinearity='relu')
        if self.cls_proj_in_1.bias is not None:
            nn.init.zeros_(self.cls_proj_in_1.bias)

        nn.init.kaiming_uniform_(self.cls_fusion.weight, nonlinearity='relu')
        if self.cls_fusion.bias is not None:
            nn.init.zeros_(self.cls_fusion.bias)
        # nn.init.kaiming_uniform_(self.cls_prj.weight, nonlinearity='relu')
        # if self.cls_prj.bias is not None:
        #     nn.init.zeros_(self.cls_prj.bias)

    def forward(self, cls_x, cls_x_1, forward_l2=False, forward_cl=False):

        # dim proj
        cls_x, cls_x_1 = self.apply_task_specific_embedding(cls_x, cls_x_1)

        # # forward l2
        # reg_result, reg_result_1 = self.forward_l2(reg_x, reg_x_1)

        # forward cls 
        cls_result, cls_result_1 = self.forward_cls(cls_x, cls_x_1)

        # get cls embed and reg embed
        result = self.apply_task_specific_output(cls_result, cls_result_1)

        return result, cls_result, cls_result_1

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

    def apply_task_specific_embedding(self, cls_x, cls_x_1):
        # cls_x = self.cls_proj_in(cls_x)
        # cls_x_1 = self.cls_proj_in_1(cls_x_1)

        cls_x = self.cls_ln(cls_x)
        cls_x = F.gelu(cls_x)

        cls_x_1 = self.cls_ln_1(cls_x_1)
        cls_x_1 = F.gelu(cls_x_1)

        return cls_x, cls_x_1

    def apply_task_specific_output(self, cls_result, cls_result_1):
        #x_1 = (cls_result + cls_result_1) / 2
        # cls_result = torch.mean(cls_result, dim=1,keepdim=True)
        # cls_result_1 = torch.mean(cls_result_1, dim=1, keepdim=True)

        
        #在编码维度拼接
        con_x_1 = torch.cat((cls_result, cls_result_1), dim=-1)
        con_x_1 = self.cls_ln_out2(con_x_1)
        con_x_1 = F.gelu(con_x_1)
        con_x_1 = torch.transpose(con_x_1, 1, 2)
        # 使用自注意力
        # for atten in self.atten:
        #     con_x_1 = atten(con_x_1, con_x_1, con_x_1) + con_x_1
        if self.num_layer == 2:
            for atten, post_atten in zip(self.atten, self.post_atten):
                con_x_1_attn, _ = atten(con_x_1, con_x_1, con_x_1)
                con_x_1_attn = post_atten(con_x_1_attn)
                con_x_1 = con_x_1 + con_x_1_attn
            con_x_1 = torch.transpose(con_x_1, 1, 2)
            con_x_1 = self.after_atten(con_x_1)
            con_x_1 = F.gelu(con_x_1)


        con_x_1 = self.cls_fusion(con_x_1)
        con_x_1 = self.cls_ln_out3(con_x_1)
        con_x_1 = F.gelu(con_x_1)
        con_x_1 = torch.mean(con_x_1, dim=1, keepdim=True)

        # x_2 = (reg_result + reg_result_1) / 2
        # # con_x_2 = torch.cat((reg_result, reg_result_1), dim=2)
        # x_2 = self.cls_ln_out2(x_2)
        # x_2 = F.gelu(x_2)
        return con_x_1
    
    def forward_l2(self, reg_x, reg_x_1):
        # forward l2 Dino
        reg_result = self.forward_moe(reg_x)
        reg_result = self.reg_ln_out(reg_result)
        reg_result = F.gelu(reg_result)

        # forward l2 clip_vit
        reg_result_1 = self.forward_moe(reg_x_1)
        reg_result_1 = self.reg_ln_out_1(reg_result_1)
        reg_result_1 = F.gelu(reg_result_1)
        
        return reg_result, reg_result_1
    
    def forward_cls(self, cls_x, cls_x_1):
        # forward cls loss
        cls_result = self.forward_moe(cls_x)
        cls_result = self.cls_ln_out(cls_result)
        cls_result = F.gelu(cls_result)

        # forward cls loss clip_vit
        cls_result_1 = self.forward_moe(cls_x_1)
        cls_result_1 = self.cls_ln_out_1(cls_result_1)
        cls_result_1 = F.gelu(cls_result_1)

        return cls_result, cls_result_1
        
def init_moe(num_experts, input_size, output_size, hidden_size, k, drop=0.):
    moe = MoE(num_experts, input_size, output_size, hidden_size, k)
    return moe

def create_moe_predictor(num_experts, hidden_size, k, num_head, num_layers, drop=0., class_num=None):
    moe = MoE_predictor(num_experts, hidden_size=hidden_size, k=k, num_head=num_head, num_layers=num_layers, drop=drop, class_num=class_num)
    return moe
