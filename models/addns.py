import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.scale_fact = 0.2
        self.trans_fact = -0.1
        self.hidden_size = hidden_size
        self.weights = nn.Parameter(torch.rand(hidden_size, hidden_size)*self.scale_fact + self.trans_fact)

    def forward(self, batch_size, time_step, op, avg_pool, mask):
        avg_pool = avg_pool.unsqueeze(-1)
        op = op.unsqueeze(-2)
        weights = self.weights.repeat(batch_size, 1, 1)

        mid_val = weights.matmul(avg_pool)
        mid_val = mid_val.repeat(time_step, 1, 1, 1)
        mid_val = mid_val.transpose(0, 1)
        
        scores = op.matmul(mid_val).squeeze() 
        scores = scores + self.bias
        scores = torch.tanh(scores).squeeze()
        

        max_score = scores.max(dim=1, keepdim=True)[0]
        scores = scores - max_score

        scores = torch.exp(scores)*mask
        sum_scores = scores.sum(dim = 1, keepdim = True)
        return scores/sum_scores