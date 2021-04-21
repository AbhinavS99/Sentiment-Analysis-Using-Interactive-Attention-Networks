from models.addns import AttentionLayer

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, embed_size, vocab_size, embedding_store, hidden_size, l2_reg_fact, max_aspect_len, max_context_len, output_dims):
        super(AttentionModel, self).__init__()

        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding_store = embedding_store
        self.embedding_store = torch.from_numpy(self.embedding_store)
        self.hidden_size = hidden_size
        self.l2_reg_fact = l2_reg_fact
        self.max_aspect_len = max_aspect_len
        self.max_context_len = max_context_len
        self.output_dims = output_dims
        
        self.store_attn = False


        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.embedding.weight.data.copy_(self.embedding_store)

        self.dropout = nn.Sequential(
            nn.Dropout(p = 0.01)
        )

        self.context_lstm = nn.Sequential(
            nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, batch_first = True)
        )

        self.context_attn = AttentionLayer(self.hidden_size)
        
        self.aspect_lstm = nn.Sequential(
            nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, batch_first = True)
        )

        self.aspect_attn =  AttentionLayer(self.hidden_size)


        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*2, self.output_dims)
        )

    def forward(self, data):
        aspect, context, aspect_mask, context_mask = data

        aspect = self.embedding(aspect)
        context = self.embedding(context)

        aspect = self.dropout(aspect)
        context = self.dropout(context)

        aspect_lstm_op, _ = self.aspect_lstm(aspect)
        context_lstm_op, _ = self.context_lstm(context)

        aspect_mask_addn_dim = aspect_mask.unsqueeze(-1)
        context_mask_addn_dim = context_mask.unsqueeze(-1)
        
        aspect_op = aspect_lstm_op*aspect_mask_addn_dim
        context_op = context_lstm_op*context_mask_addn_dim

        aspect_batch_size = aspect_op.size(0)
        aspect_time_step = aspect_op.size(1)

        context_batch_size = context_op.size(0)
        context_time_step = context_op.size(1)

        aspect_avg_pool = aspect_op.sum(dim = 1, keepdim = False)/aspect_mask.sum(dim = 1, keepdim = True)
        context_avg_pool = context_op.sum(dim = 1, keepdim = False)/context_mask.sum(dim = 1, keepdim = True)
        
        aspect_attn = self.aspect_attn(aspect_batch_size, aspect_time_step, aspect_op, context_avg_pool, aspect_mask)
        context_attn = self.context_attn(context_batch_size, context_time_step, context_op, aspect_avg_pool, context_mask)
        
        if self.store_attn:
            self.stored_aspect_attn = aspect_attn
            self.stored_context_attn = context_attn
        
        aspect_attn = aspect_attn.unsqueeze(1)
        context_attn = context_attn.unsqueeze(1)
        
        aspect_feats = aspect_attn.matmul(aspect_op)
        context_feats = context_attn.matmul(context_op)

        aspect_feats = aspect_feats.squeeze()
        context_feats = context_feats.squeeze()

        feats = torch.cat([aspect_feats, context_feats], dim = 1)
        feats = self.dropout(feats)
        op = self.fc(feats)
        return F.tanh(op)
    
    def set_attn_store(self, val):
        self.store_attn = val


