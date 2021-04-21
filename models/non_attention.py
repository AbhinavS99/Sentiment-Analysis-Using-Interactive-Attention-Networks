import torch
import torch.nn as nn
import torch.nn.functional as F


class NonAttentionModel(nn.Module):
    def __init__(self, embed_size, vocab_size, embedding_store, hidden_size, l2_reg_fact, max_aspect_len, max_context_len, output_dims):
        super(NonAttentionModel, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.embedding_store = embedding_store
        self.embedding_store = torch.from_numpy(self.embedding_store)
        self.hidden_size = hidden_size
        self.l2_reg_fact = l2_reg_fact
        self.max_aspect_len = max_aspect_len
        self.max_context_len = max_context_len
        self.output_dims = output_dims

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.embedding.weight.data.copy_(self.embedding_store)

        self.dropout = nn.Sequential(
            nn.Dropout(p = 0.01)
        )

        self.context_lstm = nn.Sequential(
            nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, batch_first = True)
        )
        
        self.aspect_lstm = nn.Sequential(
            nn.LSTM(input_size = self.embed_size, hidden_size = self.hidden_size, batch_first = True)
        )
        
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

        aspect_avg_pool = aspect_op.sum(dim = 1, keepdim = False)/aspect_mask.sum(dim = 1, keepdim = True)
        context_avg_pool = context_op.sum(dim = 1, keepdim = False)/context_mask.sum(dim = 1, keepdim = True)

        
        feats = torch.cat([aspect_avg_pool, context_avg_pool], dim = 1)
        feats = self.dropout(feats)
        op = self.fc(feats)
        return F.tanh(op)
        