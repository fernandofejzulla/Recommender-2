import torch
import torch.nn as nn
import math

"""
SASRec Model:

1) Implement the SASRec architecture, including:
2) Item embeddings and positional embeddings
3) Self-attention blocks for modeling sequential dependencies
4) Causal attention masking so that each position only attends to previous items
5) Feedforward layers, dropout, and layer normalization

A final prediction layer that scores candidate items based on the sequence representation
"""

#Feedforward layers
class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super().__init__()

        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.gelu = nn.GELU()
        #dropout applied after activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out = self.gelu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return out
    
class SASRec(nn.Module):
    """
    Self-Attentive Sequential Recommendation Model (According to the paper "Self-Attentive Sequential Recommendation")
    """
    def __init__(self, item_num, maxlen, hidden_units, num_blocks, num_heads, dropout_rate):
        super().__init__()
        self.maxlen = maxlen
        self.hidden_units = hidden_units

        # Item embeddings and positional embeddings
        self.item_embedding = nn.Embedding(item_num + 1, hidden_units, padding_idx=0)

        self.position_embedding = nn.Embedding(maxlen, hidden_units)

        #dropout on the summed embeddings (item+position)
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # Self-attention blocks for modeling sequential dependencies
        #layer normalization
        self.attention_layernorms = nn.ModuleList() #layernorm before attention
        self.forward_layernorms = nn.ModuleList() #layernorm before FFN

        #self-attention blocks
        self.attention_layers = nn.ModuleList() #multi-head self-attention
        #feedforward layers
        self.forward_layers = nn.ModuleList() #point-wise FFN

        for _ in range(num_blocks):
            self.attention_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            self.attention_layers.append(nn.MultiheadAttention(embed_dim=hidden_units, num_heads=num_heads, dropout=dropout_rate, batch_first=True))

            self.forward_layernorms.append(nn.LayerNorm(hidden_units, eps=1e-8))

            self.forward_layers.append(PointWiseFeedForward(hidden_units, dropout_rate))

        # A final prediction layer that scores candidate items based on the sequence representation
        self.last_layernorm = nn.LayerNorm(hidden_units, eps=1e-8)

    # Causal attention masking so that each position only attends to previous items
    def create_attention_mask(self, input_seq):
        device = input_seq.device
        L = self.maxlen

        #casual mask
        casual = torch.triu(torch.full((L,L), -1e9, device=device), diagonal=1)

        #padding mask
        is_pad = (input_seq == 0)

        pad_mask = torch.where(is_pad.unsqueeze(1), torch.tensor(-1e9, device=device), torch.tensor(0.0, device=device))

        #combine both
        combined = casual.unsqueeze(0) + pad_mask

        # Repeat for each attention head so shape is (B * num_heads, L, L)
        num_heads = self.attention_layers[0].num_heads
        combined = combined.repeat(1, num_heads, 1).view(-1, L, L)


        return combined
    
    def forward(self, input_seq):
        B, L = input_seq.shape

        #embedding layer
        item_embeddings = self.item_embedding(input_seq) #look up item embeddings

        positions = torch.arange(L, device=input_seq.device).unsqueeze(0) #creates indices
        position_embeddings = self.position_embedding(positions) #look up position embeddings

        sequential_embeddings = item_embeddings + position_embeddings
        sequential_embeddings = self.embedding_dropout(sequential_embeddings) #dropout

        padding_mask = (input_seq == 0)
        sequential_embeddings = sequential_embeddings * (~padding_mask).unsqueeze(-1).float() #mask out padding positions

        #causal attention masking
        attention_mask = self.create_attention_mask(input_seq)

        #self-attention blocks
        hidden = sequential_embeddings

        for i in range(len(self.attention_layers)):
            normed = self.attention_layernorms[i](hidden)
            attention_output, _ = self.attention_layers[i](query=normed, key=normed, value=normed, attn_mask=attention_mask)
            hidden = hidden + attention_output
            
            normed = self.forward_layernorms[i](hidden)
            ffn_output = self.forward_layers[i](normed)
            hidden = hidden + ffn_output
            hidden = hidden * (~padding_mask).unsqueeze(-1).float() #mask out padding positions

        hidden = self.last_layernorm(hidden)

        return hidden
    
    def predict(self, input_seq, candidate_items):
        hidden = self.forward(input_seq)
        candidate_embeddings = self.item_embedding(candidate_items)
        scores = (hidden * candidate_embeddings).sum(dim=-1) #dot product between sequence representation and candidate item embeddings

        return scores