# model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(0, 1)  # -> (seq_len, batch, d_model)
        attn_output, _ = self.self_attn(x, x, x)
        x2 = self.norm1(x + self.dropout(attn_output))
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x2))))
        x3 = self.norm2(x2 + self.dropout(ff_output))
        return x3.transpose(0, 1)  # -> (batch, seq_len, d_model)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads)
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, x, memory):
        # x: (batch, seq_len, d_model)
        # memory: (batch, mem_len, d_model)
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        memory = memory.transpose(0, 1)  # (mem_len, batch, d_model)

        self_attn_output, _ = self.self_attn(x, x, x)
        x2 = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output, _ = self.multihead_attn(x2, memory, memory)
        x3 = self.norm2(x2 + self.dropout(cross_attn_output))

        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x3))))
        x4 = self.norm3(x3 + self.dropout(ff_output))

        return x4.transpose(0, 1)  # (batch, seq_len, d_model)

class InformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, n_heads, d_ff, n_layers, pred_len, n_classes):
        super().__init__()
        self.pred_len = pred_len

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.decoder_input = nn.Parameter(torch.zeros(1, pred_len, d_model))
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x)
        memory = x

        dec_in = self.decoder_input.repeat(x.size(0), 1, 1)  # (batch, pred_len, d_model)
        for layer in self.decoder_layers:
            dec_in = layer(dec_in, memory)

        out = dec_in.mean(dim=1)  # (batch, d_model)
        return self.classifier(out)  # (batch, n_classes)
