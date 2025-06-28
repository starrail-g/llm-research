import torch
import torch.nn as nn
from config import block_size, n_embed, vocab_size,device, n_heads,n_layer,dropout
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding = nn.Embedding(block_size, n_embed)  # 可学习的位置嵌入
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_heads) for _ in range(n_layer)],
            nn.LayerNorm(n_embed),  # 最后的 LayerNorm
        )
        # nn.Sequential 用于将多个模块按顺序组合在一起
        # 这里创建了 n_layer 个 Block，每个 Block 包含自注意力和前馈网络
        # 最后添加一个 LayerNorm 层，用于对输出进行归一化处理
        self.lm_head = nn.Linear(n_embed, vocab_size)
        self.ffwn = FeedForward(n_embed)  # 前馈网络
        self.sa_head = Muti_HeadAttention(n_heads, n_embed//n_heads)  # self-attention head

    def forward(self, idx, targets=None):
        B , T = idx.shape  # B: batch size, T: sequence length
        # idx: (B, T)
        token_emb = self.token_embedding_table(idx)  # (B, T, vocab_size)
        pos_emb = self.positional_embedding(torch.arange(T,device = device))  # (T, n_embed)
        x = token_emb + pos_emb.unsqueeze(0)  # (B, T, n_embed)
        # 将位置嵌入添加到 token 嵌入中
        # pos_emb.unsqueeze(0) 将位置嵌入的形状从 (T, n_embed) 转换为 (1, T, n_embed)
        # 这样可以与 token_emb 相加，得到每个 token 的最终嵌入表示
        x = self.sa_head(x)  # (B, T, n_embed)
        # self-attention 处理
        # x 的形状是 (B, T, n_embed)，表示每个 token 的嵌入表示经过 self-attention 后的结果
        x = self.ffwn(x)  # (B, T, n_embed)
        # 前馈网络处理
        # x 的形状仍然是 (B, T, n_embed)，表示每个 token 的嵌入表示经过前馈网络后的结果
        x = x.reshape(B, T, n_embed)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        return logits, loss

class Head(nn.Module):
    """one head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))  # lower triangular matrix
        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting

    def forward(self, x):
        B,T,C = x.shape  # B: batch size, T: sequence length, C: embedding dimension
        k = self.key (x)
        q = self.query(x)
        v = self.value(x)
        # k, q, v 的形状都是 (B, T, head_size)
        weight = q@ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # apply causal mask
        weight = F.softmax(weight, dim=-1)  # (B, T, T)
        weight = self.dropout(weight)  # apply dropout
        out = weight @ v  # (B, T, head_size)

        return out


class Muti_HeadAttention(nn.Module):
    """multi-head self-attention"""
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj =  nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)  # dropout layer to prevent overfitting
        
    def forward(self, x):
        out =  torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, n_heads * head_size)
        out = self.dropout(self.proj(out))  # (B, T, n_embed)
        return out  # (B, T, n_embed)


class FeedForward(nn.Module):
    """simple feedforward network"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)  # dropout layer to prevent overfitting
        )
    def forward(self, x):
        return self.net(x)  # (B, T, n_embed)

class Block(nn.Module):
    """Transformer block : commnication followed by computation"""
    def __init__(self, n_embed, n_heads):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = Muti_HeadAttention(n_heads, head_size)  # self-attention
        self.ffn = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)  # layer normalization for self-attention
        self.ln2 = nn.LayerNorm(n_embed)  # layer normalization for feedforward network
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x  # residual connection
        x = self.ffn(self.ln2(x)) + x  # residual connection
        return x  # (B, T, n_embed)
