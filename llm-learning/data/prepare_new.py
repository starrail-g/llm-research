import os
import requests
import tiktoken
import numpy as np

#download the dataset
input_file_path = os.path.join(os.path.dirname(__file__),"input.txt")
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w', encoding='utf-8') as f:
        response = requests.get(data_url)
        f.write(response.text)
#load the dataset
with open(input_file_path,'r', encoding='utf-8') as f:
    text = f.read()
n = len(text)
train_text = text[:int(n*0.9)]
val_text = text[int(n*0.9):]
charset = sorted(list(set(text)))
vocab_size = len(charset)
#print(' '.join(charset))
#print(f"dataset has {n} characters, {vocab_size} unique.")
#print (text[:1000])
#encode the dataset
stoi = {ch: i for i, ch in enumerate(charset)}
itos = {i:ch for i,ch in enumerate(charset)}
def encode(s):
    return [stoi[c] for c in s]
def decode(l):
    return ''.join([itos[i] for i in l])

#print(encode("hello world"))
#print(decode(encode("hello world")))
#use tiktoken to encode the dataset
#encoding = tiktoken.get_encoding("gpt2")
#def encode_tiktoken(s):
#    return encoding.encode(s)
#def decode_tiktoken(l):    
#    return encoding.decode(l)


import torch
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)
#print(f"train_data has {len(train_data)} tokens, val_data has {len(val_data)} tokens.")
#print(f"train_data: {train_data[:100]}")

torch.manual_seed(1337)
batch_size = 32 # How many independent sequences will we process in parallel?
block_size = 8# What is the maximum context length for predictions?
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size , (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


#for b in range(batch_size):
#    for t in range(block_size):
 #       context = xb[b, :t+1]
 #       target = yb[b,t]
 #       print(f"when input is {decode(context.tolist())}, the target is {decode([target.item()])}")
import torch.nn as nn
from torch.nn import functional as F
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #构造一个大小为 vocab_size 的嵌入矩阵
        #每个 token 都有一个对应的向量表示
        #这里的 token_embedding_table 是一个 nn.Embedding 层
        #它的输入是一个 token 的索引，输出是该 token 的向量表示
        #vocab_size 是词汇表的大小，即 token 的数量
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B, T, C)
        # idx 是一个形状为 (B, T) 的张量，表示批次中的每个样本的 token 索引
        # logits 的形状为 (B, T, C)，其中 C 是词汇表的大小
        # logits[i, j] 是第 i 个样本在第 j 个位置的 token 的向量表示
        # logits[i, j, k] 是第 i 个样本在第 j 个位置的 token 的第 k 个维度的值，即对应于词汇表中第 k 个 token 的打分
        B,T,C = logits.shape
        logits = logits.view(B*T, C) # (B*T, C)
        #展平，为了一次性计算所有的loss
        #此时logits的每行表示一个样本的对下一个token的所有打分，共有B*T行，每行有C个打分
        if targets is None:
            loss = None
        else:
            targets = targets.view(B*T) # (B*T)
            loss = F.cross_entropy(logits, targets) # (B*T, C) , (B*T)
        #返回的 loss 是所有 B*T 个预测位置的平均损失。
        return logits, loss
    

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx) # (B, T, C), (B*T)
            logits = logits[:, -1, :] # (B, C)
            #取出最后一个时间步的 logits
            probs = F.softmax(logits, dim=-1) # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            #从最后一个时间步的 logits 中采样下一个 token
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            #将 idx 扩展到下一个时间步
        return idx

def save_checkpoint(model, optimizer, iteration, path='checkpoint.pth'):
    torch.save({
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint at iter {iteration} to {path}")

# —— 2) 加载 Checkpoint —— #
def load_checkpoint(model, optimizer, path='checkpoint.pth', map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])
    start_iter = checkpoint['iter'] + 1
    print(f"Loaded checkpoint from {path}, resuming at iter {start_iter}")
    return start_iter

#测试模型
m = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
m.to(device)
# —— 主训练流程 —— #
start_iter = 0
checkpoint_path = 'checkpoint.pth'

# 如果磁盘上已有 checkpoint，先加载它
try:
    start_iter = load_checkpoint(m, optimizer, checkpoint_path, map_location=device)
except FileNotFoundError:
    print("No checkpoint found, starting from scratch.")

num_iters = 100000
print_every = 1000
save_every = 5000

for iter in range(start_iter, num_iters):
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % print_every == 0:
        print(f"iter {iter}, loss: {loss.item():.4f}")

    if iter % save_every == 0 and iter > 0:
        save_checkpoint(m, optimizer, iter, checkpoint_path)


print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=1000)[0].tolist()))
# %%
