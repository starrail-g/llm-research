import os
import torch
current_dir = os.path.dirname(os.path.abspath(__file__))

# 数据与训练相关配置
data_path = os.path.join(current_dir, "data", "input.txt")
output_dir = os.path.join(current_dir, "checkpoints")
batch_size = 64
block_size = 256
n_embed = 384  # 嵌入向量的维度
# 模型超参数
vocab_size = None  # 根据预处理结果自动设置
lr = 1e-3
weight_decay = 3e-4
num_iters = 100000
print_every = 1000
save_every = 5000
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_heads = 6  # 注意力头数
n_layer = 4  # Transformer 层数
dropout = 0.2  # Dropout 概率