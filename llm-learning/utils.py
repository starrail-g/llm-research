import torch
import os
from config import output_dir,block_size


def save_checkpoint(model, optimizer, iteration, path=None):
    os.makedirs(output_dir, exist_ok=True)
    path = path or os.path.join(output_dir, f'ckpt_{iteration}.pth')
    torch.save({
        'iter': iteration,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict(),
    }, path)
    print(f"Saved checkpoint at iter {iteration} to {path}")


def load_checkpoint(path, model, optimizer=None, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
    return checkpoint['iter'] + 1


def generate_text(model, idx , max_len=100, temperature=1.0):
    
    for _ in range(max_len-1):
        idx_cond = idx[:,-block_size:]  # 取最后 block_size 个字符作为条件
        logits, _ = model(idx_cond)  # 获取模型输出
        logits = logits[:, -1, :]  # 只取最后一个时间步的输出
        probs = torch.softmax(logits / temperature, dim=-1)  # 应用 softmax 和温度缩放
        idx_next = torch.multinomial(probs, num_samples=1)  # 从概率分布中采样下一个字符
        idx = torch.cat((idx, idx_next), dim=1)
    return idx  # 返回生成的字符序列