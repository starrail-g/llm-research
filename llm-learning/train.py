import torch
import os
import sys

from config import lr, weight_decay, num_iters, print_every, save_every,output_dir
from prepare import get_dataloaders
from model import BigramLanguageModel
from utils import save_checkpoint, load_checkpoint, generate_text





def generate_demo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, stoi, itos = get_dataloaders()
    m = BigramLanguageModel(len(stoi)).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)
    # 加载最新模型参数
    try:
        load_checkpoint(os.path.join(output_dir,"latest.pth"), m, optimizer, map_location=device)
    except FileNotFoundError:
        print("No checkpoint found, using untrained model.")
    
    print(''.join([itos[i] for i in(generate_text(m,torch.zeros((1, 1), dtype=torch.long, device=device),1000)[0].tolist())]))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, stoi, itos = get_dataloaders()

    m = BigramLanguageModel(len(stoi)).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=weight_decay)

    start_iter = 0
    # 尝试加载已有 checkpoint
    try:
        start_iter = load_checkpoint(os.path.join(output_dir,"latest.pth"), m, optimizer, map_location=device)
    except FileNotFoundError:
        print("Starting from scratch.")

    it = start_iter
    while it < num_iters:
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if it % print_every == 0:
                print(f"iter {it}, loss: {loss.item():.4f}")
            if it % save_every == 0 and it > 0:
                save_checkpoint(m, optimizer, it, path=os.path.join(output_dir,"latest.pth"))
            it += 1
            if it >= num_iters:
                break

    # 训练结束后生成示例
    print(''.join([itos[i] for i in(generate_text(m,torch.zeros((1, 1), dtype=torch.long, device=device),1000)[0].tolist())])) 


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'generate':
        generate_demo()
    else:
        main()
