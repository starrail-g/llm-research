import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 指定缓存目录为当前目录下的 'cache' 文件夹
cache_directory = "./cache"

# 检查缓存目录是否存在，如果不存在就创建
if not os.path.exists(cache_directory):
    os.makedirs(cache_directory)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2", cache_dir=cache_directory)
model = AutoModelForCausalLM.from_pretrained("distilgpt2", cache_dir=cache_directory).to(device)

prompt = "Hello"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 使用 max_new_tokens 生成 60 个新 token
output_ids = model.generate(
    **inputs,
    max_new_tokens=60,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

result = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(result)
