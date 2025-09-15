import torch
import transformers
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
# login()


from datasets import load_dataset, load_from_disk

# 下载 GSM8K 数据集
dataset = load_dataset("xiaodongguaAIGC/X-R1-7500")

# 保存数据集到本地
dataset.save_to_disk("./datasets/MATH")
dataset = load_from_disk("./datasets/MATH")
print(dataset)


# dataset.push_to_hub("LIMO10240")