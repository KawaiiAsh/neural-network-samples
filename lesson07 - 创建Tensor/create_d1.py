import torch

# 使用arange创建等差序列的1维张量
tensor1d_arange = torch.arange(start=0, end=5, step=1)
print("1维张量 (使用arange):", tensor1d_arange)

# 使用linspace创建在指定区间内均匀分布的1维张量
tensor1d_linspace = torch.linspace(start=0, end=1, steps=5)
print("1维张量 (使用linspace):", tensor1d_linspace)
