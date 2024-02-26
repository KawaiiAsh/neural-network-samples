import torch

# 创建一个0维张量
scalar = torch.tensor(42)
print("0维张量（标量）:", scalar)
print("Scalar value:", scalar.item())  # 使用.item()方法从中提取Python数值
print("Shape of scalar:", scalar.shape)  # 打印形状，应为空因为它是0维的
