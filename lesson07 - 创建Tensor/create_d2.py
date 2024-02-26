import torch

# 创建一个2x3的全零2维张量
tensor2d_zeros = torch.zeros(2, 3)
print("\n2维张量 (全零):", tensor2d_zeros)

# 创建一个3x2的全一2维张量
tensor2d_ones = torch.ones(3, 2)
print("2维张量 (全一):", tensor2d_ones)

# 创建一个2x3的随机2维张量
tensor2d_rand = torch.rand(2, 3)
print("2维张量 (随机):", tensor2d_rand)
