import torch

# 创建一个0维张量
# 0维张量，也称为标量，是一个单一的数值。它没有维度，可以看作是一个简单的数字。
scalar = torch.tensor(5)
print("0维张量（标量）:", scalar)
print("Shape of scalar:", scalar.shape)

# 创建一个1维张量
# 1维张量是一个数字序列，也可以看作是向量。它包含了一行或一列的数字。
vector = torch.tensor([1, 2, 3, 4])
print("\n1维张量（向量）:", vector)
print("Shape of vector:", vector.shape)

# 创建一个2维张量
# 2维张量可以被看作是数字的表格，通常用于存储矩阵。
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("\n2维张量（矩阵）:", matrix)
print("Shape of matrix:", matrix.shape)

# 创建一个3维张量
# 3维张量可以用于存储多个矩阵，常见于处理时间序列数据或图像集。
tensor3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print("\n3维张量:", tensor3d)
print("Shape of tensor3d:", tensor3d.shape)

# 创建一个4维张量
# 4维张量常用于处理图像数据（包含批次大小、颜色通道数、图像高度和宽度）或序列数据。
tensor4d = torch.tensor([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]])
print("\n4维张量:", tensor4d)
print("Shape of tensor4d:", tensor4d.shape)
