import torch
from torch import autograd

# 创建张量
x = torch.tensor(1.)  # 输入张量 x
a = torch.tensor(1., requires_grad=True)  # 参数 a，并设置其需要计算梯度
b = torch.tensor(2., requires_grad=True)  # 参数 b，并设置其需要计算梯度
c = torch.tensor(3., requires_grad=True)  # 参数 c，并设置其需要计算梯度

# 定义计算图
y = a ** 2 * x + b * x + c  # 计算结果 y

# 打印计算梯度前的梯度值
print('before:', a.grad, b.grad, c.grad)

# 计算梯度
grads = autograd.grad(y, [a, b, c])  # 计算 y 对参数 a、b、c 的梯度

# 打印计算梯度后的梯度值
print('after:', a.grad[0], b.grad[0], c.grad[0])
