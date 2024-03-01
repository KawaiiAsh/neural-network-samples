import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 生成随机数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 + 2 * X + np.random.randn(100, 1)

# 转换为 PyTorch 的 Tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 定义线性回归模型
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 初始化模型
input_size = 1
output_size = 1
model = LinearRegression(input_size, output_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 获取训练后的参数
with torch.no_grad():
    predicted = model(X_tensor).detach().numpy()

# 画出数据点和拟合直线
plt.scatter(X, y)
plt.plot(X, predicted, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with PyTorch')
plt.show()