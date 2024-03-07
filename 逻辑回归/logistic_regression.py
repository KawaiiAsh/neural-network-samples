import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成更多的随机数据
torch.manual_seed(42)
np.random.seed(42)
X = torch.randn(100, 2) * 2
X[:50] += 2
Y = torch.cat([torch.zeros(50), torch.ones(50)]).unsqueeze(1)

# 定义逻辑回归模型
class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入维度为2，输出维度为1

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

# 初始化模型
model = LogisticRegression()

# 定义损失函数为均方误差（MSE）
criterion = nn.MSELoss()

# 定义优化器
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
losses = []  # 用于存储每个epoch的损失值
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(X)
    # 计算损失
    loss = criterion(outputs, Y)
    # 反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())  # 记录当前epoch的损失值

    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 绘制损失曲线
plt.plot(losses)  # 使用plt.plot()绘制损失曲线
plt.xlabel('Epochs')  # 添加x轴标签
plt.ylabel('Loss')  # 添加y轴标签
plt.title('Training Loss')  # 添加标题
plt.show()  # 显示损失曲线图

# 测试模型
with torch.no_grad():
    predicted = model(X)
    predicted = predicted > 0.5
    accuracy = (predicted == Y.byte()).float().mean()
    print(f'Accuracy: {accuracy.item()*100:.2f}%')

    # 可视化预测结果
    plt.scatter(X[:, 0], X[:, 1], c=predicted.squeeze().numpy(), cmap='coolwarm')

    # 绘制决策边界
    w, b = model.linear.weight.data.numpy()[0], model.linear.bias.data.numpy()[0]
    x_boundary = np.array([X[:, 0].min(), X[:, 0].max()])
    y_boundary = (-1 / w[1]) * (w[0] * x_boundary + b)
    plt.plot(x_boundary, y_boundary, color='black')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Predicted Classes')
    plt.show()
