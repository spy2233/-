import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data_utils

# ---------------- GPU 设置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ---------------- 模型定义 ----------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=5, padding=2), # 14x14 -> 14x14
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14 -> 7x7
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(7*7*64, 10)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ---------------- 数据加载 ----------------
transform = transforms.ToTensor()
train_data = torchvision.datasets.MNIST(root='mnist', train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST(root='mnist', train=False, download=True, transform=transform)

train_loader = data_utils.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(test_data, batch_size=64, shuffle=False)

# ---------------- 模型初始化 ----------------
cnn = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

# ---------------- 训练 ----------------
num_epochs = 5
for epoch in range(num_epochs):
    cnn.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], 平均loss: {total_loss/len(train_loader):.4f}")

# ---------------- 测试 ----------------
cnn.eval()
correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        test_loss += loss_func(outputs, labels).item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"测试集平均loss: {test_loss/len(test_loader):.4f}, 准确率: {correct/total:.4f}")

