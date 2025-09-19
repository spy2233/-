from CNN import CNN
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import cv2

# ---------------- GPU 设置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 数据加载
train_data = datasets.MNIST(
    root="mnist", train=True, download=True, transform=transforms.ToTensor()
)
test_data = datasets.MNIST(
    root="mnist", train=False, download=True, transform=transforms.ToTensor()
)

train_loader = data_utils.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(test_data, batch_size=64, shuffle=True)

# 定义模型并移动到 GPU
cnn = CNN().to(device)  # GPU 版本

# --------- 训练部分 ---------
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

for epoch in range(1):
    for index, (images, labels) in enumerate(train_loader):
        # 将数据移动到 GPU
        images, labels = images.to(device), labels.to(device)

        outputs = cnn(images)
        loss = loss_func(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("当前轮次：{}，批次：{}/{}，loss={:.4f}".format(
            epoch+1, index+1, len(train_loader), loss.item()
        ))

# --------- 测试部分 ---------
loss_test = 0
rightValue = 0
cnn.eval()  # 设置评估模式

with torch.no_grad():
    for index2, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        loss_test += loss_func(outputs, labels).item()
        _, pred = outputs.max(1)
        rightValue += (pred == labels).sum().item()

        current_accuracy = rightValue / ((index2+1)*test_loader.batch_size)
        print("测试轮次：{}，批次：{}/{}，累计loss={:.4f}，当前准确率={:.4f}".format(
            epoch+1, index2+1, len(test_loader), loss_test, current_accuracy
        ))

total_accuracy = rightValue / len(test_data)
print("测试集平均loss={:.4f}, 平均准确率={:.4f}".format(
    loss_test/len(test_loader), total_accuracy
))
