import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils

from CNN import CNN

# ---------------- GPU 设置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# 加载测试数据
test_data = datasets.MNIST(root="mnist", train=False, download=True, transform=transforms.ToTensor())
test_loader = data_utils.DataLoader(test_data, batch_size=64, shuffle=False)

# 加载模型并移动到 GPU
cnn = torch.load("model/mnist_model.pkl", map_location=device)
cnn.to(device)
cnn.eval()  # 设置评估模式

# 定义损失函数
loss_func = torch.nn.CrossEntropyLoss()

# 测试
loss_test = 0
rightValue = 0

for index, (images, labels) in enumerate(test_loader):
    # 将数据移动到 GPU
    images, labels = images.to(device), labels.to(device)

    outputs = cnn(images)
    loss_test += loss_func(outputs, labels).item()
    _, pred = outputs.max(1)
    rightValue += (pred == labels).sum().item()

    # 将数据移动到 CPU 处理显示
    images_cpu = images.cpu().numpy()
    labels_cpu = labels.cpu().numpy()
    preds_cpu = pred.cpu().numpy()

    for idx in range(images_cpu.shape[0]):
        im_data = images_cpu[idx].reshape(28, 28) * 255  # MNIST 单通道
        im_data = im_data.astype('uint8')
        im_label = labels_cpu[idx]
        im_pred = preds_cpu[idx]

        print(f"预测值: {im_pred}, 真实值: {im_label}")
        cv2.imshow("MNIST Test", im_data)
        cv2.waitKey(0)

# 输出测试结果
total_accuracy = rightValue / len(test_data)
print("测试集平均loss={:.4f}, 平均准确率={:.4f}".format(loss_test / len(test_loader), total_accuracy))
print("OpenCV版本:", cv2.__version__)
