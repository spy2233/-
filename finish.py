import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data_utils
import cv2
from PIL import Image
import os
from CNN import CNN   # 你的 CNN 定义文件 CNN.py

# ---------------- GPU 设置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ---------------- 数据加载 ----------------
train_data = datasets.MNIST(root="mnist", train=True, download=True, transform=transforms.ToTensor())
test_data = datasets.MNIST(root="mnist", train=False, download=True, transform=transforms.ToTensor())

train_loader = data_utils.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = data_utils.DataLoader(test_data, batch_size=64, shuffle=False)

# ---------------- 定义模型 ----------------
cnn = CNN().to(device)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

# ---------------- 训练 ----------------
epochs = 1   # 先训练1轮，你可以改大一些（比如 5 或 10）
for epoch in range(epochs):
    cnn.train()
    for index, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (index+1) % 200 == 0:
            print(f"训练轮次 {epoch+1}/{epochs}, 批次 {index+1}/{len(train_loader)}, loss={loss.item():.4f}")

# ---------------- 测试 ----------------
cnn.eval()
right, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = cnn(images)
        _, pred = outputs.max(1)
        right += (pred == labels).sum().item()
        total += labels.size(0)

accuracy = right / total
print(f"测试集准确率: {accuracy:.4f}")

# ---------------- 保存模型 ----------------
model_path = r"E:\python\PythonProject10\model\mnist_model.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)
torch.save(cnn.state_dict(), model_path)
print(f"模型已保存到: {model_path}")

# ---------------- 图像预测 ----------------
def predict_image(model, img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_display = img_cv.copy()

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = outputs.max(1)
        predicted_label = pred.item()

    # 显示预测结果
    cv2.putText(img_display, f"预测: {predicted_label}", (10,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
    cv2.imshow("预测结果", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return predicted_label

# 使用训练好的模型来预测你的图片
img_path = r"""C:\Users\22337\Desktop\24eb0ee74c844af6aca82a67341b6cf2.jpg"""
cnn.eval()
result = predict_image(cnn, img_path)
print("这张图片的预测结果是:", result)
