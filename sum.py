import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from CNN import CNN  # 你的 CNN 网络定义

# ---------------- GPU 设置 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备:", device)

# ---------------- 加载模型 ----------------
model_path = r"E:\python\PythonProject10\model\mnist_model.pth"
model = CNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"模型已加载: {model_path}")

# ---------------- 图像预处理 ----------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),   # 转灰度
    transforms.Resize((28, 28)),                   # 调整为 28x28
    transforms.ToTensor(),                         # 转 tensor
    transforms.Normalize((0.1307,), (0.3081,))     # MNIST 归一化
])

def predict_image(model, img_path):
    # 读取图片
    img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise FileNotFoundError(f"未找到图片: {img_path}")
    img_display = img_cv.copy()

    # 转换为 PIL，再做 transform
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    img_tensor = transform(img_pil).unsqueeze(0).to(device)  # [1,1,28,28]

    # 模型推理
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = outputs.max(1)
        predicted_label = pred.item()

    # 在图上标注预测结果
    cv2.putText(img_display, f"预测: {predicted_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
    cv2.imshow("预测结果", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return predicted_label

# ---------------- 使用示例 ----------------
img_path = r"""C:\Users\22337\Desktop\picture.jpg"""
result = predict_image(model, img_path)
print("预测结果:", result)
