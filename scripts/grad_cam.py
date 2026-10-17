import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------- 配置：改成你自己的图片路径 --------------------------
img_path = r"D:\med_img_project\chest_xray\test\PNEUMONIA\person1_virus_6.jpeg"
save_path = "grad_cam_result.jpg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 图像预处理 --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------- 加载优化后的模型 --------------------------
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("best_model_optimized.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------- Grad-CAM 核心逻辑 --------------------------
features = []

# 钩子：提取最后一层卷积的输出特征图
def forward_hook(module, input, output):
    features.append(output.data.cpu().numpy())

model.layer4.register_forward_hook(forward_hook)

# 加载并预处理图片
img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(device)

# 前向传播得到预测结果
output = model(img_tensor)
_, pred_class = torch.max(output, 1)
class_idx = pred_class.item()

# 取出特征图和对应类别的全连接权重
feat_map = features[0][0]  # 形状 [512, 7, 7]
fc_weights = model.fc.weight.data.cpu().numpy()[class_idx]

# 加权求和生成类激活热力图
cam = np.zeros(feat_map.shape[1:], dtype=np.float32)
for i, w in enumerate(fc_weights):
    cam += w * feat_map[i]

# 归一化 + 放大到原图尺寸
cam = np.maximum(cam, 0)  # 只保留正激活
cam = cv2.resize(cam, (224, 224))
cam = (cam - cam.min()) / (cam.max() - cam.min())

# -------------------------- 叠加原图并保存 --------------------------
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (224, 224))

# 生成红色热力图并与原图融合
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
result = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

cv2.imwrite(save_path, result)
print(f"✅ Grad-CAM结果已保存到：{save_path}")
print(f"模型预测结果：{'PNEUMONIA 肺炎' if class_idx == 1 else 'NORMAL 正常'}")