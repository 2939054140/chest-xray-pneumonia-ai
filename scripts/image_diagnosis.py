import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np
import os

def diagnose_chest_xray(image_path, model_path, heatmap_save_dir="results/heatmaps"):
    """
    胸部X光肺炎诊断函数
    输入：图片路径、模型权重路径、热力图保存目录
    输出：字典，包含 diagnosis(诊断结果)、confidence(置信度)、heatmap_path(热力图保存路径)
    """
        # 实现逻辑：
    # 1. 检查 heatmap_save_dir 目录，不存在则自动创建
    # 2. 构建 ResNet18 模型，修改最后一层全连接为 2 分类，加载 model_path 对应的权重，切换为 eval 模式
    # 3. 定义图像预处理流水线：Resize 到 224x224 → 转 Tensor → 用 ImageNet 均值标准差归一化
    # 4. 读取输入图片，预处理后增加 batch 维度，送入模型前向传播
    # 5. 计算 softmax 概率，取最大概率对应的类别，置信度转百分比数值
    # 6. 注册前向钩子捕获最后一层卷积输出，生成 Grad-CAM 热力图，叠加到原图上保存
    # 1. 检查 heatmap_save_dir 目录，不存在则自动创建
    if not os.path.exists(heatmap_save_dir):
        os.makedirs(heatmap_save_dir, exist_ok=True)

    # 2. 构建 ResNet18 模型，修改最后一层全连接为 2 分类，加载 model_path 对应的权重，切换为 eval 模式
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. 定义图像预处理流水线：Resize 到 224x224 → 转 Tensor → 用 ImageNet 均值标准差归一化
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 4. 读取输入图片，预处理后增加 batch 维度，送入模型前向传播
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    output = model(input_tensor)

    # 5. 计算 softmax 概率，取最大概率对应的类别，置信度转百分比数值
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)
    pred_idx = pred.item()
    confidence = conf.item() * 100.0
    diagnosis = "肺炎" if pred_idx == 1 else "正常"

    # 6. 注册前向钩子捕获最后一层卷积输出，生成 Grad-CAM 热力图，叠加到原图上保存
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)
    score = output[0, pred_idx]
    score.backward()

    fh.remove()
    bh.remove()

    feat = features[0].detach().cpu().numpy()[0]
    grad = gradients[0].detach().cpu().numpy()[0]
    weights = np.mean(grad, axis=(1, 2))
    cam = np.zeros(feat.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * feat[i]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    cam = cv2.resize(cam, (224, 224))

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    heatmap_path = os.path.join(heatmap_save_dir, f"{base_name}_gradcam.jpg")
    cv2.imwrite(heatmap_path, overlay)

    # 7. 返回字典：diagnosis 为 "正常"/"肺炎"，confidence 为 float 百分比，heatmap_path 为保存的完整路径
    return {
        "diagnosis": diagnosis,
        "confidence": float(confidence),
        "heatmap_path": os.path.abspath(heatmap_path),
    }
