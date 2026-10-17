import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -------------------------- 全局配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# -------------------------- 加载模型（缓存加速） --------------------------
@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("best_model_optimized.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------- Grad-CAM 生成函数 --------------------------
def generate_gradcam(img_pil):
    features = []
    
    # 前向钩子：提取最后一层卷积特征
    def forward_hook(module, input, output):
        features.append(output.data.cpu().numpy())
    
    handle = model.layer4.register_forward_hook(forward_hook)
    
    # 预处理 + 前向传播
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    output = model(img_tensor)
    _, pred = torch.max(output, 1)
    class_idx = pred.item()
    
    handle.remove()
    
    # 加权求和生成热力图
    feat_map = features[0][0]
    fc_weights = model.fc.weight.data.cpu().numpy()[class_idx]
    
    cam = np.zeros(feat_map.shape[1:], dtype=np.float32)
    for i, w in enumerate(fc_weights):
        cam += w * feat_map[i]
    
    # 后处理：只保留正激活 + 归一化 + 放大
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    
    # 与原图叠加
    img_np = np.array(img_pil.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
    
    return result, class_idx

# -------------------------- 页面UI布局 --------------------------
st.title("胸部X光肺炎智能诊断系统")
st.subheader("基于ResNet18迁移学习 + Grad-CAM可解释性可视化")

st.markdown("---")
uploaded_file = st.file_uploader("上传胸部X光片（支持JPG、PNG格式）", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col_original, col_heatmap = st.columns(2)
    
    img = Image.open(uploaded_file).convert("RGB")
    
    with col_original:
        st.image(img, caption="原始胸片", use_column_width=True)
    
    cam_img, pred_idx = generate_gradcam(img)
    
    with col_heatmap:
        st.image(cam_img, caption="Grad-CAM 激活热力图", use_column_width=True)
    
    st.markdown("---")
    label_text = "**肺炎**" if pred_idx == 1 else "**正常**"
    st.markdown(f"### 模型预测结果：{label_text}")
    
    if pred_idx == 1:
        st.warning("⚠️ 检测到肺炎特征，建议进一步临床检查确认")
    else:
        st.success("✅ 未检测到明显肺炎特征")

st.markdown("---")
st.caption("* 本系统仅用于医学影像AI研究演示，不构成任何临床诊断建议")