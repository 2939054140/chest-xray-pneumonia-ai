import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import numpy as np

# -------------------------- 配置 --------------------------
data_root = r"D:\med_img_project\chest_xray"
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 数据加载 --------------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(data_root, "test"), transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("测试集图片总数：", len(test_dataset))
print("类别：", test_dataset.classes)

# -------------------------- 加载模型 --------------------------
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("best_model_optimized.pth", map_location=device))
model = model.to(device)
model.eval()

# -------------------------- 预测推理 --------------------------
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -------------------------- 计算指标 --------------------------
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print("\n========== 测试集评估结果 ==========")
print(f"准确率：{accuracy*100:.2f}%")
print(f"精确率：{precision*100:.2f}%")
print(f"召回率：{recall*100:.2f}%")
print(f"AUC：{auc:.4f}")
print("\n混淆矩阵：")
print("TN  FP")
print(cm[0])
print("FN  TP")
print(cm[1])