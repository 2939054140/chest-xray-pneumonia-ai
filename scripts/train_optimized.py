import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

# -------------------------- 配置参数 --------------------------
data_root = r"D:\med_img_project\chest_xray"
batch_size = 8
num_epochs = 15
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# -------------------------- 数据预处理 --------------------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join(data_root, "val"), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print("训练集数量：", len(train_dataset))
print("各类样本数：", np.bincount(train_dataset.targets))

# -------------------------- 核心优化：计算类别权重 --------------------------
# 样本越少，权重越高，平衡类别对损失的影响
class_counts = np.bincount(train_dataset.targets)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * 2
class_weights = class_weights.to(device)
print("类别权重：", class_weights)

# -------------------------- 构建模型 --------------------------
model = models.resnet18(pretrained=True)

# 冻结前3层，只微调最后一层卷积 + 分类头
for name, param in model.named_parameters():
    if "layer4" not in name and "fc" not in name:
        param.requires_grad = False

# 替换最后一层分类头
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# 加权交叉熵损失 + 只优化可训练参数
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

# -------------------------- 训练循环 --------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n===== 第 {epoch+1}/{num_epochs} 轮 =====")
    
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"训练损失：{train_loss:.4f}  训练准确率：{train_acc:.2f}%")
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = 100 * correct / total
    print(f"验证准确率：{val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_optimized.pth")
        print("保存当前最优模型")

print(f"\n训练完成，最优验证准确率：{best_acc:.2f}%")