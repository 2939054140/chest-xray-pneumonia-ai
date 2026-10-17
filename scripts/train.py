import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------------- 配置参数 --------------------------
data_root = r"D:\med_img_project\chest_xray"
batch_size = 8
num_epochs = 10
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("使用设备：", device)

# -------------------------- 数据预处理 --------------------------
# 训练集：基础预处理 + 数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # 随机水平翻转（医学影像合理增强）
    transforms.RandomRotation(15),              # 小角度旋转
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet均值标准差
])

# 验证/测试集：只做基础预处理，不做增强
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
print("验证集数量：", len(val_dataset))

# -------------------------- 构建模型 --------------------------
# 加载ResNet18预训练模型
model = models.resnet18(pretrained=True)

# 冻结主干网络参数，只训练最后的分类头
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层全连接层为二分类
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# 损失函数和优化器（只优化分类头）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# -------------------------- 训练循环 --------------------------
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"\n===== 第 {epoch+1}/{num_epochs} 轮 =====")
    
    # 训练阶段
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
    
    # 验证阶段
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
    
    # 保存最好的模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("保存当前最优模型")

print(f"\n训练完成，最优验证准确率：{best_acc:.2f}%")