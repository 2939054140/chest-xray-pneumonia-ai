import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据集路径，和你本地的对应上就行
data_root = r"D:\med_img_project\chest_xray"

# 基础预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 加载训练集
train_dataset = datasets.ImageFolder(os.path.join(data_root, "train"), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 打印验证信息
print("训练集图片总数：", len(train_dataset))
print("类别名称：", train_dataset.classes)
print("类别对应编号：", train_dataset.class_to_idx)

# 取一批数据测试
images, labels = next(iter(train_loader))
print("单批图片形状：", images.shape)
print("对应标签：", labels)