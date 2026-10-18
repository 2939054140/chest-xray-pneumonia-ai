import os
import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

sys.path.append(os.path.dirname(__file__))

def evaluate_model(test_dir, model_path):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ========== 模型只加载一次 ==========
    print("正在加载模型...")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("模型加载完成，开始评估...")

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 类别映射
    class_map = {
        "NORMAL": 0,   # 正常对应标签0
        "PNEUMONIA": 1 # 肺炎对应标签1
    }
    label_map = {0: "正常", 1: "肺炎"}
    
    tp = tn = fp = fn = 0
    total = 0
    error_count = 0

    print("=" * 60)
    print("开始批量评估测试集...")
    print("=" * 60)

    for label_name, true_label in class_map.items():
        class_dir = os.path.join(test_dir, label_name)
        if not os.path.isdir(class_dir):
            continue
        
        img_list = [f for f in os.listdir(class_dir) 
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\n正在评估 {label_name} 类别，共 {len(img_list)} 张图片...")
        
        for idx, img_name in enumerate(img_list):
            img_path = os.path.join(class_dir, img_name)
            try:
                # 读取图片 + 预处理
                img = Image.open(img_path).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # 纯推理，不生成热力图
                with torch.no_grad():
                    output = model(img_tensor)
                    _, pred_idx = torch.max(output, 1)
                    pred_label = pred_idx.item()
                
                # 统计混淆矩阵
                if true_label == 1 and pred_label == 1:
                    tp += 1
                elif true_label == 0 and pred_label == 0:
                    tn += 1
                elif true_label == 0 and pred_label == 1:
                    fp += 1
                elif true_label == 1 and pred_label == 0:
                    fn += 1
                
                total += 1
                
                if (idx + 1) % 50 == 0:
                    print(f"  已完成 {idx+1}/{len(img_list)}")
                    
            except Exception as e:
                error_count += 1
                print(f"  跳过损坏图片：{img_name}")

    # 计算指标
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 输出结果
    print("\n" + "=" * 60)
    print("📊 模型性能评估结果")
    print("=" * 60)
    print(f"测试集总样本数：{total}")
    print(f"图片读取失败数：{error_count}")
    print(f"真阳性(TP)：{tp}  假阳性(FP)：{fp}")
    print(f"假阴性(FN)：{fn}  真阴性(TN)：{tn}")
    print("-" * 60)
    print(f"整体准确率 Accuracy：{accuracy*100:.2f}%")
    print(f"肺炎精确率 Precision：{precision*100:.2f}%")
    print(f"肺炎召回率 Recall：{recall*100:.2f}%")
    print(f"肺炎F1值 F1-Score：{f1*100:.2f}%")
    print("=" * 60)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total": total
    }


if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    test_dir = os.path.join(base_dir, "chest_xray/test")
    model_path = os.path.join(base_dir, "models/best_model.pth")
    
    evaluate_model(test_dir, model_path)