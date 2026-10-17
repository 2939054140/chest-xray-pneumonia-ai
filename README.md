\# 胸部X光肺炎智能诊断系统



基于 ResNet18 迁移学习的胸部X光片肺炎二分类项目，包含模型训练、效果评估、Grad-CAM 可解释性可视化与 Streamlit 交互演示界面。



\## 项目简介

针对胸部X光肺炎筛查场景，采用迁移学习方案构建 ResNet18 二分类模型。通过加权交叉熵损失优化类别不平衡问题，同时实现 Grad-CAM 类激活热力图可视化，提供模型决策的可解释性，并封装为可本地运行的 Web 交互工具。



\## 技术栈

\- 深度学习框架：PyTorch、TorchVision

\- 图像处理：OpenCV、Pillow

\- 数据与评估：NumPy、scikit-learn

\- 可解释性可视化：Grad-CAM

\- Web 交互：Streamlit



\## 核心成果

1\.  在 624 张独立测试集上达到 \*\*AUC 0.9646\*\*，召回率 \*\*97.44%\*\*，整体准确率 \*\*89.74%\*\*

2\.  针对 1:3 的类别不平衡问题，采用加权交叉熵损失与分层微调策略优化，肺炎漏诊率降至 2.6%，适配医疗筛查“宁可误判不可漏诊”的需求

3\.  实现 Grad-CAM 类激活热力图可视化，直观展示模型决策关注的肺部病灶区域，验证模型学习有效性

4\.  基于 Streamlit 搭建 Web 交互界面，支持图片上传、实时预测、热力图同步展示



\## 目录结构

├── scripts/ # 核心算法脚本

│ ├── dataset\_test.py

│ ├── train.py

│ ├── train\_optimized.py

│ ├── test\_eval.py

│ └── grad\_cam.py

├── app/ # 交互界面

│ └── app.py

├── models/ # 模型权重文件

├── docs/ # 项目文档与复盘

├── results/ # 可视化结果图

└── README.md





\## 使用说明

\### 环境配置

```bash

conda create -n med\_img python=3.9 -y

conda activate med\_img

pip install torch torchvision opencv-python pillow numpy scikit-learn streamlit



\##数据验证

python scripts/dataset\_test.py



\##模型训练

\# 基线模型训练

python scripts/train.py

\# 优化版模型训练

python scripts/train\_optimized.py



\##测试评估

python scripts/test\_eval.py



\##启动交互界面

streamlit run app/app.py



数据集说明

使用 Kaggle 公开数据集 Chest X-Ray Images (Pneumonia)，共 5863 张胸部 X 光灰度图，分为正常（NORMAL）与肺炎（PNEUMONIA）两类，官方划分训练 / 验证 / 测试集。

注意

本项目仅用于医学影像 AI 研究与学习演示，不构成任何临床诊断建议。



