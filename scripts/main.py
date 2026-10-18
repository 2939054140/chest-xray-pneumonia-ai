import os
import sys
from image_diagnosis import diagnose_chest_xray
from knowledge_base import search_knowledge
from report_generator import generate_report

def main(image_path, model_path, guide_pdf_path):
    print("=" * 60)
    print("【1/3】正在加载模型并进行影像诊断...")
    print("=" * 60)

    image_result = diagnose_chest_xray(
        image_path=image_path,
        model_path=model_path,
        heatmap_save_dir="results/heatmaps"
    )
    
    print(f"\n✅ 影像诊断完成")
    print(f"诊断结果：{image_result['diagnosis']}")
    print(f"置信度：{image_result['confidence']:.4f}")
    print(f"热力图保存路径：{image_result['heatmap_path']}")

    print("\n" + "=" * 60)
    print("【2/3】正在检索诊疗指南依据...")
    print("=" * 60)

    # 正常诊断用标准健康建议，异常诊断才检索肺炎指南
    if image_result['diagnosis'] == "正常":
        knowledge_texts = [
            "胸部影像学未见明确渗出、实变、占位等异常征象，肺野清晰，肺纹理走行自然。",
            "无异常发现者建议年度健康体检复查胸部影像学；如有咳嗽、发热、胸痛等呼吸道症状及时就诊。"
        ]
        print("\n✅ 诊断为正常，使用标准健康评估建议")
    else:
        query = f"{image_result['diagnosis']} 诊断标准 治疗原则"
        knowledge_texts = search_knowledge(query, guide_pdf_path)
        print(f"\n✅ 指南检索完成，共找到 {len(knowledge_texts)} 条相关片段")
        for i, text in enumerate(knowledge_texts):
            print(f"片段{i+1}：{text[:80]}...")

    print("\n" + "=" * 60)
    print("【3/3】正在生成结构化诊疗报告...")
    print("=" * 60)

    final_report = generate_report(image_result, knowledge_texts)

    print("\n" + "=" * 60)
    print("📋 最终辅助诊疗报告")
    print("=" * 60)
    print(final_report)
    print("=" * 60)

    # 自动保存报告
    report_save_dir = "results/reports"
    os.makedirs(report_save_dir, exist_ok=True)
    
    image_filename = os.path.basename(image_path)
    report_filename = os.path.splitext(image_filename)[0] + "_report.txt"
    report_save_path = os.path.join(report_save_dir, report_filename)
    
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("胸部X光辅助诊疗报告\n")
        f.write("=" * 60 + "\n")
        f.write(f"影像文件：{image_filename}\n")
        f.write(f"诊断结果：{image_result['diagnosis']}\n")
        f.write(f"置信度：{image_result['confidence']:.2f}%\n")
        f.write(f"热力图路径：{image_result['heatmap_path']}\n")
        f.write("=" * 60 + "\n\n")
        f.write(final_report)
    
    print(f"\n💾 报告已自动保存至：{report_save_path}")


if __name__ == "__main__":
    # 固定配置，不用每次改
    trained_model = r"D:\med_img_project\models\best_model.pth"
    guide_pdf = r"D:\med_img_project\data\cap_guide_2018.pdf"

    # 支持命令行传图片路径
    if len(sys.argv) > 1:
        test_xray_image = sys.argv[1]
    else:
        # 默认测试图片
        test_xray_image = r"D:\med_img_project\chest_xray\test\PNEUMONIA\person21_virus_53.jpeg"
    
    main(test_xray_image, trained_model, guide_pdf)