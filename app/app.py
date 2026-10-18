import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))

import streamlit as st
from PIL import Image
from image_diagnosis import diagnose_chest_xray
from knowledge_base import search_knowledge
from report_generator import generate_report

# 页面配置
st.set_page_config(page_title="胸部X光肺炎辅助诊疗系统", layout="wide")
st.title("🩺 胸部X光肺炎辅助诊疗系统")
st.caption("技术架构：ResNet18 影像诊断 + Grad-CAM 病灶可视化 + 诊疗指南向量检索 + 大模型结构化报告")

# ========== 侧边栏配置 ==========
with st.sidebar:
    st.header("⚙️ 系统配置")
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    model_path = st.text_input(
        "模型权重路径",
        value=os.path.join(base_dir, "models/best_model.pth")
    )
    guide_pdf_path = st.text_input(
        "诊疗指南PDF路径",
        value=os.path.join(base_dir, "data/cap_guide_2018.pdf")
    )
    heatmap_save_dir = st.text_input(
        "热力图保存目录",
        value=os.path.join(base_dir, "results/heatmaps")
    )
    st.divider()
    st.info("配置修改后，上传新图片即可生效")

# ========== 主界面上传区 ==========
uploaded_file = st.file_uploader(
    "上传胸部X光影像",
    type=["jpg", "jpeg", "png"],
    help="支持 JPG、PNG 格式的胸部正位片"
)

# ========== 上传后全流程处理 ==========
if uploaded_file is not None:
    # 项目内临时目录，全英文路径，避开OpenCV中文路径坑
    temp_dir = os.path.join(base_dir, "results/temp")
    os.makedirs(temp_dir, exist_ok=True)
    tmp_image_path = os.path.join(temp_dir, "upload_tmp.jpeg")

    # 写入本地临时文件
    with open(tmp_image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    try:
        # 第一步：影像诊断
        with st.spinner("正在加载模型、进行影像诊断..."):
            image_result = diagnose_chest_xray(
                image_path=tmp_image_path,
                model_path=model_path,
                heatmap_save_dir=heatmap_save_dir
            )

        # 诊断结果概览卡片
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("诊断结果", image_result['diagnosis'])
        with col2:
            conf = image_result['confidence']
            st.metric("模型置信度", f"{conf:.2f}%")
        with col3:
            if conf < 60:
                st.error("⚠️ 置信度不足，建议人工复核")
            else:
                st.success("✅ 置信度达标")

        # 影像对比展示
        st.subheader("影像对比")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.image(uploaded_file, caption="原始胸片", use_container_width=True)
        with img_col2:
            heatmap_img = Image.open(image_result['heatmap_path'])
            st.image(heatmap_img, caption="Grad-CAM 病灶热力图（高亮区为模型关注区域）", use_container_width=True)

        # 第二步+第三步：指南检索 + 报告生成
        with st.spinner("正在匹配诊疗指南、生成结构化报告..."):
            if image_result['diagnosis'] == "正常":
                knowledge_texts = [
                    "胸部影像学未见明确渗出、实变、占位等异常征象，肺野清晰，肺纹理走行自然。",
                    "无异常发现者建议年度健康体检复查胸部影像学；如有咳嗽、发热、胸痛等呼吸道症状及时就诊。"
                ]
            else:
                query = f"{image_result['diagnosis']} 诊断标准 治疗原则"
                knowledge_texts = search_knowledge(query, guide_pdf_path)
            
            final_report = generate_report(image_result, knowledge_texts)

        # 展示最终报告
        st.subheader("📋 结构化诊疗报告")
        st.markdown(final_report.replace("\n", "\n\n"))

        # 报告下载按钮
        st.download_button(
            label="⬇️ 下载完整诊疗报告",
            data=final_report,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_诊疗报告.txt",
            mime="text/plain"
        )

    finally:
        # 清理临时文件
        if os.path.exists(tmp_image_path):
            os.remove(tmp_image_path)

else:
    st.info("👆 请在上方上传胸部X光影像，系统将自动完成「影像诊断→指南匹配→报告生成」全流程")

st.markdown("---")
st.caption("* 本系统仅用于医学影像AI研究演示，不构成任何临床诊断建议")