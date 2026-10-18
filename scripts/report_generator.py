import os
from openai import OpenAI

def generate_report(image_result, knowledge_texts):
    """
    真实调用DeepSeek生成结构化诊疗报告
    输入：影像诊断结果字典、知识库相关片段列表
    输出：大模型实时生成的诊疗报告字符串
    """
    # 初始化真实API客户端，无任何模拟、桩代码
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        timeout=30
    )

    # 提取参数，Python层面做置信度判断
    diagnosis = image_result.get("diagnosis", "未知")
    confidence = float(image_result.get("confidence", 0))
    
    # 低于0.6自动生成警告文本，拼入提示词开头
    warn_text = ""
    if confidence < 60:
        warn_text = "【⚠️ 置信度不足，建议人工复核】\n\n"

    # 拼接指南原文
    knowledge_str = ""
    for idx, text in enumerate(knowledge_texts):
        knowledge_str += f"依据片段{idx+1}：{text}\n"
    if not knowledge_texts:
        knowledge_str = "无相关指南片段"

    # 系统约束规则
    system_prompt = """你是诊疗报告整理员，严格遵守以下所有规则：
1. 所有诊疗建议必须完全基于用户提供的诊疗指南依据片段，绝对禁止编造任何医学内容。
2. 如果用户提示开头有【⚠️ 置信度不足，建议人工复核】，必须完整保留在报告最开头。
3. 输出必须固定为四段结构，标题严格不变：
【诊断结论】
【置信度】
【指南依据】
【诊疗建议】
4. 语言简洁专业，符合临床报告风格，不额外引申，不添加输入以外的任何信息。"""

    # 最终用户提示词
    user_prompt = f"""{warn_text}影像诊断结果：
诊断：{diagnosis}
置信度：{confidence:.1f}%

诊疗指南依据片段：
{knowledge_str}

请严格按照规则生成结构化诊疗报告。"""

    # ========== 调用API前强制打印完整提示词，用于验证参数 ==========
    print("="*60)
    print("【最终传入模型的系统提示词】")
    print(system_prompt)
    print("\n【最终传入模型的用户提示词】")
    print(user_prompt)
    print("="*60)
    # =============================================================

    # 真实调用DeepSeek接口，无任何硬编码返回
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # 示例调用，可自行修改测试数据验证参数透传
    test_image = {"diagnosis": "肺炎", "confidence": 0.95}
    test_knowledge = [
        "社区获得性肺炎的诊断标准包括：新近出现的咳嗽、咳痰，发热，肺部啰音，白细胞异常，胸部影像学显示斑片状浸润影。",
        "抗感染治疗是肺炎治疗的核心，青壮年无基础疾病者首选青霉素类或第一代头孢菌素。"
    ]
    print(generate_report(test_image, test_knowledge))