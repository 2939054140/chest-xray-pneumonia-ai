import os
from openai import OpenAI

# 先确认能不能读到密钥
print("API Key:", os.getenv("DEEPSEEK_API_KEY")[:10] + "...")

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好，说一句话就行"}],
    temperature=0.1
)

print("返回结果：", response.choices[0].message.content)