import os
import json
import pandas as pd
import time
import requests
from tqdm import tqdm
import re

API_KEY = 'sk-25a10e97a9fbfed98a9c591bbc605792'
MODEL_NAME = "Baichuan3-Turbo"
BASE_URL = "https://api.baichuan-ai.com/v1/chat/completions"

INPUT_CSV = "/Users/taijieshengwu/A-QA-System-for-Public-Art-Design/data.csv"
OUTPUT_DIR = "/Users/taijieshengwu/A-QA-System-for-Public-Art-Design/RAG_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

def clean_text(text):
    # 去除所有换行符、制表符、回车等
    text = re.sub(r'[\n\r\t]', '', text)
    # 去除所有空格（包括全角空格）
    text = re.sub(r'[\s\u3000]+', '', text)
    # 去除所有非中英文和数字字符（保留常见标点可自行添加）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9,.!?。，！？、：:（）()]', '', text)
    return text


QA_PAIRS_HUMAN_PROMPT_TEMPLATE = """  
请按以下格式整理学习成果：

<Context>
{text}
</Context>

请输出如下 JSON 格式（严格仿照结构，仅填写内容）：

[
  {{
    "作者": "无",
    "出处": "无",
    "年份": "无"
  }},
  {{
    "摘要": "无",
    "关键词": "无"
  }},
  {{
    "涉及艺术门类": "无",
    "涉及艺术家": "无",
    "涉及地区": "无"
  }},
  {{
    "核心观点/研究发现": "无",
    "理论模型/分析框架": "无",
    "研究方法": "无",
    "案例": "无",
    "学术意义": "无",
    "创新点": "无"
  }},
  {{
    "主要内容": "无"
  }}
]

我们开始吧！
"""

QA_PAIRS_SYSTEM_PROMPT = """  
你是一个面向公共艺术设计领域的文献分析助手。接下来我将提供一段文献内容，请你分析并输出结构化信息，整理出以下字段：

- “作者”，"出处"，"年份"，"摘要"，"关键词"
- "涉及艺术门类"，"涉及艺术家"，"涉及地区"
- "核心观点/研究发现"，"理论模型/分析框架"
- "研究方法"，"案例"，"学术意义"，"创新点"，"主要内容"

要求：
- 如果某项没有信息，请填写字符串“无”
- 请以 JSON 数组格式输出，不添加多余字段
- 所有字段内容必须为字符串
"""

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def call_baichuan_api(text):
    try:
        cleaned_text = clean_text(text)
    except: 
        cleaned_text = text
    user_prompt = QA_PAIRS_HUMAN_PROMPT_TEMPLATE.format(text=cleaned_text)

    messages = [
        {"role": "system", "content": QA_PAIRS_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.3,
        "stream": False
    }

    try:
        response = requests.post(BASE_URL, data=json.dumps(payload), headers=HEADERS, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


for i in tqdm([59,187,153]):#range(len(df))
    title = df.iloc[i]["title"]
    content = df.iloc[i]["content"]

    result = call_baichuan_api(content)

    if result:
        print(f"[✓] {title}")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        output_path = os.path.join(OUTPUT_DIR, f"{safe_title}.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result)
    else:
        print(f"[✗] Failed to process: {title}")
        time.sleep(2)
# import os

# # 文件路径
# file_path = '/Users/taijieshengwu/A-QA-System-for-Public-Art-Design/dataset.json'

# # 读取并处理每一行
# with open(file_path, 'r', encoding='utf-8') as f:
#     lines = f.readlines()

# # 去掉末尾空白字符
# lines = [line.strip() for line in lines]

# # 删除指定的两行
# lines = [line for line in lines if line not in [
#     '{"question": "问题1", "answer": "答案1"}',
#     '{"question": "问题2", "answer": "答案2"}'
# ]]

# # 给每行末尾加逗号（除了最后一行）
# lines = [line + ',' for line in lines[:-1]] + [lines[-1]]

# # 加上 JSON 列表结构
# lines.insert(0, '[')
# lines.append(']')

# # 保存为新文件（或者覆盖原文件）
# new_file_path = '/Users/taijieshengwu/A-QA-System-for-Public-Art-Design/dataset_cleaned.json'
# with open(new_file_path, 'w', encoding='utf-8') as f:
#     f.write('\n'.join(lines))

# print(f"处理完成，清洗后的文件保存在：{new_file_path}")
