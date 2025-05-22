import os
import json
import pandas as pd
import time
import requests
from tqdm import tqdm

API_KEY = 'sk-25a10e97a9fbfed98a9c591bbc605792'
MODEL_NAME = "Baichuan3-Turbo"
BASE_URL = "https://api.baichuan-ai.com/v1/chat/completions"

INPUT_CSV = "/Users/taijieshengwu/Documents/arts-RAG/data.csv"
OUTPUT_DIR = "/Users/taijieshengwu/Documents/arts-RAG/RAG_data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")

# 去除多余空行、空格
def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned_lines)


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


for i in tqdm(range(153,len(df))):
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
