from fastapi import FastAPI, Request
from rag import build_prompt
import requests
import configparser
import logging
import uvicorn

# 配置日志，将日志保存到文件
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

# 读取配置文件
config = configparser.ConfigParser()
config.read('config.ini')

# 从配置文件中获取LLM API地址
llm_api_url = config.get('llm', 'api_url')

# 从配置文件中获取系统提示文件路径
sys_prompt_path = config.get('settings', 'sys_prompt_path')

try:
    with open(sys_prompt_path, 'r', encoding='utf-8') as f:
        sys_prompt = f.read()
except FileNotFoundError:
    logging.error(f"系统提示文件未找到，请检查 {sys_prompt_path} 是否存在。")
    sys_prompt = ""

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        data = await request.json()
        user_message = data["messages"][-1]["content"]
        logging.info(f"Received user message: {user_message}")

        user_prompt = build_prompt(user_message)
        logging.info(f"Built user prompt: {user_prompt}")

        payload = {
            "model": config.get('llm', 'llm_model'),
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": float(config.get('settings', 'temperature'))
        }

        response = requests.post(llm_api_url, json=payload)
        response.raise_for_status()
        logging.info("Successfully sent request to VLLM API")
        return response.json()
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run("main:app", host=config.get("api", "host"), port=int(config.get("api", "port")), reload=True)
