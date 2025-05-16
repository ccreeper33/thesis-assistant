from fastapi import FastAPI, Request
from rag import build_prompt
from client import send_chat_completion, get_model_list
import configparser
import logging
import uvicorn

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/main.log"),
        logging.StreamHandler()
    ]
)

config = configparser.ConfigParser()
config.read('config.ini')

sys_prompt_path = config.get('settings', 'sys_prompt_path')
temperature = float(config.get('settings', 'temperature'))
host = config.get("api", "host")
port = int(config.get("api", "port"))

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
        user_model = data.get("model")
        if not user_model:
            return {"error": "必须提供 'model' 参数"}

        user_msg = data["messages"][-1]["content"]
        user_prompt = build_prompt(user_msg)

        prompt_msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = send_chat_completion(prompt_msgs, model=user_model, temperature=temperature)
        return response
    except Exception as e:
        logging.error(f"处理请求出错: {e}")
        return {"error": str(e)}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": get_model_list()
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host=host, port=port, reload=True)
