from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from rag import build_prompt
from client import send_chat_completion, get_model_list, stream_chat_completion
import configparser
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/server.log"),
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
        user_message = data["messages"][-1]["content"]
        model = data["model"]
        temperature = float(data.get("temperature", 0.7))
        stream = data.get("stream", False)

        user_prompt = build_prompt(user_message)
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if stream:
            async def event_generator():
                async for chunk in stream_chat_completion(messages, model, temperature):
                    yield chunk
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")
        else:
            return send_chat_completion(messages, model, temperature)
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return {"error": str(e)}


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": get_model_list()
    }
