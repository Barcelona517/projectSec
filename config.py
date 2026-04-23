from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

WORKSPACE_ROOT = Path(os.getenv("AGENT_WORKSPACE_ROOT", BASE_DIR)).resolve()
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")
MAX_TURNS = int(os.getenv("MAX_TURNS", "8"))
HISTORY_FILE = Path(os.getenv("HISTORY_FILE", BASE_DIR / "chat_history.json"))
SYSTEM_PROMPT = """你是一个可调用本地工具的 Python 智能体助手。\n""" \
    "当用户需求需要读取或修改本地文件时，优先使用工具。" \
    "请在必要时使用工具，不要编造工具执行结果。" \
    "当任务已完成时，直接给出 final answer。"
