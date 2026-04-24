from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime
from html import escape
import io
import json
import os
import re
import socket
from uuid import uuid4

import gradio as gr

from config import HISTORY_FILE, MODEL_NAME, WORKSPACE_ROOT
from main import load_history, run_agent, save_history


CONVERSATIONS_FILE = HISTORY_FILE.with_name("conversations.json")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_title_from_messages(messages: list[dict]) -> str:
    keyword_titles = [
        (["界面", "ui", "布局", "页面", "样式", "按钮"], "界面布局调整"),
        (["历史", "会话", "多轮", "上下文"], "历史会话与上下文"),
        (["读取", "read", "文件", "目录", "list"], "文件读取与浏览"),
        (["写入", "保存", "write", "创建"], "文件写入操作"),
        (["报错", "错误", "异常", "失败", "error"], "问题排查与修复"),
        (["端口", "启动", "运行", "web_ui", "gradio"], "服务启动配置"),
        (["模型", "deepseek", "api", "key"], "模型与密钥配置"),
    ]

    for item in messages:
        if item.get("role") == "user":
            content = str(item.get("content", "")).strip()
            if content:
                lowered = content.lower()
                for kws, title in keyword_titles:
                    if any(k in lowered for k in kws):
                        return title

                cleaned = re.sub(r"[\r\n]+", " ", content)
                cleaned = re.sub(r"^(请问|请|帮我|麻烦|你能|可以|我想|能不能|是否)\s*", "", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。！？,.!?：:;；")
                if not cleaned:
                    return "新对话"
                return cleaned[:14] + ("..." if len(cleaned) > 14 else "")
    return "新对话"


def _new_conversation(messages: list[dict] | None = None) -> dict:
    msgs = list(messages or [])
    return {
        "id": str(uuid4()),
        "title": _make_title_from_messages(msgs),
        "updated_at": _now_iso(),
        "messages": msgs,
    }


def _persist_conversations(conversations: list[dict], active_id: str) -> None:
    payload = {
        "active_id": active_id,
        "conversations": conversations,
    }
    CONVERSATIONS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    active = _find_conversation(conversations, active_id)
    save_history(HISTORY_FILE, active.get("messages", []))


def _find_conversation(conversations: list[dict], conv_id: str) -> dict:
    for conv in conversations:
        if conv.get("id") == conv_id:
            return conv
    return conversations[0]


def _load_or_init_conversations() -> tuple[list[dict], str]:
    if CONVERSATIONS_FILE.exists():
        try:
            data = json.loads(CONVERSATIONS_FILE.read_text(encoding="utf-8"))
            conversations = data.get("conversations", [])
            active_id = data.get("active_id", "")
            if isinstance(conversations, list) and conversations:
                convs = [c for c in conversations if isinstance(c, dict) and "id" in c and "messages" in c]
                if convs:
                    if not any(c.get("id") == active_id for c in convs):
                        active_id = convs[0].get("id", "")
                    return convs, active_id
        except json.JSONDecodeError:
            pass

    old_messages = load_history(HISTORY_FILE)
    first = _new_conversation(old_messages)
    conversations = [first]
    _persist_conversations(conversations, first["id"])
    return conversations, first["id"]


def _conversation_choices(conversations: list[dict]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for conv in sorted(conversations, key=lambda c: c.get("updated_at", ""), reverse=True):
        title = _make_title_from_messages(conv.get("messages", []))
        if title == "新对话":
            title = str(conv.get("title", "新对话"))
        updated = str(conv.get("updated_at", ""))
        stamp = updated.replace("T", " ")[:16]
        choices.append((f"{title}  ·  {stamp}", str(conv.get("id"))))
    return choices


def _format_assistant_content(thought: str, answer: str) -> str:
    safe_answer = escape(answer or "")
    thought = (thought or "").strip()
    if thought and thought != answer:
        safe_thought = escape(thought)
        return (
            f"<div class='ai-thought'>思考: {safe_thought}</div>"
            f"<div class='ai-answer'>回答: {safe_answer}</div>"
        )
    return f"<div class='ai-answer'>{safe_answer}</div>"


def _build_history_sidebar(agent_history: list[dict]) -> str:
    user_msgs = [m.get("content", "") for m in agent_history if m.get("role") == "user"]
    user_msgs = [str(x).strip() for x in user_msgs if str(x).strip()]
    if not user_msgs:
        return "暂无历史对话"

    lines = []
    for idx, text in enumerate(user_msgs[-20:], 1):
        short = text if len(text) <= 28 else f"{text[:28]}..."
        lines.append(f"{idx}. {short}")
    return "\n".join(lines)


def _history_to_chat_messages(agent_history: list[dict]) -> list[dict[str, str]]:
    chat_messages: list[dict[str, str]] = []
    for item in agent_history:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if role == "assistant":
            chat_messages.append({"role": role, "content": _format_assistant_content("", content)})
        else:
            chat_messages.append({"role": role, "content": content})
    return chat_messages


def _submit_message(
    user_message: str,
    chat_messages: list[dict[str, str]] | None,
    conversations: list[dict] | None,
    current_conv_id: str,
) -> tuple[list[dict[str, str]], list[dict], str, str, gr.update]:
    user_message = (user_message or "").strip()
    convs = list(conversations or [])
    if not convs:
        convs, current_conv_id = _load_or_init_conversations()

    current_conv = _find_conversation(convs, current_conv_id)
    agent_history = list(current_conv.get("messages", []))

    if not user_message:
        return (
            chat_messages or _history_to_chat_messages(agent_history),
            convs,
            current_conv_id,
            "",
            gr.update(choices=_conversation_choices(convs), value=current_conv_id),
        )

    ui_messages = list(chat_messages or [])
    current_agent_history = list(agent_history)

    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            answer, new_agent_history = run_agent(user_message, current_agent_history)
        logs = buf.getvalue().splitlines()
        thoughts = [line.split("Thought/Reply:", 1)[1].strip() for line in logs if "Thought/Reply:" in line]
        thought_text = "\n".join([t for t in thoughts if t])

        current_conv["messages"] = new_agent_history
        current_conv["title"] = _make_title_from_messages(new_agent_history)
        current_conv["updated_at"] = _now_iso()
        _persist_conversations(convs, current_conv_id)

        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append(
            {
                "role": "assistant",
                "content": _format_assistant_content(thought_text, answer),
            }
        )
        return (
            ui_messages,
            convs,
            current_conv_id,
            "",
            gr.update(choices=_conversation_choices(convs), value=current_conv_id),
        )
    except Exception as exc:  # noqa: BLE001
        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append(
            {
                "role": "assistant",
                "content": _format_assistant_content("", f"Agent Error: {exc}"),
            }
        )
        return (
            ui_messages,
            convs,
            current_conv_id,
            "",
            gr.update(choices=_conversation_choices(convs), value=current_conv_id),
        )


def _select_conversation(conv_id: str, conversations: list[dict]) -> tuple[list[dict[str, str]], str]:
    convs = list(conversations or [])
    if not convs:
        convs, conv_id = _load_or_init_conversations()
    selected = _find_conversation(convs, conv_id)
    save_history(HISTORY_FILE, selected.get("messages", []))
    return _history_to_chat_messages(selected.get("messages", [])), str(selected.get("id"))


def _new_chat(conversations: list[dict]) -> tuple[gr.update, list[dict], str, list[dict[str, str]], str]:
    convs = list(conversations or [])
    conv = _new_conversation([])
    convs.append(conv)
    active_id = conv["id"]
    _persist_conversations(convs, active_id)
    return (
        gr.update(choices=_conversation_choices(convs), value=active_id),
        convs,
        active_id,
        [],
        "",
    )


def _clear_current_chat(
    conversations: list[dict],
    current_conv_id: str,
) -> tuple[list[dict[str, str]], list[dict], str, gr.update, str]:
    convs = list(conversations or [])
    if not convs:
        convs, current_conv_id = _load_or_init_conversations()
    conv = _find_conversation(convs, current_conv_id)
    conv["messages"] = []
    conv["title"] = "新对话"
    conv["updated_at"] = _now_iso()
    _persist_conversations(convs, current_conv_id)
    return [], convs, current_conv_id, gr.update(choices=_conversation_choices(convs), value=current_conv_id), ""


def build_demo() -> gr.Blocks:
    conversations, active_id = _load_or_init_conversations()
    active = _find_conversation(conversations, active_id)
    initial_chat_messages = _history_to_chat_messages(active.get("messages", []))
    initial_choices = _conversation_choices(conversations)

    with gr.Blocks(title="Mini OpenClaw Chat") as demo:
        gr.HTML("""
        <div class="header-wrap">
          <div class="page-title">pre OpenClaw</div>
          <div class="page-meta">模型: """ + escape(MODEL_NAME) + """ | 受限目录: """ + escape(str(WORKSPACE_ROOT)) + """</div>
        </div>
        """)

        with gr.Row(elem_id="main-row"):
            with gr.Column(scale=3, min_width=260, elem_id="left-panel"):
                gr.Markdown("### 历史对话")
                history_list = gr.Radio(
                    choices=initial_choices,
                    value=active_id,
                    elem_id="history-list",
                    interactive=True,
                    show_label=False,
                )
                new_chat_btn = gr.Button("+ 新建对话", elem_id="new-chat-btn")

            with gr.Column(scale=9, elem_id="right-panel"):
                chatbot = gr.Chatbot(
                    value=initial_chat_messages,
                    buttons=["copy"],
                    layout="bubble",
                    sanitize_html=False,
                    elem_id="chat-window",
                )
                message_box = gr.Textbox(
                    label="输入消息",
                    placeholder="输入内容后回车或点发送",
                    lines=3,
                    elem_id="input-box",
                )

                with gr.Row(elem_id="actions-row"):
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空当前会话")

        conversations_state = gr.State(conversations)
        current_conv_id_state = gr.State(active_id)

        send_btn.click(
            fn=_submit_message,
            inputs=[message_box, chatbot, conversations_state, current_conv_id_state],
            outputs=[chatbot, conversations_state, current_conv_id_state, message_box, history_list],
        )
        message_box.submit(
            fn=_submit_message,
            inputs=[message_box, chatbot, conversations_state, current_conv_id_state],
            outputs=[chatbot, conversations_state, current_conv_id_state, message_box, history_list],
        )
        history_list.change(
            fn=_select_conversation,
            inputs=[history_list, conversations_state],
            outputs=[chatbot, current_conv_id_state],
        )
        new_chat_btn.click(
            fn=_new_chat,
            inputs=[conversations_state],
            outputs=[history_list, conversations_state, current_conv_id_state, chatbot, message_box],
        )
        clear_btn.click(
            fn=_clear_current_chat,
            inputs=[conversations_state, current_conv_id_state],
            outputs=[chatbot, conversations_state, current_conv_id_state, history_list, message_box],
        )

    return demo


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _pick_port(preferred: int, max_tries: int = 30) -> int:
    for port in range(preferred, preferred + max_tries):
        if _is_port_available(port):
            return port
    raise RuntimeError(f"无法找到可用端口，尝试范围: {preferred}-{preferred + max_tries - 1}")


def main() -> None:
    demo = build_demo()
    preferred_port = int(os.getenv("WEB_PORT", "7860"))
    server_port = _pick_port(preferred_port)
    if server_port != preferred_port:
        print(f"端口 {preferred_port} 已占用，自动切换到 {server_port}")
    demo.launch(server_name="127.0.0.1", server_port=server_port, inbrowser=True, css="""
    html, body, .gradio-container {
        height: 100vh !important;
        overflow: hidden !important;
    }
    .gradio-container {
        max-width: 100% !important;
        padding: 10px 12px 12px 12px !important;
    }
    .header-wrap {
        padding: 4px 4px 6px 4px;
    }
    .page-title {
        font-size: 22px;
        font-weight: 700;
        line-height: 1.1;
    }
    .page-meta {
        color: #666;
        margin-top: 2px;
        font-size: 13px;
    }
    #main-row {
        height: calc(100vh - 66px);
        gap: 10px;
    }
    #left-panel, #right-panel {
        height: 100%;
    }
    #left-panel {
        display: flex;
        flex-direction: column;
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 10px;
        overflow: hidden;
    }
    #history-list {
        flex: 1 1 auto;
        min-height: 0;
        overflow-y: auto;
        border: 1px solid #e5e5e5;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 8px;
    }
    #new-chat-btn {
        flex: 0 0 auto;
    }
    #right-panel {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    #chat-window {
        flex: 1 1 auto;
        min-height: 0;
        border: 1px solid #ddd;
        border-radius: 12px;
    }
    #input-box {
        flex: 0 0 auto;
        margin-top: 2px;
    }
    #actions-row {
        flex: 0 0 auto;
    }
    .ai-thought {
        color: #888;
        font-size: 13px;
        margin-bottom: 6px;
        white-space: pre-wrap;
    }
    .ai-answer {
        color: var(--body-text-color, #eaeaea);
        font-size: 15px;
        white-space: pre-wrap;
    }
    """)


if __name__ == "__main__":
    main()
