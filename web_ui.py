from __future__ import annotations

from contextlib import redirect_stdout
from html import escape
import io
import os

import gradio as gr

from config import HISTORY_FILE, MODEL_NAME, WORKSPACE_ROOT
from main import load_history, run_agent, save_history


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
    agent_history: list[dict] | None,
) -> tuple[list[dict[str, str]], list[dict], str, str]:
    user_message = (user_message or "").strip()
    if not user_message:
        return chat_messages or [], agent_history or [], "", _build_history_sidebar(agent_history or [])

    ui_messages = list(chat_messages or [])
    current_agent_history = list(agent_history or [])

    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            answer, new_agent_history = run_agent(user_message, current_agent_history)
        logs = buf.getvalue().splitlines()
        thoughts = [line.split("Thought/Reply:", 1)[1].strip() for line in logs if "Thought/Reply:" in line]
        thought_text = "\n".join([t for t in thoughts if t])

        save_history(HISTORY_FILE, new_agent_history)
        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append(
            {
                "role": "assistant",
                "content": _format_assistant_content(thought_text, answer),
            }
        )
        return ui_messages, new_agent_history, "", _build_history_sidebar(new_agent_history)
    except Exception as exc:  # noqa: BLE001
        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append(
            {
                "role": "assistant",
                "content": _format_assistant_content("", f"Agent Error: {exc}"),
            }
        )
        return ui_messages, current_agent_history, "", _build_history_sidebar(current_agent_history)


def _clear_chat() -> tuple[list[dict[str, str]], list[dict], str, str]:
    save_history(HISTORY_FILE, [])
    return [], [], "", "暂无历史对话"


def build_demo() -> gr.Blocks:
    initial_agent_history = load_history(HISTORY_FILE)
    initial_chat_messages = _history_to_chat_messages(initial_agent_history)
    initial_history_md = _build_history_sidebar(initial_agent_history)

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
                history_md = gr.Markdown(value=initial_history_md, elem_id="history-content")

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
                    lines=4,
                    elem_id="input-box",
                )

                with gr.Row(elem_id="actions-row"):
                    send_btn = gr.Button("发送", variant="primary")
                    clear_btn = gr.Button("清空会话")

        agent_state = gr.State(initial_agent_history)

        send_btn.click(
            fn=_submit_message,
            inputs=[message_box, chatbot, agent_state],
            outputs=[chatbot, agent_state, message_box, history_md],
        )
        message_box.submit(
            fn=_submit_message,
            inputs=[message_box, chatbot, agent_state],
            outputs=[chatbot, agent_state, message_box, history_md],
        )
        clear_btn.click(
            fn=_clear_chat,
            inputs=[],
            outputs=[chatbot, agent_state, message_box, history_md],
        )

    return demo


def main() -> None:
    demo = build_demo()
    server_port = int(os.getenv("WEB_PORT", "7860"))
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
        padding: 6px 4px 10px 4px;
    }
    .page-title {
        font-size: 30px;
        font-weight: 700;
        line-height: 1.1;
    }
    .page-meta {
        color: #666;
        margin-top: 4px;
        font-size: 13px;
    }
    #main-row {
        height: calc(100vh - 88px);
        gap: 10px;
    }
    #left-panel, #right-panel {
        height: 100%;
    }
    #left-panel {
        border: 1px solid #ddd;
        border-radius: 12px;
        padding: 10px;
    }
    #history-content {
        height: calc(100% - 48px);
        overflow-y: auto;
        white-space: pre-wrap;
        font-size: 14px;
    }
    #right-panel {
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    #chat-window {
        flex: 1 1 auto;
        min-height: 0;
        border: 1px solid #ddd;
        border-radius: 12px;
    }
    #input-box {
        flex: 0 0 auto;
    }
    .ai-thought {
        color: #888;
        font-size: 13px;
        margin-bottom: 6px;
        white-space: pre-wrap;
    }
    .ai-answer {
        color: #111;
        font-size: 15px;
        white-space: pre-wrap;
    }
    """)


if __name__ == "__main__":
    main()
