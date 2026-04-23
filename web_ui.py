from __future__ import annotations

import os

import gradio as gr

from config import HISTORY_FILE, MODEL_NAME, WORKSPACE_ROOT
from main import load_history, run_agent, save_history


def _history_to_chat_messages(agent_history: list[dict]) -> list[dict[str, str]]:
    chat_messages: list[dict[str, str]] = []
    for item in agent_history:
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        chat_messages.append({"role": role, "content": content})
    return chat_messages


def _submit_message(
    user_message: str,
    chat_messages: list[dict[str, str]] | None,
    agent_history: list[dict] | None,
) -> tuple[list[dict[str, str]], list[dict], str]:
    user_message = (user_message or "").strip()
    if not user_message:
        return chat_messages or [], agent_history or [], ""

    ui_messages = list(chat_messages or [])
    current_agent_history = list(agent_history or [])

    try:
        answer, new_agent_history = run_agent(user_message, current_agent_history)
        save_history(HISTORY_FILE, new_agent_history)
        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append({"role": "assistant", "content": answer})
        return ui_messages, new_agent_history, ""
    except Exception as exc:  # noqa: BLE001
        ui_messages.append({"role": "user", "content": user_message})
        ui_messages.append({"role": "assistant", "content": f"Agent Error: {exc}"})
        return ui_messages, current_agent_history, ""


def _clear_chat() -> tuple[list[dict[str, str]], list[dict], str]:
    save_history(HISTORY_FILE, [])
    return [], [], ""


def build_demo() -> gr.Blocks:
    initial_agent_history = load_history(HISTORY_FILE)
    initial_chat_messages = _history_to_chat_messages(initial_agent_history)

    with gr.Blocks(title="Mini OpenClaw Chat") as demo:
        gr.Markdown(
            """
            <div class=\"app-shell\">
              <div class=\"chat-title\">Mini OpenClaw Chat</div>
              <div class=\"chat-subtitle\">DeepSeek 风格对话界面，支持连续多轮上下文。</div>
            </div>
            """
        )
        gr.Markdown(f"当前模型: **{MODEL_NAME}**  |  受限目录: **{WORKSPACE_ROOT}**")

        chatbot = gr.Chatbot(
            value=initial_chat_messages,
            height=560,
            buttons=["copy"],
            layout="bubble",
        )
        message_box = gr.Textbox(
            label="输入消息",
            placeholder="例如：请先读取 README.md，再总结这个项目做了什么",
            lines=3,
        )

        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空会话")

        agent_state = gr.State(initial_agent_history)

        send_btn.click(
            fn=_submit_message,
            inputs=[message_box, chatbot, agent_state],
            outputs=[chatbot, agent_state, message_box],
        )
        message_box.submit(
            fn=_submit_message,
            inputs=[message_box, chatbot, agent_state],
            outputs=[chatbot, agent_state, message_box],
        )
        clear_btn.click(
            fn=_clear_chat,
            inputs=[],
            outputs=[chatbot, agent_state, message_box],
        )

    return demo


def main() -> None:
    demo = build_demo()
    server_port = int(os.getenv("WEB_PORT", "7860"))
    demo.launch(server_name="127.0.0.1", server_port=server_port, inbrowser=True, css="""
    .app-shell {max-width: 980px; margin: 0 auto;}
    .chat-title {font-size: 1.8rem; font-weight: 700; letter-spacing: 0.02em;}
    .chat-subtitle {opacity: 0.8; margin-top: 4px;}
    """)


if __name__ == "__main__":
    main()
