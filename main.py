from __future__ import annotations

from pathlib import Path
import json

from config import HISTORY_FILE, MAX_TURNS, MODEL_NAME, SYSTEM_PROMPT, WORKSPACE_ROOT
from llm_client import build_client
from tooling import ToolExecutionError, ToolRegistry


def load_history(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def save_history(path: Path, messages: list[dict]) -> None:
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")


def run_agent(user_input: str, history: list[dict]) -> tuple[str, list[dict]]:
    client = build_client()
    tools = ToolRegistry(WORKSPACE_ROOT)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": user_input}]

    final_answer = "未获得最终回答。"

    for turn in range(1, MAX_TURNS + 1):
        print(f"\n=== Turn {turn} ===")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools.all_for_openai(),
            tool_choice="auto",
            temperature=0.2,
        )

        msg = resp.choices[0].message
        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
        }

        if msg.tool_calls:
            assistant_message["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
        messages.append(assistant_message)

        if msg.content:
            print(f"Thought/Reply: {msg.content}")

        if not msg.tool_calls:
            final_answer = msg.content or "(空回答)"
            break

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            raw_args = tc.function.arguments
            print(f"Action: {tool_name}({raw_args})")
            try:
                tool_result = tools.execute(tool_name, raw_args)
            except ToolExecutionError as exc:
                tool_result = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
            except Exception as exc:  # noqa: BLE001
                tool_result = json.dumps({"ok": False, "error": f"工具执行异常: {exc}"}, ensure_ascii=False)

            print(f"Observation: {tool_result}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )

    new_history = [m for m in messages if m["role"] != "system"]
    return final_answer, new_history


def main() -> None:
    print("Mini OpenClaw 已启动。输入 exit 退出。")
    print(f"受限工作目录: {WORKSPACE_ROOT}")
    history = load_history(HISTORY_FILE)

    while True:
        user_input = input("\nYou> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            answer, history = run_agent(user_input, history)
        except Exception as exc:  # noqa: BLE001
            print(f"Agent Error: {exc}")
            continue

        save_history(HISTORY_FILE, history)
        print(f"\nFinal Answer: {answer}")


if __name__ == "__main__":
    main()
