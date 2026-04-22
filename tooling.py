from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json


class ToolExecutionError(Exception):
    pass


def safe_resolve_path(root: Path, user_path: str) -> Path:
    candidate = (root / user_path).resolve() if not Path(user_path).is_absolute() else Path(user_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ToolExecutionError(f"路径越界: {candidate} 不在允许目录 {root} 内") from exc
    return candidate


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolRegistry:
    def __init__(self, root: Path):
        self.root = root
        self._tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        self.register(
            Tool(
                name="list_files",
                description="列出指定目录下的文件与子目录",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "相对工作目录路径，默认 .",
                        }
                    },
                    "required": [],
                    "additionalProperties": False,
                },
                handler=self._list_files,
            )
        )

        self.register(
            Tool(
                name="read_text_file",
                description="读取文本文件内容",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录文件路径"},
                        "max_chars": {
                            "type": "integer",
                            "description": "最多返回字符数，默认 4000",
                            "minimum": 100,
                            "maximum": 20000,
                        },
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                handler=self._read_text_file,
            )
        )

        self.register(
            Tool(
                name="write_text_file",
                description="写入文本文件（覆盖或追加）",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "相对工作目录文件路径"},
                        "content": {"type": "string", "description": "要写入的文本内容"},
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                            "description": "overwrite 覆盖，append 追加",
                        },
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False,
                },
                handler=self._write_text_file,
            )
        )

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def all_for_openai(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(self, name: str, raw_arguments: str) -> str:
        if name not in self._tools:
            raise ToolExecutionError(f"未知工具: {name}")

        try:
            arguments = json.loads(raw_arguments or "{}")
            if not isinstance(arguments, dict):
                raise ToolExecutionError("工具参数必须是 JSON object")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"工具参数不是合法 JSON: {exc}") from exc

        return self._tools[name].handler(arguments)

    def _list_files(self, args: dict[str, Any]) -> str:
        rel_path = args.get("path", ".")
        path = safe_resolve_path(self.root, rel_path)
        if not path.exists():
            raise ToolExecutionError(f"路径不存在: {rel_path}")
        if not path.is_dir():
            raise ToolExecutionError(f"目标不是目录: {rel_path}")

        items = []
        for p in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            kind = "dir" if p.is_dir() else "file"
            items.append({"name": p.name, "type": kind})
        return json.dumps({"path": str(path), "items": items}, ensure_ascii=False)

    def _read_text_file(self, args: dict[str, Any]) -> str:
        rel_path = args["path"]
        max_chars = int(args.get("max_chars", 4000))
        path = safe_resolve_path(self.root, rel_path)

        if not path.exists():
            raise ToolExecutionError(f"文件不存在: {rel_path}")
        if not path.is_file():
            raise ToolExecutionError(f"目标不是文件: {rel_path}")

        content = path.read_text(encoding="utf-8")
        clipped = content[:max_chars]
        return json.dumps(
            {
                "path": str(path),
                "content": clipped,
                "truncated": len(content) > len(clipped),
                "total_chars": len(content),
            },
            ensure_ascii=False,
        )

    def _write_text_file(self, args: dict[str, Any]) -> str:
        rel_path = args["path"]
        content = args["content"]
        mode = args.get("mode", "overwrite")

        path = safe_resolve_path(self.root, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "append":
            with path.open("a", encoding="utf-8") as f:
                f.write(content)
        else:
            path.write_text(content, encoding="utf-8")

        return json.dumps({"path": str(path), "mode": mode, "written_chars": len(content)}, ensure_ascii=False)
