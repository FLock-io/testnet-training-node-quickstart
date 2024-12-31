from typing import Dict, Any, List, Tuple
import json

DEFAULT_TOOL_PROMPT = (
    "You have access to the following tools:\n{tool_text}"
    "Use the following format if using a tool:\n"
    "```\n"
    "Action: tool name (one of [{tool_names}])\n"
    "Action Input: the input to the tool, in a JSON format representing the kwargs "
    """(e.g. ```{{"input": "hello world", "num_beams": 5}}```)\n"""
    "```\n"
)

DEFAULT_FUNCTION_SLOTS = "Action: {name}\nAction Input: {arguments}\n"


def tool_formater(tools: List[Dict[str, Any]]) -> str:
    tool_text = ""
    tool_names = []
    for tool in tools:
        param_text = ""
        for name, param in tool["parameters"]["properties"].items():
            required, enum, items = "", "", ""
            if name in tool["parameters"].get("required", []):
                required = ", required"

            if param.get("enum", None):
                enum = ", should be one of [{}]".format(", ".join(param["enum"]))

            if param.get("items", None):
                items = ", where each item should be {}".format(
                    param["items"].get("type", "")
                )

            param_text += "  - {name} ({type}{required}): {desc}{enum}{items}\n".format(
                name=name,
                type=param.get("type", ""),
                required=required,
                desc=param.get("description", ""),
                enum=enum,
                items=items,
            )

        tool_text += "> Tool Name: {name}\nTool Description: {desc}\nTool Args:\n{args}\n".format(
            name=tool["name"], desc=tool.get("description", ""), args=param_text
        )
        tool_names.append(tool["name"])

    return DEFAULT_TOOL_PROMPT.format(
        tool_text=tool_text, tool_names=", ".join(tool_names)
    )


def function_formatter(tool_calls, function_slots=DEFAULT_FUNCTION_SLOTS) -> str:
    functions: List[Tuple[str, str]] = []
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]  # parrallel function calls

    for tool_call in tool_calls:
        functions.append(
            (tool_call["name"], json.dumps(tool_call["arguments"], ensure_ascii=False))
        )

    elements = []
    for name, arguments in functions:
        text = function_slots.format(name=name, arguments=arguments)
        elements.append(text)

    return "\n".join(elements) + "\n"
