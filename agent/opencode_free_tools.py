"""OpenCode Free wire names for enabled Hermes tools, without new executors."""

import copy
import json


def adapt_free_tools(tools, aliases):
    """Split real file-search capabilities into glob/grep and retain reverse dispatch."""
    aliases = dict(aliases)
    taken = {tool.get("name") for tool in tools}
    rewritten, targets = [], {}
    for tool in tools:
        name = tool.get("name")
        original = aliases.get(name, name)
        names = {"terminal": ("bash",), "read_file": ("read",),
                 "search_files": ("glob", "grep")}.get(original)
        if not names or any(alias in taken for alias in names):
            rewritten.append(tool)
            continue
        aliases.pop(name, None)
        for alias in names:
            adapted = copy.deepcopy(tool)
            adapted["name"] = alias
            if original == "search_files":
                target = "files" if alias == "glob" else "content"
                parameters = adapted["parameters"]
                parameters["properties"]["target"] = {
                    "type": "string", "enum": [target], "default": target,
                    "description": "Search target fixed by this tool's purpose.",
                }
                parameters["required"] = list(dict.fromkeys([*parameters.get("required", []), "target"]))
                adapted["description"] = (
                    "Find files by glob pattern. " if alias == "glob" else "Search file contents by regex. "
                ) + adapted.get("description", "")
                targets[alias] = target
            rewritten.append(adapted)
            aliases[alias] = original
            taken.add(alias)
    return rewritten, aliases, targets


def rewrite_free_history(items, aliases, targets):
    """Replay canonical Hermes calls using this request's matching wire names."""
    rewritten = []
    for item in items:
        if item.get("type") != "function_call":
            rewritten.append(item)
            continue
        original = item.get("name")
        candidates = [alias for alias, name in aliases.items() if name == original]
        if candidates:
            alias = candidates[0]
            if original == "search_files":
                try:
                    arguments = json.loads(item.get("arguments") or "{}")
                except (ValueError, TypeError):
                    arguments = {}
                target = arguments.get("target", "content") if isinstance(arguments, dict) else "content"
                alias = next((name for name in candidates if targets.get(name) == target), alias)
            item = {**item, "name": alias}
        rewritten.append(item)
    return rewritten


def bind_search_target(arguments, target):
    """Make glob/grep dispatch use the intended search mode even if the model omits it."""
    try:
        parsed = json.loads(arguments)
    except (ValueError, TypeError):
        return arguments  # Preserve malformed calls for the normal argument-error path.
    if not isinstance(parsed, dict):
        return arguments
    return json.dumps({**parsed, "target": target})


def adapt_free_chat_tools(tools):
    """Apply the same real-tool aliases to Chat Completions' nested schemas."""
    flattened = [{**tool["function"], "type": "function"} for tool in tools]
    adapted, aliases, targets = adapt_free_tools(flattened, {})
    return [
        {"type": "function", "function": {key: value for key, value in tool.items() if key != "type"}}
        for tool in adapted
    ], aliases, targets


def rewrite_free_chat_history(messages, aliases, targets):
    """Rewrite assistant function names without mutating the canonical conversation."""
    rewritten = []
    for message in messages:
        calls = message.get("tool_calls")
        if calls:
            adapted = []
            for call in calls:
                function = call.get("function", {})
                item = rewrite_free_history([{"type": "function_call", **function}], aliases, targets)[0]
                adapted.append({**call, "function": {**function, "name": item.get("name")}})
            message = {**message, "tool_calls": adapted}
        rewritten.append(message)
    return rewritten
