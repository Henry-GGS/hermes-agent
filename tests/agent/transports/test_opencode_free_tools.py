"""Real-tool dispatch and history replay for OpenCode Free wire aliases."""

import copy
import json
from types import SimpleNamespace
import pytest

from agent.chat_completion_helpers import build_api_kwargs
from agent.transports.codex import ResponsesApiTransport
from agent.transports.chat_completions import ChatCompletionsTransport
from run_agent import AIAgent


def _agent(mode):
    return AIAgent(
        provider="opencode-free", api_key="opencode-zen-free-keyless",
        base_url="https://opencode.ai/zen/v1", model=("muse-spark-1.3-contributor-free"
            if mode == "codex_responses" else "nemotron-3.5-lightning-free"),
        api_mode=mode, session_id="free-tool-adapter-test",
        enabled_toolsets=["terminal", "file"], quiet_mode=True,
        skip_memory=True, skip_context_files=True, save_trajectories=False,
    )


def _definitions(kwargs):
    return {tool.get("name"): tool for outer in kwargs["tools"] for tool in [outer.get("function", outer)]}


def _response(mode, name, arguments, index=0):
    if mode == "codex_responses":
        return SimpleNamespace(status="completed", output=[SimpleNamespace(
            type="function_call", id=f"fc_{index}", call_id=f"call_{index}", name=name, arguments=arguments,
        )])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason="tool_calls", message=SimpleNamespace(
        content=None, tool_calls=[SimpleNamespace(id=f"call_{index}", function=SimpleNamespace(
            name=name, arguments=arguments,
        ))],
    ))])


@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions"])
def test_free_aliases_dispatch_real_searches_and_replay_canonical_history(tmp_path, mode):
    from model_tools import handle_function_call

    agent = _agent(mode)
    original_tools = copy.deepcopy(agent.tools)
    messages = [{"role": "user", "content": "Find files and search their contents"}]
    first = build_api_kwargs(agent, messages)
    definitions = _definitions(first)
    assert {"bash", "read", "glob", "grep"} <= definitions.keys()
    assert agent.tools == original_tools
    assert definitions["glob"]["parameters"]["properties"]["target"]["enum"] == ["files"]
    assert definitions["grep"]["parameters"]["properties"]["target"]["enum"] == ["content"]

    (tmp_path / "adapter.txt").write_text("adapter_unique_marker\n", encoding="utf-8")
    calls = []
    for index, (name, pattern) in enumerate((("glob", "*.txt"), ("grep", "adapter_unique_marker"))):
        response = _response(mode, name, json.dumps({"pattern": pattern, "path": str(tmp_path), "target": "invalid"}), index)
        call = agent._get_transport().normalize_response(response).tool_calls[0]
        assert call.name == "search_files"
        arguments = json.loads(call.arguments)
        assert arguments["target"] == ("files" if name == "glob" else "content")
        output = handle_function_call(call.name, arguments, task_id="free-adapter-test")
        assert "adapter.txt" in output
        if name == "grep":
            assert "adapter_unique_marker" in output
        calls.append((call, output))

    messages.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": call.id, "type": "function", "function": {"name": call.name, "arguments": call.arguments}}
        for call, _ in calls
    ]})
    messages.extend({"role": "tool", "tool_call_id": call.id, "content": output} for call, output in calls)
    second = build_api_kwargs(agent, messages)
    assert second["tools"] == first["tools"]
    assert second.get("instructions") == first.get("instructions")
    assert second.get("prompt_cache_key") == first.get("prompt_cache_key")
    if mode == "codex_responses":
        assert [item["name"] for item in second["input"] if item.get("type") == "function_call"] == ["glob", "grep"]
    else:
        assert [call["function"]["name"] for message in second["messages"]
                for call in message.get("tool_calls", [])] == ["glob", "grep"]
    assert all(call.name == "search_files" for call, _ in calls)


@pytest.mark.parametrize("mode", ["codex_responses", "chat_completions"])
def test_paid_and_disabled_tools_keep_their_contracts_and_reset_aliases(mode):
    agent = _agent(mode)
    transport = ResponsesApiTransport() if mode == "codex_responses" else ChatCompletionsTransport()
    messages = [{"role": "user", "content": "Hello"}]
    free = transport.build_kwargs(agent.model, messages, agent.tools,
                                  provider="opencode-free", base_url=agent.base_url)
    assert {"bash", "read", "glob", "grep"} <= _definitions(free).keys()
    paid = transport.build_kwargs("gpt-5.6-luna", messages, agent.tools,
                                  provider="opencode-go", base_url="https://opencode.ai/zen/go/v1")
    names = _definitions(paid).keys()
    assert {"terminal", "read_file", "hermes_search_files" if mode == "codex_responses" else "search_files"} <= names
    assert not {"bash", "read", "glob", "grep"} & names
    response = _response(mode, "glob", "{}")
    call = transport.normalize_response(response).tool_calls[0]
    assert call.name == "glob" and call.arguments == "{}"
    assert transport._last_free_search_targets == {}

    disabled = transport.build_kwargs(agent.model, messages, [],
                                      provider="opencode-free", base_url=agent.base_url)
    assert "tools" not in disabled
    terminal_only = [tool for tool in agent.tools if tool.get("function", {}).get("name") == "terminal"]
    limited = transport.build_kwargs(agent.model, messages, terminal_only,
                                     provider="opencode-free", base_url=agent.base_url)
    assert _definitions(limited).keys() == {"bash"}
