import asyncio
import json
import os
from unittest.mock import AsyncMock

import pytest
import respx
from fastapi.testclient import TestClient

import litellm


def _init_proxy_with_config(config_filename: str) -> TestClient:
    """Helper to initialize the FastAPI proxy app with a given config file."""
    # Reset global proxy state between tests
    from litellm.proxy.proxy_server import cleanup_router_config_variables, initialize, app

    cleanup_router_config_variables()
    filepath = os.path.dirname(os.path.abspath(__file__))
    config_fp = os.path.join(filepath, "test_configs", config_filename)
    # Initialize proxy with our test config
    asyncio.run(initialize(config=config_fp, debug=False))
    return TestClient(app)


@pytest.fixture(scope="function")
def client_agentrouter(monkeypatch) -> TestClient:
    # Master key for proxy auth
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-test-master")
    # AgentRouter key for upstream provider auth
    monkeypatch.setenv("AGENTROUTER_API_KEY", "fake-agentrouter-key")

    # Ensure httpx is used so respx can intercept
    litellm.disable_aiohttp_transport = True

    return _init_proxy_with_config("test_agentrouter_config.yaml")


@pytest.mark.asyncio
async def test_proxy_agentrouter_claude_nonstream_chat_completions_routes_to_messages_and_headers(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    # Mock AgentRouter /v1/messages (Anthropic) endpoint
    route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Anthropic hello via AgentRouter!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200

    # Upstream called
    assert route.called
    upstream_req = respx_mock.calls[0].request
    assert upstream_req.url.path.endswith("/v1/messages")
    # Anthropic-style headers
    assert ("x-api-key" in upstream_req.headers) or ("X-Api-Key" in upstream_req.headers)
    assert ("authorization" not in upstream_req.headers) and (
        "Authorization" not in upstream_req.headers
    )
    # User-Agent injected
    assert upstream_req.headers.get("User-Agent") == "claude-cli/2.0.15 (external, cli)" or upstream_req.headers.get(
        "user-agent"
    ) == "claude-cli/2.0.15 (external, cli)"

    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Anthropic hello via AgentRouter!"


@pytest.mark.asyncio
async def test_proxy_agentrouter_openai_nonstream_chat_completions_routes_to_chat_completions_and_headers(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    # Mock AgentRouter /v1/chat/completions endpoint
    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT-5!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200

    # Upstream called
    assert route.called
    upstream_req = respx_mock.calls[0].request
    assert upstream_req.url.path.endswith("/v1/chat/completions")
    # OpenAI-style headers
    assert ("authorization" in upstream_req.headers) or ("Authorization" in upstream_req.headers)
    assert not (("x-api-key" in upstream_req.headers) or ("X-Api-Key" in upstream_req.headers))

    data = resp.json()
    assert data["choices"][0]["message"]["content"] == "Hello from GPT-5!"


@pytest.mark.asyncio
async def test_proxy_agentrouter_claude_streaming_chat_completions(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    # Mock upstream SSE stream from AgentRouter (Anthropic format)
    anthropic_chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-haiku-20241022"}}\n\n',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" from"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" Claude!"}}\n\n',
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]

    route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(anthropic_chunks),
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
        "stream": True,
    }

    full_text = ""
    with client_agentrouter.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                # Some infra may split frames; skip unparsable lines
                continue
            # OpenAI-style streaming delta after proxy normalization
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if isinstance(delta, dict) and delta.get("content"):
                    full_text += delta["content"]

    # Upstream called
    assert route.called
    assert "Hello" in full_text and "Claude!" in full_text


@pytest.mark.parametrize(
    "agentrouter_model,expected_endpoint",
    [
        ("agentrouter/claude-3-5-haiku-20241022", "/v1/messages"),
        ("agentrouter/claude-haiku-4-5-20251001", "/v1/messages"),
        ("agentrouter/claude-sonnet-4-20250514", "/v1/messages"),
        ("agentrouter/gpt-5", "/v1/chat/completions"),
        ("agentrouter/deepseek-v3.1", "/v1/chat/completions"),
        ("agentrouter/deepseek-r1-0528", "/v1/chat/completions"),
        ("agentrouter/glm-4.6", "/v1/chat/completions"),
    ],
)
@pytest.mark.asyncio
async def test_proxy_health_test_connection_agentrouter_all_models(
    client_agentrouter: TestClient,
    respx_mock: respx.MockRouter,
    monkeypatch,
    agentrouter_model: str,
    expected_endpoint: str,
):
    # Patch prisma_client to bypass DB None check and auth checks to pass
    import litellm.proxy.proxy_server as proxy_server_mod
    from litellm.proxy.management_endpoints import model_management_endpoints

    proxy_server_mod.prisma_client = object()  # non-None sentinel
    monkeypatch.setattr(
        model_management_endpoints.ModelManagementAuthChecks,
        "can_user_make_model_call",
        AsyncMock(return_value=None),
    )

    # Mock upstream messages endpoint
    # Use stripped model (AgentRouter transformer removes provider prefix)
    clean_model = agentrouter_model.split("/", 1)[-1]

    if expected_endpoint == "/v1/messages":
        route = respx_mock.post(f"https://agentrouter.org{expected_endpoint}").respond(
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": clean_model,
                "content": [{"type": "text", "text": "Health OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }
        )
    else:
        route = respx_mock.post(f"https://agentrouter.org{expected_endpoint}").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": clean_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Health OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            }
        )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "mode": "chat",
        "litellm_params": {
            "model": agentrouter_model,
            "custom_llm_provider": "agentrouter",
            "api_key": "fake-agentrouter-key",
        },
        "model_info": {},
    }

    # Anthropic models require max_tokens
    if expected_endpoint == "/v1/messages":
        payload["litellm_params"]["max_tokens"] = 100

    resp = client_agentrouter.post("/health/test_connection", json=payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "success"
    assert route.called


@pytest.mark.asyncio
async def test_proxy_agentrouter_openai_streaming_chat_completions(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    # Mock upstream SSE stream from AgentRouter (OpenAI format)
    openai_chunks = [
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"content":" from"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{"content":" GPT!"},"finish_reason":null}]}\n\n',
        b'data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(openai_chunks),
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }

    full_text = ""
    with client_agentrouter.stream("POST", "/v1/chat/completions", json=payload, headers=headers) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if isinstance(delta, dict) and delta.get("content"):
                    full_text += delta["content"]

    # Upstream called
    assert route.called
    assert "Hello" in full_text and "GPT!" in full_text


@pytest.mark.asyncio
async def test_proxy_agentrouter_with_function_calling(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    # Mock response with function call
    route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"location": "San Francisco"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "What's the weather in San Francisco?"}],
        "max_tokens": 100,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert route.called

    data = resp.json()
    # Check that tool calls are properly formatted in response
    assert "choices" in data
    assert len(data["choices"]) > 0


@pytest.mark.asyncio
async def test_proxy_agentrouter_cross_model_headers_verification(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    """
    Cross-model test: Verify that Claude models use x-api-key and OpenAI models use Authorization.
    """
    # Test 1: Claude model should use x-api-key
    claude_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Claude response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert claude_route.called

    # Verify Claude request has x-api-key, NOT Authorization
    claude_req = respx_mock.calls[0].request
    assert ("x-api-key" in claude_req.headers) or ("X-Api-Key" in claude_req.headers)
    assert ("authorization" not in claude_req.headers) and ("Authorization" not in claude_req.headers)

    # Reset mock
    respx_mock.calls.clear()

    # Test 2: OpenAI model should use Authorization
    openai_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "GPT response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert openai_route.called

    # Verify OpenAI request has Authorization, NOT x-api-key
    openai_req = respx_mock.calls[0].request  # Index 0 after clear
    assert ("authorization" in openai_req.headers) or ("Authorization" in openai_req.headers)
    assert not (("x-api-key" in openai_req.headers) or ("X-Api-Key" in openai_req.headers))


@pytest.mark.asyncio
async def test_proxy_agentrouter_wrong_endpoint_for_model_type(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    """
    Cross-model test: Verify routing logic prevents wrong endpoint usage.
    Claude -> /v1/messages only
    OpenAI -> /v1/chat/completions only
    """
    # Mock both endpoints
    messages_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )

    chat_completions_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}

    # Test 1: Claude model should route to /v1/messages
    payload = {
        "model": "agentrouter/claude-3-5-haiku-20241022",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 16,
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert messages_route.called
    assert respx_mock.calls[0].request.url.path.endswith("/v1/messages")

    # Reset
    respx_mock.calls.clear()

    # Test 2: OpenAI model should route to /v1/chat/completions
    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": "Hi"}],
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert chat_completions_route.called
    assert respx_mock.calls[0].request.url.path.endswith("/v1/chat/completions")


@pytest.mark.asyncio
async def test_proxy_agentrouter_model_prefix_cleaning(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    """
    Cross-model test: Verify model names have provider prefix removed before API call.
    """
    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-v3.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/deepseek-v3.1",  # With prefix
        "messages": [{"role": "user", "content": "Hi"}],
    }

    resp = client_agentrouter.post("/v1/chat/completions", json=payload, headers=headers)
    assert resp.status_code == 200
    assert route.called

    # Verify model name was cleaned (prefix removed)
    upstream_req = respx_mock.calls[0].request
    import json as json_module
    body = json_module.loads(upstream_req.read())
    assert body["model"] == "deepseek-v3.1"
    assert "agentrouter/" not in body["model"]


@pytest.mark.asyncio
async def test_proxy_agentrouter_cross_format_anthropic_messages_with_openai_model(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    """
    Proxy cross-format test: Call /v1/messages with OpenAI model (gpt-5).
    Should auto-convert to OpenAI format.
    """
    # Mock OpenAI endpoint
    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from GPT-5 via cross-format!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    # User sends Anthropic Messages API request
    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        "max_tokens": 100,
    }

    # Call /v1/messages endpoint (Anthropic format)
    resp = client_agentrouter.post("/v1/messages", json=payload, headers=headers)
    assert resp.status_code == 200
    assert route.called

    # Verify upstream was called with OpenAI format
    upstream_req = respx_mock.calls[0].request
    assert upstream_req.url.path.endswith("/v1/chat/completions")  # NOT /v1/messages

    # Verify response is in Anthropic format
    data = resp.json()
    assert data["type"] == "message"
    assert data["role"] == "assistant"
    assert data["content"][0]["type"] == "text"
    assert "Hello from GPT-5" in data["content"][0]["text"]
    assert "input_tokens" in data["usage"]
    assert "output_tokens" in data["usage"]


@pytest.mark.asyncio
async def test_proxy_agentrouter_cross_format_with_tool_calling(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter
):
    """
    Proxy cross-format test: Tool calling with cross-format conversion.
    """
    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_weather_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "Tokyo", "unit": "celsius"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 60, "completion_tokens": 40, "total_tokens": 100},
        }
    )

    headers = {"Authorization": "Bearer sk-test-master"}
    payload = {
        "model": "agentrouter/gpt-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "What's the weather in Tokyo?"}]}],
        "max_tokens": 200,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        ]
    }

    resp = client_agentrouter.post("/v1/messages", json=payload, headers=headers)
    assert resp.status_code == 200
    assert route.called

    data = resp.json()
    # Verify tool_use in Anthropic format
    assert data["stop_reason"] == "tool_use"
    assert len(data["content"]) > 0
    tool_use_block = next((c for c in data["content"] if c["type"] == "tool_use"), None)
    assert tool_use_block is not None
    assert tool_use_block["name"] == "get_weather"
    assert tool_use_block["input"]["location"] == "Tokyo"


@pytest.mark.asyncio
async def test_proxy_agentrouter_cross_format_all_models(
    client_agentrouter: TestClient, respx_mock: respx.MockRouter, monkeypatch
):
    """
    Proxy cross-format test: Verify all non-Claude models work via /v1/messages.
    """
    import litellm.proxy.proxy_server as proxy_server_mod
    from litellm.proxy.management_endpoints import model_management_endpoints

    proxy_server_mod.prisma_client = object()
    monkeypatch.setattr(
        model_management_endpoints.ModelManagementAuthChecks,
        "can_user_make_model_call",
        AsyncMock(return_value=None),
    )

    test_models = [
        "agentrouter/gpt-5",
        "agentrouter/deepseek-v3.1",
        "agentrouter/grok-2",
    ]

    for model in test_models:
        respx_mock.calls.clear()

        clean_model = model.split("/", 1)[-1]
        route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": clean_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": f"Response from {clean_model}"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            }
        )

        headers = {"Authorization": "Bearer sk-test-master"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Test"}]}],
            "max_tokens": 100,
        }

        resp = client_agentrouter.post("/v1/messages", json=payload, headers=headers)
        assert resp.status_code == 200, f"Failed for {model}"
        assert route.called, f"Route not called for {model}"

        data = resp.json()
        assert data["type"] == "message", f"Wrong type for {model}"
        assert data["content"][0]["text"] == f"Response from {clean_model}", f"Wrong content for {model}"


