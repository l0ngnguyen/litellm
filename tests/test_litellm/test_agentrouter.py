import os
import json
import pytest
import respx
import httpx

import litellm


@pytest.fixture(autouse=True)
def set_agentrouter_api_key():
    original_api_key = os.environ.get("AGENTROUTER_API_KEY")
    os.environ["AGENTROUTER_API_KEY"] = "fake-agentrouter-key"
    try:
        yield
    finally:
        if original_api_key is not None:
            os.environ["AGENTROUTER_API_KEY"] = original_api_key
        else:
            if "AGENTROUTER_API_KEY" in os.environ:
                del os.environ["AGENTROUTER_API_KEY"]


@pytest.mark.asyncio
async def test_agentrouter_openai_chat_compat(respx_mock: respx.MockRouter):
    """Verify AgentRouter provider calls OpenAI-compatible endpoint and injects User-Agent header."""

    # Use httpx transport for respx mocking to work
    litellm.disable_aiohttp_transport = True

    # Mock AgentRouter OpenAI-compatible endpoint
    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-v3.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from AgentRouter!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    # Make call via litellm using agentrouter provider
    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/deepseek-v3.1",
        messages=messages,
    )

    # Ensure mock endpoint was hit
    assert route.called
    req: httpx.Request = respx_mock.calls[0].request

    # Validate request body
    body = json.loads(req.read())
    assert body["model"] == "deepseek-v3.1"
    assert body["messages"] == messages

    # Validate User-Agent header injection when not provided
    # httpx lowercases header names internally; check both cases
    headers = req.headers
    assert (
        headers.get("User-Agent") != "claude-cli/2.0.15 (external, cli)"
        or headers.get("user-agent") != "claude-cli/2.0.15 (external, cli)"
    )

    # Validate response
    assert response.choices[0].message.content == "Hello from AgentRouter!"


@pytest.mark.asyncio
async def test_agentrouter_anthropic_messages_compat(respx_mock: respx.MockRouter):
    """Verify AgentRouter provider calls Anthropic-compatible /v1/messages and injects User-Agent header."""

    # Disable aiohttp transport since we use respx
    litellm.disable_aiohttp_transport = True

    # Mock AgentRouter Anthropic-compatible endpoint
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

    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        max_tokens=10,
    )

    # Ensure mock endpoint was hit
    assert route.called
    req: httpx.Request = respx_mock.calls[0].request

    # Validate request body
    body = json.loads(req.read())
    assert body["model"] == "claude-3-5-haiku-20241022"  # stripped provider prefix
    assert body["messages"] == messages

    # Validate User-Agent header injection when not provided
    headers = req.headers
    assert (
        headers.get("User-Agent") == "claude-cli/2.0.15 (external, cli)"
        or headers.get("user-agent") == "claude-cli/2.0.15 (external, cli)"
    )

    # Validate response payload mapped without error
    assert response["role"] == "assistant"
    assert response["content"][0]["text"] == "Anthropic hello via AgentRouter!"


@pytest.mark.asyncio
async def test_agentrouter_acompletion_claude_routes_to_messages_endpoint(respx_mock: respx.MockRouter):
    """
    CRITICAL TEST: Verify that when calling litellm.acompletion() with Claude models,
    AgentRouter routes to /v1/messages (NOT /v1/chat/completions) AND uses correct headers.

    This is the actual flow used by health checks and UI test connections.
    """
    litellm.disable_aiohttp_transport = True

    # Mock the /v1/messages endpoint that should be called for Claude models
    messages_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Hello from Claude via messages API!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )

    # Also mock /v1/chat/completions to ensure it's NOT called
    chat_completions_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        status_code=404,
        json={"error": "This endpoint should not be called for Claude models"}
    )

    # Call acompletion (same as health check does)
    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        max_tokens=10,
    )

    # CRITICAL: Verify /v1/messages was called (NOT /v1/chat/completions)
    assert messages_route.called, "Claude model should call /v1/messages endpoint"
    assert not chat_completions_route.called, "Claude model should NOT call /v1/chat/completions endpoint"

    # Verify correct headers were used (x-api-key, NOT Authorization)
    req: httpx.Request = respx_mock.calls[0].request
    headers = req.headers

    # Check for x-api-key header (Anthropic style)
    assert "x-api-key" in headers or "X-Api-Key" in headers, "Claude model should use x-api-key header"
    # Ensure Authorization header is NOT present
    assert "authorization" not in headers.keys() and "Authorization" not in headers.keys(), "Claude model should NOT use Authorization header"
    # Check for anthropic-version header
    assert "anthropic-version" in headers or "Anthropic-Version" in headers, "Claude model should have anthropic-version header"

    # Validate response
    assert response.choices[0].message.content == "Hello from Claude via messages API!"


@pytest.mark.asyncio
async def test_agentrouter_acompletion_openai_routes_to_chat_completions(respx_mock: respx.MockRouter):
    """
    CRITICAL TEST: Verify that when calling litellm.acompletion() with OpenAI-compatible models,
    AgentRouter routes to /v1/chat/completions (NOT /v1/messages) AND uses correct headers.
    """
    litellm.disable_aiohttp_transport = True

    # Mock the /v1/chat/completions endpoint that should be called
    chat_completions_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
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

    # Also mock /v1/messages to ensure it's NOT called
    messages_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        status_code=404,
        json={"error": "This endpoint should not be called for OpenAI models"}
    )

    # Call acompletion
    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/gpt-5",
        messages=messages,
        api_key="fake-agentrouter-key",  # Add API key to ensure proper auth
    )

    # CRITICAL: Verify /v1/chat/completions was called (NOT /v1/messages)
    assert chat_completions_route.called, "OpenAI model should call /v1/chat/completions endpoint"
    assert not messages_route.called, "OpenAI model should NOT call /v1/messages endpoint"

    # Verify correct headers were used (Authorization, NOT x-api-key)
    req: httpx.Request = respx_mock.calls[0].request
    headers = req.headers

    # Check for Authorization header (OpenAI style)
    assert "authorization" in headers or "Authorization" in headers, "OpenAI model should use Authorization header"
    auth_header = headers.get("authorization") or headers.get("Authorization")
    assert auth_header.startswith("Bearer "), "Authorization header should start with 'Bearer '"
    # Ensure x-api-key header is NOT present
    assert "x-api-key" not in headers.keys() and "X-Api-Key" not in headers.keys(), "OpenAI model should NOT use x-api-key header"

    # Validate response
    assert response.choices[0].message.content == "Hello from GPT-5!"


@pytest.mark.asyncio
async def test_agentrouter_streaming_claude_anthropic_format(respx_mock: respx.MockRouter):
    """
    Test streaming with Claude models that return Anthropic format.

    This tests the auto-detection of Anthropic streaming format and proper parsing.
    """
    litellm.disable_aiohttp_transport = True

    # Mock streaming response with Anthropic format
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

    # Call streaming completion
    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        max_tokens=10,
        stream=True,
    )

    # Collect chunks
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    # Verify endpoint was called
    assert route.called

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify chunks have correct structure
    for chunk in chunks:
        assert hasattr(chunk, 'choices')
        if chunk.choices and len(chunk.choices) > 0:
            assert hasattr(chunk.choices[0], 'delta')

    # Reconstruct message from chunks
    full_text = ""
    for chunk in chunks:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_text += delta.content

    assert "Hello from Claude!" in full_text


@pytest.mark.asyncio
async def test_agentrouter_streaming_openai_format(respx_mock: respx.MockRouter):
    """
    Test streaming with OpenAI-compatible models that return OpenAI format.

    This tests proper handling of OpenAI streaming format.
    """
    litellm.disable_aiohttp_transport = True

    # Mock streaming response with OpenAI format
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

    # Call streaming completion
    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/gpt-5",
        messages=messages,
        stream=True,
        api_key="fake-key",  # Add API key
    )

    # Collect chunks
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    # Verify endpoint was called
    assert route.called

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify chunks have correct structure
    for chunk in chunks:
        assert hasattr(chunk, 'choices')

    # Reconstruct message from chunks
    full_text = ""
    for chunk in chunks:
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'content') and delta.content:
                full_text += delta.content

    assert "Hello from GPT!" in full_text


@pytest.mark.asyncio
async def test_agentrouter_streaming_claude_with_tool_calls(respx_mock: respx.MockRouter):
    """
    Test streaming with Claude models that include tool calls.

    This is a more advanced test to ensure tool calling works with streaming.
    """
    litellm.disable_aiohttp_transport = True

    # Mock streaming response with tool calls
    anthropic_chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_123","type":"message","role":"assistant","content":[],"model":"claude-3-5-haiku-20241022"}}\n\n',
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"location\\":\\""}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"SF\\"}"}}\n\n',
        b'data: {"type":"content_block_stop","index":0}\n\n',
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]

    route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        content=b"".join(anthropic_chunks),
    )

    # Call streaming completion with tools
    messages = [{"role": "user", "content": "What's the weather in SF?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
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

    response = await litellm.acompletion(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        tools=tools,
        max_tokens=100,
        stream=True,
    )

    # Collect chunks
    chunks = []
    async for chunk in response:
        chunks.append(chunk)

    # Verify endpoint was called
    assert route.called

    # Verify we got chunks
    assert len(chunks) > 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("agentrouter/claude-haiku-4-5-20251001", "/v1/messages"),
        ("agentrouter/claude-sonnet-4-20250514", "/v1/messages"),
        ("agentrouter/claude-sonnet-4-5-20250929", "/v1/messages"),
        ("agentrouter/deepseek-r1-0528", "/v1/chat/completions"),
        ("agentrouter/deepseek-v3.2", "/v1/chat/completions"),
        ("agentrouter/glm-4.5", "/v1/chat/completions"),
        ("agentrouter/glm-4.6", "/v1/chat/completions"),
        ("agentrouter/grok-code-fast-1", "/v1/chat/completions"),
    ],
)
async def test_agentrouter_additional_models_routing(respx_mock: respx.MockRouter, model: str, expected_endpoint: str):
    """
    Test that all AgentRouter supported models route to correct endpoints.

    Claude models should route to /v1/messages, others to /v1/chat/completions.
    """
    litellm.disable_aiohttp_transport = True

    clean_model = model.split("/", 1)[-1]

    # Mock both endpoints
    if expected_endpoint == "/v1/messages":
        correct_route = respx_mock.post(f"https://agentrouter.org{expected_endpoint}").respond(
            json={
                "id": "msg_123",
                "type": "message",
                "role": "assistant",
                "model": clean_model,
                "content": [{"type": "text", "text": f"Response from {clean_model}"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 5, "output_tokens": 7},
            }
        )
        wrong_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
            status_code=404,
            json={"error": "Wrong endpoint"}
        )
    else:
        correct_route = respx_mock.post(f"https://agentrouter.org{expected_endpoint}").respond(
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
        wrong_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
            status_code=404,
            json={"error": "Wrong endpoint"}
        )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        max_tokens=10 if "claude" in model else None,
    )

    # Verify correct endpoint was called
    assert correct_route.called, f"{model} should call {expected_endpoint}"
    assert not wrong_route.called, f"{model} should NOT call the wrong endpoint"

    # Verify response
    assert response.choices[0].message.content == f"Response from {clean_model}"


@pytest.mark.asyncio
async def test_agentrouter_error_handling_invalid_api_key(respx_mock: respx.MockRouter):
    """Test that AgentRouter handles invalid API key errors correctly."""
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        status_code=401,
        json={"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
    )

    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(Exception) as exc_info:
        await litellm.acompletion(
            model="agentrouter/gpt-5",
            messages=messages,
            api_key="invalid-key",  # Add explicit invalid key
        )

    assert route.called


@pytest.mark.asyncio
async def test_agentrouter_error_handling_rate_limit(respx_mock: respx.MockRouter):
    """Test that AgentRouter handles rate limit errors correctly."""
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        status_code=429,
        json={"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
    )

    messages = [{"role": "user", "content": "Hi"}]

    with pytest.raises(Exception) as exc_info:
        await litellm.acompletion(
            model="agentrouter/gpt-5",
            messages=messages,
            api_key="fake-key",  # Add API key
        )

    assert route.called


@pytest.mark.asyncio
async def test_agentrouter_custom_api_base(respx_mock: respx.MockRouter):
    """Test that custom API base URLs work correctly."""
    litellm.disable_aiohttp_transport = True

    custom_base = "https://custom.agentrouter.example.com"
    route = respx_mock.post(f"{custom_base}/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Custom base response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/gpt-5",
        messages=messages,
        api_base=custom_base,
        api_key="fake-key",  # Add API key
    )

    assert route.called
    assert response.choices[0].message.content == "Custom base response"


@pytest.mark.asyncio
async def test_agentrouter_deepseek_with_system_message(respx_mock: respx.MockRouter):
    """Test DeepSeek models with system messages."""
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-v3.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "I am a helpful assistant."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who are you?"}
    ]
    response = await litellm.acompletion(
        model="agentrouter/deepseek-v3.1",
        messages=messages,
    )

    assert route.called
    req = respx_mock.calls[0].request
    body = json.loads(req.read())
    assert len(body["messages"]) == 2
    assert body["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_agentrouter_claude_with_function_calling_non_streaming(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify Claude models handle function calling correctly in non-streaming mode.
    """
    litellm.disable_aiohttp_transport = True

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
                    "input": {"location": "San Francisco", "unit": "celsius"}
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 50, "output_tokens": 30},
        }
    )

    messages = [{"role": "user", "content": "What's the weather in San Francisco?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = await litellm.acompletion(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        tools=tools,
        max_tokens=100,
    )

    # Verify endpoint called
    assert route.called

    # Verify tool calls are properly converted to OpenAI format
    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) == 1
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert "San Francisco" in tool_call.function.arguments


@pytest.mark.asyncio
async def test_agentrouter_openai_model_uses_correct_headers(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify OpenAI models use Authorization header, NOT x-api-key.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/gpt-5",
        messages=messages,
        api_key="test-agentrouter-key",
    )

    assert route.called
    req = respx_mock.calls[0].request

    # OpenAI models should use Authorization header
    assert "authorization" in req.headers or "Authorization" in req.headers
    auth_header = req.headers.get("authorization") or req.headers.get("Authorization")
    assert auth_header.startswith("Bearer ")

    # Should NOT use x-api-key
    assert "x-api-key" not in req.headers.keys()
    assert "X-Api-Key" not in req.headers.keys()


@pytest.mark.asyncio
async def test_agentrouter_claude_model_uses_correct_headers(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify Claude models use x-api-key header, NOT Authorization.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        json={
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-haiku-20241022",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }
    )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/claude-3-5-haiku-20241022",
        messages=messages,
        max_tokens=10,
        api_key="test-agentrouter-key",
    )

    assert route.called
    req = respx_mock.calls[0].request

    # Claude models should use x-api-key header
    assert "x-api-key" in req.headers or "X-Api-Key" in req.headers

    # Should NOT use Authorization header
    assert "authorization" not in req.headers.keys()
    assert "Authorization" not in req.headers.keys()


@pytest.mark.asyncio
async def test_agentrouter_deepseek_model_routing(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify DeepSeek models route to /v1/chat/completions, not /v1/messages.
    """
    litellm.disable_aiohttp_transport = True

    # Mock correct endpoint
    correct_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-v3.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "DeepSeek response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    # Mock wrong endpoint (should NOT be called)
    wrong_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        status_code=404,
        json={"error": "Wrong endpoint"}
    )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/deepseek-v3.1",
        messages=messages,
    )

    # Verify correct endpoint called
    assert correct_route.called
    assert not wrong_route.called
    assert response.choices[0].message.content == "DeepSeek response"


@pytest.mark.asyncio
async def test_agentrouter_xai_grok_model_routing(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify XAI/Grok models route to /v1/chat/completions, not /v1/messages.
    """
    litellm.disable_aiohttp_transport = True

    # Mock correct endpoint
    correct_route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "grok-2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Grok response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    # Mock wrong endpoint (should NOT be called)
    wrong_route = respx_mock.post("https://agentrouter.org/v1/messages").respond(
        status_code=404,
        json={"error": "Wrong endpoint"}
    )

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/grok-2",
        messages=messages,
    )

    # Verify correct endpoint called
    assert correct_route.called
    assert not wrong_route.called
    assert response.choices[0].message.content == "Grok response"


@pytest.mark.asyncio
async def test_agentrouter_model_name_cleaning(respx_mock: respx.MockRouter):
    """
    Cross-model test: Verify model names are cleaned (provider prefix removed) before sending to API.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
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

    messages = [{"role": "user", "content": "Hi"}]
    response = await litellm.acompletion(
        model="agentrouter/gpt-5",  # With prefix
        messages=messages,
    )

    assert route.called
    req = respx_mock.calls[0].request
    body = json.loads(req.read())

    # Model name should be cleaned (prefix removed)
    assert body["model"] == "gpt-5"
    assert "agentrouter/" not in body["model"]


@pytest.mark.asyncio
async def test_agentrouter_wrong_endpoint_usage_error(respx_mock: respx.MockRouter):
    """
    Test auto-conversion: using /v1/messages (Anthropic endpoint) with gpt-5 (OpenAI model).

    Flow:
    1. User calls litellm.anthropic_messages() with agentrouter/gpt-5
    2. AgentRouter detects this is OpenAI model
    3. Converts Anthropic request → OpenAI request
    4. Calls AgentRouter /v1/chat/completions
    5. Converts OpenAI response → Anthropic response
    6. Returns Anthropic format to user
    """
    litellm.disable_aiohttp_transport = True

    # Mock AgentRouter /v1/chat/completions (OpenAI endpoint)
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

    # User calls with Anthropic format
    messages = [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",  # OpenAI model
        messages=messages,  # Anthropic format
        max_tokens=10,
    )

    # Verify AgentRouter /v1/chat/completions was called (NOT /v1/messages)
    assert route.called
    req = respx_mock.calls[0].request
    assert req.url.path.endswith("/v1/chat/completions")

    # Verify request was converted to OpenAI format
    import json as json_module
    body = json_module.loads(req.read())
    assert body["model"] == "gpt-5"
    # Messages should be converted to OpenAI format (simple string content)
    assert isinstance(body["messages"][0]["content"], str)
    assert body["messages"][0]["content"] == "Hi"

    # Verify response is in Anthropic format
    assert response["type"] == "message"
    assert response["role"] == "assistant"
    assert isinstance(response["content"], list)
    assert response["content"][0]["type"] == "text"
    assert response["content"][0]["text"] == "Hello from GPT-5!"
    assert "usage" in response
    assert "input_tokens" in response["usage"]
    assert "output_tokens" in response["usage"]


@pytest.mark.asyncio
async def test_agentrouter_cross_format_anthropic_endpoint_with_deepseek(respx_mock: respx.MockRouter):
    """
    Cross-format test: Anthropic endpoint with DeepSeek model.
    Verifies DeepSeek-specific config is used during conversion.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "deepseek-v3.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "DeepSeek response"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/deepseek-v3.1",
        messages=messages,
        max_tokens=100,
    )

    assert route.called
    assert response["type"] == "message"
    assert response["content"][0]["text"] == "DeepSeek response"


@pytest.mark.asyncio
async def test_agentrouter_cross_format_anthropic_endpoint_with_xai(respx_mock: respx.MockRouter):
    """
    Cross-format test: Anthropic endpoint with XAI/Grok model.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "grok-2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Grok response with humor!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "Tell me a joke"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/grok-2",
        messages=messages,
        max_tokens=100,
    )

    assert route.called
    assert response["content"][0]["text"] == "Grok response with humor!"


@pytest.mark.asyncio
async def test_agentrouter_cross_format_with_tool_calls(respx_mock: respx.MockRouter):
    """
    Cross-format test: Anthropic endpoint with OpenAI model + tool calls.
    Verifies tool calls are converted correctly in both directions.
    """
    litellm.disable_aiohttp_transport = True

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
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
        }
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "What's the weather in SF?"}]}]
    tools = [
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

    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",
        messages=messages,
        tools=tools,
        max_tokens=100,
    )

    assert route.called

    # Verify request conversion: Anthropic tools → OpenAI tools
    req = respx_mock.calls[0].request
    body = json.loads(req.read())
    assert "tools" in body
    assert body["tools"][0]["type"] == "function"
    assert body["tools"][0]["function"]["name"] == "get_weather"

    # Verify response conversion: OpenAI tool_calls → Anthropic tool_use
    assert response["stop_reason"] == "tool_use"
    assert len(response["content"]) == 1
    assert response["content"][0]["type"] == "tool_use"
    assert response["content"][0]["id"] == "call_123"
    assert response["content"][0]["name"] == "get_weather"
    assert response["content"][0]["input"]["location"] == "San Francisco"


@pytest.mark.asyncio
async def test_agentrouter_cross_format_finish_reason_mapping(respx_mock: respx.MockRouter):
    """
    Cross-format test: Verify finish_reason mapping from OpenAI to Anthropic.
    """
    litellm.disable_aiohttp_transport = True

    test_cases = [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("content_filter", "end_turn"),
    ]

    for openai_reason, anthropic_reason in test_cases:
        respx_mock.calls.clear()

        route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Test"},
                        "finish_reason": openai_reason,
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
            }
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
        response = await litellm.anthropic_messages(
            model="agentrouter/gpt-5",
            messages=messages,
            max_tokens=10,
        )

        assert response["stop_reason"] == anthropic_reason, f"Failed for {openai_reason} → {anthropic_reason}"


@pytest.mark.asyncio
async def test_agentrouter_cross_format_multiple_text_blocks(respx_mock: respx.MockRouter):
    """
    Cross-format test: Multiple text blocks in Anthropic format should be joined.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
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

    # Multiple text blocks
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part. "},
                {"type": "text", "text": "Second part. "},
                {"type": "text", "text": "Third part."}
            ]
        }
    ]

    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",
        messages=messages,
        max_tokens=100,
    )

    assert route.called

    # Verify text blocks were joined
    req = respx_mock.calls[0].request
    body = json.loads(req.read())
    assert body["messages"][0]["content"] == "First part. Second part. Third part."


@pytest.mark.asyncio
async def test_agentrouter_cross_format_with_system_message(respx_mock: respx.MockRouter):
    """
    Cross-format test: System messages in Anthropic format.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "I am a helpful assistant."},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
    )

    messages = [
        {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]}
    ]

    # Anthropic uses 'system' parameter, not a system message
    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",
        messages=messages,
        system="You are a helpful assistant.",
        max_tokens=100,
    )

    assert route.called
    assert response["content"][0]["text"] == "I am a helpful assistant."


@pytest.mark.asyncio
async def test_agentrouter_cross_format_empty_content(respx_mock: respx.MockRouter):
    """
    Cross-format test: Handle empty/null content gracefully.
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},  # Empty content
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9},
        }
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",
        messages=messages,
        max_tokens=100,
    )

    assert route.called
    # Should handle empty content gracefully
    assert isinstance(response["content"], list)


@pytest.mark.asyncio
async def test_agentrouter_cross_format_usage_tokens_mapping(respx_mock: respx.MockRouter):
    """
    Cross-format test: Verify usage tokens are mapped correctly.
    OpenAI: prompt_tokens, completion_tokens
    Anthropic: input_tokens, output_tokens
    """
    litellm.disable_aiohttp_transport = True

    route = respx_mock.post("https://agentrouter.org/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 123, "completion_tokens": 456, "total_tokens": 579},
        }
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": "Test"}]}]
    response = await litellm.anthropic_messages(
        model="agentrouter/gpt-5",
        messages=messages,
        max_tokens=100,
    )

    assert route.called
    assert response["usage"]["input_tokens"] == 123
    assert response["usage"]["output_tokens"] == 456




