# AgentRouter Provider Documentation

## Tổng quan

AgentRouter là một provider proxy hỗ trợ routing requests đến nhiều LLM providers khác nhau (OpenAI, Anthropic, v.v.). Provider này tự động phát hiện model type và route đến endpoint phù hợp với headers đúng format.

## Kiến trúc

### 1. Hai luồng xử lý phối hợp

#### Luồng 1: Setup ban đầu (main.py, dòng 2758-2808)

```python
# File: litellm/main.py
elif custom_llm_provider == "agentrouter":
    # Step 1: Setup api_base
    api_base = api_base or "https://agentrouter.org"

    # Step 2: Get API key từ environment
    api_key = api_key or get_secret("AGENTROUTER_API_KEY")

    # Step 3: Add User-Agent header (preliminary)
    headers["User-Agent"] = "claude-cli/2.0.15 (external, cli)"

    # Step 4: Delegate đến base_llm_http_handler
    response = base_llm_http_handler.completion(
        custom_llm_provider="agentrouter",  # ← Trigger transformer lookup
        ...
    )
```

**Vai trò:**
- Setup các giá trị mặc định (api_base, api_key)
- Đảm bảo User-Agent header tồn tại
- Delegate việc xử lý chi tiết cho `base_llm_http_handler`

#### Luồng 2: Transformation logic (transformation.py)

```python
# File: litellm/llms/agentrouter/chat/transformation.py
class AgentrouterConfig(OpenAIGPTConfig):
    """Provider-specific transformation logic"""
```

**Vai trò:**
- Được gọi TỰ ĐỘNG bởi `base_llm_http_handler` dựa trên `custom_llm_provider="agentrouter"`
- Thực hiện logic chi tiết cho từng model type
- Transform request/response giữa các format khác nhau

### 2. Luồng hoàn chỉnh từ đầu đến cuối

```
┌─────────────────────────────────────────────────────────────────┐
│ User calls:                                                     │
│ litellm.completion(model="agentrouter/claude-3-5-sonnet")      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ main.py (line 2758): HIGH-LEVEL SETUP                          │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • api_base = "https://agentrouter.org"                      │ │
│ │ • api_key = get_secret("AGENTROUTER_API_KEY")              │ │
│ │ • headers["User-Agent"] = "claude-cli/2.0.15"              │ │
│ └─────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ base_llm_http_handler.completion()                             │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ 1. Lookup config class:                                     │ │
│ │    config = get_llm_provider_config("agentrouter")         │ │
│ │    → Returns: AgentrouterConfig instance                   │ │
│ └─────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ transformation.py: PROVIDER-SPECIFIC LOGIC                     │
│                                                                 │
│ Step 1: validate_environment()                                 │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • Detect model type: "claude" in model name?                │ │
│ │                                                              │ │
│ │ IF Claude model:                                            │ │
│ │   ✓ headers["x-api-key"] = api_key                         │ │
│ │   ✓ headers["anthropic-version"] = "2023-06-01"            │ │
│ │   ✓ headers["content-type"] = "application/json"           │ │
│ │                                                              │ │
│ │ ELSE (OpenAI/other models):                                 │ │
│ │   ✓ Call super().validate_environment()                    │ │
│ │   ✓ → headers["Authorization"] = f"Bearer {api_key}"       │ │
│ │                                                              │ │
│ │ • Ensure User-Agent header exists                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Step 2: get_complete_url()                                     │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • Detect model type: "claude" in model name?                │ │
│ │                                                              │ │
│ │ IF Claude model:                                            │ │
│ │   → https://agentrouter.org/v1/messages                     │ │
│ │                                                              │ │
│ │ ELSE (OpenAI/other models):                                 │ │
│ │   → https://agentrouter.org/v1/chat/completions             │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Step 3: Send HTTP Request                                      │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ POST {url} with finalized headers                           │ │
│ └─────────────────────────────────────────────────────────────┘ │
│                                                                 │
│ Step 4: transform_response()                                   │
│ ┌─────────────────────────────────────────────────────────────┐ │
│ │ • Detect model type: "claude" in model name?                │ │
│ │                                                              │ │
│ │ IF Claude model:                                            │ │
│ │   ✓ Delegate to AnthropicConfig.transform_response()       │ │
│ │   ✓ Transform Anthropic format → OpenAI format             │ │
│ │                                                              │ │
│ │ ELSE (OpenAI/other models):                                 │ │
│ │   ✓ Call super().transform_response()                      │ │
│ │   ✓ Use OpenAI transformer                                  │ │
│ └─────────────────────────────────────────────────────────────┘ │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Return: ModelResponse (chuẩn OpenAI format)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Chiến lược Minimal Override

### Nguyên tắc thiết kế

1. **Chỉ override những gì thực sự cần thiết**
2. **Sử dụng `super()` để giữ nguyên logic của parent class**
3. **Chỉ thêm custom logic cho các trường hợp đặc biệt**

### Các method được override

#### 1. `get_complete_url()`

**Lý do override:** Model type quyết định endpoint khác nhau

```python
def get_complete_url(self, api_base, model, ...):
    clean_model = model.split("/", 1)[-1]

    if "claude" in clean_model.lower():
        endpoint = "/v1/messages"          # Anthropic Messages API
    else:
        endpoint = "/v1/chat/completions"  # OpenAI Chat Completions

    return f"{base}{endpoint}"
```

**Không thể dùng super():** Vì parent class luôn trả về `/chat/completions`

#### 2. `validate_environment()`

**Lý do override:** Header format khác nhau cho mỗi model type

```python
def validate_environment(self, headers, model, api_key, ...):
    clean_model = model.split("/", 1)[-1]

    if "claude" in clean_model.lower():
        # Custom logic cho Claude - KHÔNG dùng super()
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
        headers["content-type"] = "application/json"
    else:
        # OpenAI models - DÙng super() để giữ nguyên logic cũ
        headers = super().validate_environment(
            headers=headers,
            model=clean_model,
            api_key=api_key,
            ...
        )

    # Ensure User-Agent cho tất cả models
    if "user-agent" not in headers:
        headers["User-Agent"] = "claude-cli/2.0.15 (external, cli)"

    return headers
```

**Dùng super() có chọn lọc:**
- Claude models: Custom headers → KHÔNG gọi super()
- OpenAI models: Gọi super() để dùng `Authorization: Bearer {api_key}`

#### 3. `transform_response()`

**Lý do override:** Response format khác nhau cho mỗi model type

```python
def transform_response(self, model, raw_response, ...):
    clean_model = model.split("/", 1)[-1]

    if "claude" in clean_model.lower():
        # Claude models - delegate đến Anthropic transformer
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig
        anthropic_config = AnthropicConfig()
        return anthropic_config.transform_response(
            model=clean_model,
            raw_response=raw_response,
            ...
        )
    else:
        # OpenAI models - gọi super() để dùng OpenAI transformer
        return super().transform_response(
            model=clean_model,
            raw_response=raw_response,
            ...
        )
```

**Dùng super() có chọn lọc:**
- Claude models: Delegate đến AnthropicConfig
- OpenAI models: Gọi super() để dùng OpenAI transformation

### Các method KHÔNG cần override

- ❌ `get_error_class()` - Parent class xử lý tốt rồi
- ❌ `transform_request()` - Parent class xử lý tốt rồi
- ❌ `get_supported_openai_params()` - Parent class xử lý tốt rồi (trừ class thứ 2)

## Hai transformer classes

### Class 1: `AgentrouterConfig` (extends OpenAIGPTConfig)

**Mục đích:** Xử lý requests theo OpenAI Chat Completions format

**Sử dụng cho:**
- OpenAI models: gpt-4, gpt-3.5-turbo, v.v.
- Claude models qua Chat Completions endpoint
- Bất kỳ model nào khác support OpenAI-compatible API

**Endpoint:** `/v1/chat/completions` (hoặc `/v1/messages` cho Claude)

### Class 2: `AgentRouterAnthropicMessagesConfig` (extends AnthropicMessagesConfig)

**Mục đích:** Xử lý requests theo Anthropic Messages API native format

**Sử dụng cho:**
- Anthropic models với native Messages API
- Khi user muốn dùng Anthropic-specific features (thinking, extended_thinking, v.v.)

**Endpoint:** `/v1/messages`

**Điểm khác biệt:**
- Không transform từ OpenAI format sang Anthropic
- Passthrough native Anthropic request format
- Support đầy đủ Anthropic-specific params

## Model Detection Logic

### Phát hiện Claude models

```python
clean_model = model.split("/", 1)[-1] if "/" in model else model

if "claude" in clean_model.lower():
    # → Claude model
else:
    # → OpenAI hoặc model khác
```

**Ví dụ:**
- `"agentrouter/claude-3-5-sonnet"` → `"claude-3-5-sonnet"` → Claude ✓
- `"agentrouter/gpt-4"` → `"gpt-4"` → OpenAI ✓
- `"claude-3-opus"` → `"claude-3-opus"` → Claude ✓

## Environment Variables

```bash
# API key cho AgentRouter
AGENTROUTER_API_KEY=your-agentrouter-key
# Hoặc
AR_API_KEY=your-agentrouter-key

# API base (optional, default: https://agentrouter.org)
AGENTROUTER_API_BASE=https://agentrouter.org
```

## Usage Examples

### Example 1: Claude model

```python
import litellm

response = litellm.completion(
    model="agentrouter/claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Internally:
# → URL: https://agentrouter.org/v1/messages
# → Headers: {
#     "x-api-key": "...",
#     "anthropic-version": "2023-06-01",
#     "content-type": "application/json",
#     "User-Agent": "claude-cli/2.0.15 (external, cli)"
# }
# → Response transformed: Anthropic format → OpenAI format
```

### Example 2: OpenAI model

```python
import litellm

response = litellm.completion(
    model="agentrouter/gpt-4",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

# Internally:
# → URL: https://agentrouter.org/v1/chat/completions
# → Headers: {
#     "Authorization": "Bearer ...",
#     "Content-Type": "application/json",
#     "User-Agent": "claude-cli/2.0.15 (external, cli)"
# }
# → Response: Already in OpenAI format, no transformation needed
```

## Testing

### Unit tests

```bash
# Run all AgentRouter tests
pytest tests/test_litellm/test_agentrouter.py -v

# Tests included:
# ✓ Claude model routes to /v1/messages endpoint
# ✓ OpenAI model routes to /v1/chat/completions endpoint
# ✓ Anthropic Messages API compatibility
# ✓ OpenAI Chat API compatibility
```

### Manual testing

```bash
# Test với Claude model
python -c "
import litellm
response = litellm.completion(
    model='agentrouter/claude-3-5-sonnet-20241022',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(response)
"

# Test với OpenAI model
python -c "
import litellm
response = litellm.completion(
    model='agentrouter/gpt-4',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print(response)
"
```

## Debugging

### Enable debug logging

```python
import litellm
litellm.set_verbose = True

# Sẽ in ra:
# - Headers được gửi đi
# - URL endpoint được gọi
# - Request/response body
```

### Common issues

1. **Authentication errors**
   - Kiểm tra `AGENTROUTER_API_KEY` environment variable
   - Verify API key còn hiệu lực

2. **Wrong endpoint**
   - Check model name có chứa "claude" hay không
   - Verify `get_complete_url()` logic

3. **Wrong headers**
   - Check `validate_environment()` logic
   - Verify model detection đúng

## Architecture Decisions

### Tại sao không dùng một class duy nhất?

**Option 1:** Một class xử lý tất cả
```python
class AgentrouterConfig:
    def validate_environment(...):
        if is_anthropic_native_format:
            # Logic 1
        elif is_claude_via_chat_completions:
            # Logic 2
        elif is_openai:
            # Logic 3
```

**Problem:** Logic phức tạp, khó maintain, nhiều if/else

**Option 2:** Hai class riêng biệt (Current approach)
```python
class AgentrouterConfig(OpenAIGPTConfig):
    # Chat Completions format cho mọi model

class AgentRouterAnthropicMessagesConfig(AnthropicMessagesConfig):
    # Native Anthropic Messages format
```

**Benefits:**
- ✅ Separation of concerns
- ✅ Dễ maintain
- ✅ Tận dụng inheritance
- ✅ Ít if/else logic

### Tại sao override minimal?

**Philosophy:**
> "Don't reinvent the wheel. Reuse parent class logic whenever possible."

**Benefits:**
1. **Ít code hơn** → ít bugs hơn
2. **Tận dụng tested code** → parent class đã được test kỹ
3. **Dễ maintain** → changes trong parent tự động áp dụng
4. **Rõ ràng hơn** → chỉ override những gì khác biệt

**Example:**
```python
# ❌ BAD: Duplicate all logic
def validate_environment(self, ...):
    # Copy 50 lines from parent
    headers["Authorization"] = f"Bearer {api_key}"
    headers["Content-Type"] = "application/json"
    # ... many more lines
    return headers

# ✅ GOOD: Reuse parent, only customize what's needed
def validate_environment(self, ...):
    if "claude" in model:
        headers["x-api-key"] = api_key  # Only this is different
    else:
        headers = super().validate_environment(...)  # Reuse parent
    return headers
```

## Future Enhancements

### Potential improvements

1. **More model type detection**
   - Support more providers (Cohere, AI21, v.v.)
   - Better model name parsing

2. **Caching**
   - Cache model type detection results
   - Reduce string parsing overhead

3. **Configuration**
   - Allow custom endpoint mapping
   - Support custom header templates

4. **Error handling**
   - Better error messages for auth failures
   - Retry logic với fallback

## Related Files

- `litellm/main.py` (line 2758-2808) - High-level routing
- `litellm/llms/agentrouter/chat/transformation.py` - Transformation logic
- `litellm/llms/openai/chat/gpt_transformation.py` - Parent class
- `litellm/llms/anthropic/chat/transformation.py` - Anthropic transformer
- `tests/test_litellm/test_agentrouter.py` - Unit tests
