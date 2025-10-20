"""
AgentRouter Anthropic Config

Handles Claude models through AgentRouter.
Supports cross-format: can handle OpenAI Chat Completions API requests and convert them.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.base_utils import map_developer_role_to_system_role

from .format_converters import (
    convert_openai_messages_to_anthropic,
    convert_openai_tools_to_anthropic,
    convert_anthropic_response_to_openai,
)


AGENTROUTER_DEFAULT_BASE = "https://agentrouter.org/v1"


def _clean_model(model: str) -> str:
    """Remove provider prefix from model name."""
    if "/" in model:
        return model.split("/", 1)[-1]
    return model


class AgentrouterAnthropicConfig(AnthropicMessagesConfig):
    """
    Configuration for Claude models through AgentRouter.

    Supports:
    - Claude 3.5 Haiku
    - Claude Sonnet 4
    - Claude Sonnet 4.5
    - Other Claude models

    Also supports cross-format: can handle OpenAI Chat Completions API requests
    and automatically convert to Anthropic Messages format.
    """

    def __init__(self):
        """Initialize Anthropic config for AgentRouter."""
        super().__init__()
        self.base_config = AnthropicMessagesConfig()

    def _ensure_agentrouter_base(self, api_base: Optional[str]) -> str:
        """
        Ensure AgentRouter base URL is properly formatted for Anthropic.

        For Anthropic models, we need base WITHOUT /v1 suffix because
        Anthropic config will add /v1/messages.

        Args:
            api_base: Custom API base URL or None

        Returns:
            Properly formatted API base URL (without /v1)
        """
        if api_base is None:
            # Default base includes /v1, so remove it
            return AGENTROUTER_DEFAULT_BASE.rstrip("/v1")

        base = api_base.rstrip("/")

        # Remove /v1 suffix if present to let Anthropic config handle it
        if base.endswith("/v1"):
            return base[:-3]
        return base

    # ========== Anthropic Messages API Methods ==========

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get complete URL for Anthropic messages endpoint."""
        return self.base_config.get_complete_url(
            api_base=self._ensure_agentrouter_base(api_base),
            api_key=api_key,
            model=_clean_model(model),
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Validate environment and set up headers for Anthropic request."""
        headers, _ = self.validate_anthropic_messages_environment(
            headers=headers,
            model=_clean_model(model),
            messages=messages,  # type: ignore
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=self._ensure_agentrouter_base(api_base),
        )

        # Ensure User-Agent is set for Claude models
        if not any(h.lower() == "user-agent" for h in headers.keys()):
            headers["User-Agent"] = "claude-cli/2.0.15 (external, cli)"

        return headers

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        """Validate environment for Anthropic Messages API."""
        api_base = self._ensure_agentrouter_base(api_base)

        headers, api_base = self.base_config.validate_anthropic_messages_environment(  # type: ignore
            headers=headers,
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

        if not any(h.lower() == "user-agent" for h in headers.keys()):
            headers["User-Agent"] = "claude-cli/2.0.15 (external, cli)"

        return headers, api_base

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request for Anthropic messages.

        When using Claude via OpenAI /v1/chat/completions endpoint, messages may contain
        OpenAI-format images that need to be converted to Anthropic format.
        """
        # Convert OpenAI image format to Anthropic format if needed
        converted_messages = self._convert_openai_images_to_anthropic(messages)

        return self.transform_anthropic_messages_request(
            model=_clean_model(model),
            messages=converted_messages,  # type: ignore
            anthropic_messages_optional_request_params=optional_params,
            litellm_params=litellm_params,  # type: ignore
            headers=headers,
        )

    def _convert_openai_images_to_anthropic(self, messages: list) -> list:
        """
        Convert OpenAI image_url format to Anthropic image format.

        OpenAI: {"type": "image_url", "image_url": {"url": "..."}}
        Anthropic: {"type": "image", "source": {"type": "url", "url": "..."}}
        """
        converted = []
        for msg in messages:
            if not isinstance(msg, dict):
                converted.append(msg)
                continue

            content = msg.get("content")
            if not isinstance(content, list):
                # String content, no conversion needed
                converted.append(msg)
                continue

            # Convert content blocks
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue

                # Check if this is an OpenAI image_url block
                if block.get("type") == "image_url":
                    # Convert to Anthropic format
                    image_url_data = block.get("image_url", {})
                    url = image_url_data.get("url", "") if isinstance(image_url_data, dict) else ""

                    # Determine source type based on URL
                    if url.startswith("data:"):
                        # Base64 image
                        # Extract media type and base64 data
                        # Format: data:image/jpeg;base64,/9j/4AAQ...
                        media_type = "image/jpeg"  # default
                        data = url
                        if ";" in url and "," in url:
                            header, data = url.split(",", 1)
                            if ":" in header:
                                media_type = header.split(":")[1].split(";")[0]

                        new_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data
                            }
                        })
                    else:
                        # URL image
                        new_content.append({
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": url
                            }
                        })
                else:
                    # Not an image_url block, keep as-is
                    new_content.append(block)

            # Create new message with converted content
            new_msg = msg.copy()
            new_msg["content"] = new_content
            converted.append(new_msg)

        return converted

    def translate_developer_role_to_system_role(
        self,
        messages: List[Dict],
    ) -> List[Dict]:
        """Translate developer role to system role."""
        return map_developer_role_to_system_role(messages=messages)  # type: ignore

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params,
        headers: dict,
    ) -> Dict:
        """Transform request for Anthropic Messages API."""
        return self.base_config.transform_anthropic_messages_request(
            model=_clean_model(model),
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: Any,
        logging_obj: Any,
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> Any:
        """
        Transform response from Anthropic messages.

        Converts Anthropic response to OpenAI ModelResponse format for LiteLLM.
        """
        import json
        import time
        from litellm.types.utils import Message, Choices, Usage

        # Get Anthropic response (returns dict-like object)
        anthropic_response = self.transform_anthropic_messages_response(
            model=_clean_model(model),
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        # Convert to ModelResponse
        message_content = ""
        tool_calls = []

        content_blocks = anthropic_response.get("content") or []
        for content_block in content_blocks:
            # content_block can be Pydantic model or dict, handle both
            if isinstance(content_block, dict):
                block_type = content_block.get("type")
                block_text = content_block.get("text", "")
                block_id = content_block.get("id", "")
                block_name = content_block.get("name", "")
                block_input = content_block.get("input", {})
            else:
                # Pydantic model - access attributes directly
                block_type = getattr(content_block, "type", None)
                block_text = getattr(content_block, "text", "")
                block_id = getattr(content_block, "id", "")
                block_name = getattr(content_block, "name", "")
                block_input = getattr(content_block, "input", {})

            if block_type == "text":
                message_content += block_text
            elif block_type == "tool_use":
                from litellm.types.utils import ChatCompletionMessageToolCall, Function
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=block_id,
                        type="function",
                        function=Function(
                            name=block_name,
                            arguments=json.dumps(block_input)
                        )
                    )
                )

        message = Message(
            content=message_content or None,
            role="assistant",
            tool_calls=tool_calls if tool_calls else None,
        )

        choice = Choices(
            finish_reason=anthropic_response.get("stop_reason", "stop"),
            index=0,
            message=message,
        )

        usage_data = anthropic_response.get("usage") or {}
        usage = Usage(
            prompt_tokens=usage_data.get("input_tokens", 0),
            completion_tokens=usage_data.get("output_tokens", 0),
            total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
        )

        model_response.id = anthropic_response.get("id", "")
        model_response.choices = [choice]
        model_response.created = int(time.time())
        model_response.model = model
        model_response.object = "chat.completion"
        model_response.usage = usage

        return model_response

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ):
        """Transform response from Anthropic Messages API."""
        return self.base_config.transform_anthropic_messages_response(
            model=_clean_model(model),
            raw_response=raw_response,
            logging_obj=logging_obj
        )

    def get_model_response_iterator(
        self,
        streaming_response: Any,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        """Get model response iterator for sync streaming."""
        from litellm.llms.anthropic.chat.handler import ModelResponseIterator
        return ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: Any,
    ) -> AsyncIterator:
        """Get async streaming response iterator for Claude models."""
        return self.base_config.get_async_streaming_response_iterator(
            model=_clean_model(model),
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )

    def get_supported_anthropic_messages_params(self, model: str) -> list:
        """Get list of supported Anthropic parameters."""
        return self.base_config.get_supported_anthropic_messages_params(
            model=_clean_model(model)
        )

    # ========== Cross-Format Support: OpenAI → Anthropic ==========

    def transform_openai_chat_request(
        self,
        model: str,
        messages: List[Dict],
        optional_params: Dict,
        litellm_params,
        headers: dict,
    ) -> Dict:
        """
        Cross-format: Transform OpenAI Chat Completions request to Anthropic format.

        This allows users to call /v1/chat/completions endpoint with Claude models.
        Flow: OpenAI request → Anthropic request → AgentRouter /v1/messages

        Args:
            model: Model name
            messages: Messages in OpenAI format
            optional_params: OpenAI-specific params
            litellm_params: LiteLLM params
            headers: Request headers

        Returns:
            Request dict in Anthropic format
        """
        # Convert OpenAI messages to Anthropic messages format
        anthropic_messages = convert_openai_messages_to_anthropic(messages)

        # Convert OpenAI optional params to Anthropic optional params
        anthropic_params = optional_params.copy()

        # Map OpenAI-specific params to Anthropic equivalents
        if "stop" in anthropic_params:
            # OpenAI: stop, Anthropic: stop_sequences
            anthropic_params["stop_sequences"] = anthropic_params.pop("stop")

        if "tools" in anthropic_params:
            # Convert OpenAI tools to Anthropic tools format
            anthropic_params["tools"] = convert_openai_tools_to_anthropic(
                anthropic_params["tools"]
            )

        # Use Anthropic transformation
        return self.transform_anthropic_messages_request(
            model=model,
            messages=anthropic_messages,
            anthropic_messages_optional_request_params=anthropic_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_openai_chat_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: Any,
        logging_obj: Any,
    ) -> Dict:
        """
        Cross-format: Transform Anthropic response to OpenAI Chat Completions format.

        This converts the Anthropic response back to OpenAI format for the user.
        Flow: Anthropic response → OpenAI response

        Args:
            model: Model name
            raw_response: Raw HTTP response from AgentRouter
            model_response: Model response object
            logging_obj: Logging object

        Returns:
            Response dict in OpenAI format
        """
        # Get Anthropic response
        anthropic_response = self.transform_anthropic_messages_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        # Convert Anthropic response to OpenAI format
        # anthropic_response is AnthropicMessagesResponse (Pydantic), convert to dict
        if isinstance(anthropic_response, dict):
            response_dict = anthropic_response
        else:
            response_dict = anthropic_response.model_dump()  # type: ignore

        return convert_anthropic_response_to_openai(response_dict)  # type: ignore

    # ========== Additional Methods ==========

    @property
    def max_retry_on_unprocessable_entity_error(self) -> int:
        """Returns the max retry count for UnprocessableEntityError."""
        return 0

    @property
    def has_custom_stream_wrapper(self) -> bool:
        """Anthropic does not have custom stream wrapper."""
        return False

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        """Anthropic supports stream parameter in request body."""
        return True

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get AgentRouter API key from environment or parameter."""
        from litellm.secret_managers.main import get_secret_str
        return api_key or get_secret_str("AGENTROUTER_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """Get AgentRouter API base URL (without /v1 for Anthropic)."""
        from litellm.secret_managers.main import get_secret_str

        base = api_base or get_secret_str("AGENTROUTER_API_BASE") or AGENTROUTER_DEFAULT_BASE

        # Remove /v1 suffix for Anthropic (it adds /v1/messages)
        base = base.rstrip("/")
        if base.endswith("/v1"):
            return base[:-3]
        return base

    def __getattr__(self, name: str) -> Any:
        """
        Delegate any undefined attributes/methods to the base config.

        This ensures compatibility with all base class methods and properties
        without explicitly delegating each one.
        """
        return getattr(self.base_config, name)
