"""
AgentRouter provider – Unified transformation config

Single class that dispatches to the correct, already-supported provider
transformation based on the model. Supports both OpenAI-compatible
Chat Completions and Anthropic Messages (native) flows.
"""

from __future__ import annotations

import json
import time
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.xai.chat.transformation import XAIChatConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


AGENTROUTER_DEFAULT_BASE = "https://agentrouter.org/v1"


def _clean_model(model: str) -> str:
    if "/" in model:
        return model.split("/", 1)[-1]
    return model


class ModelType(Enum):
    OPENAI = "gpt"
    ANTHROPIC = "claude"
    DEEPSEEK = "deepseek"
    XAI = "grok"


class AgentrouterConfig(AnthropicMessagesConfig, OpenAIGPTConfig):

    def __init__(self, model: str):
        super().__init__()

        model = _clean_model(model)
        self.config: Union[AnthropicMessagesConfig, BaseConfig]
        self.model_type: ModelType
        if "claude" in model:
            self.config = AnthropicMessagesConfig()
            self.model_type = ModelType.ANTHROPIC
        elif "deepseek" in model:
            self.config  = DeepSeekChatConfig()
            self.model_type = ModelType.DEEPSEEK
        elif "xai" in model or "grok" in model:
            self.config  = XAIChatConfig()
            self.model_type = ModelType.XAI
        else:
            self.config  = OpenAIGPTConfig()
            self.model_type = ModelType.OPENAI

    def _ensure_agentrouter_base(self, api_base: Optional[str]) -> str:
        """
        Ensure AgentRouter base URL is properly formatted.
        Handles both default and custom API bases.
        """
        # Use provided base or default constant (which includes /v1)
        if api_base is None:
            base = AGENTROUTER_DEFAULT_BASE
        else:
            base = api_base.rstrip("/")

        # For Anthropic models, we need base without /v1 suffix
        # because Anthropic config will add /v1/messages
        if self.model_type == ModelType.ANTHROPIC:
            # Remove /v1 suffix if present to let Anthropic config handle it
            if base.endswith("/v1"):
                return base[:-3]
            return base

        # For OpenAI-compatible models, ensure /v1 is present
        # If base doesn't have /v1, add it
        if not base.endswith("/v1"):
            return f"{base}/v1"
        return base

    # ---------- Chat (OpenAI-compatible) ----------
    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return self.config.get_complete_url(
            api_base=self._ensure_agentrouter_base(api_base),
            api_key=api_key,
            model=_clean_model(model),
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def _ensure_openai_base(self) -> BaseConfig:
        if not isinstance(self.config, BaseConfig):
            raise ValueError("Model is not an OpenAI-compatible model; cannot use OpenAI-compatible flow.")
        return self.config  # type: ignore

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # For Claude models, use Anthropic messages flow
        if self.model_type == ModelType.ANTHROPIC:
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
        # For non-Claude models, use OpenAI flow
        headers = self._ensure_openai_base().validate_environment(
            headers=headers,
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=self._ensure_agentrouter_base(api_base),
        )
        return headers

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # For Claude models, use Anthropic messages transform
        if self.model_type == ModelType.ANTHROPIC:
            return self.transform_anthropic_messages_request(
                model=_clean_model(model),
                messages=messages,  # type: ignore
                anthropic_messages_optional_request_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        # For non-Claude models, use OpenAI transform
        return self._ensure_openai_base().transform_request(
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ):
        return self.config.sign_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=self._ensure_agentrouter_base(api_base),
            api_key=api_key,
            model=_clean_model(model) if model else None,
            stream=stream,
            fake_stream=fake_stream,
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
        # For Claude models, convert Anthropic response to ModelResponse
        if self.model_type == ModelType.ANTHROPIC:
            from litellm.types.utils import (
                Message,
                Choices,
                Usage,
            )

            # Get Anthropic response (returns dict-like object)
            anthropic_response = self.transform_anthropic_messages_response(
                model=_clean_model(model),
                raw_response=raw_response,
                logging_obj=logging_obj,
            )

            # Convert to ModelResponse
            message_content = ""
            tool_calls = []

            # anthropic_response is a TypedDict-like object, access with dict syntax
            content_blocks = anthropic_response.get("content") or []
            for content_block in content_blocks:
                if content_block.get("type") == "text":
                    message_content += content_block.get("text", "")
                elif content_block.get("type") == "tool_use":
                    from litellm.types.utils import ChatCompletionMessageToolCall, Function
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=content_block.get("id", ""),
                            type="function",
                            function=Function(
                                name=content_block.get("name", ""),
                                arguments=json.dumps(content_block.get("input", {}))
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

        # For non-Claude models, use OpenAI transform
        return self._ensure_openai_base().transform_response(
            model=_clean_model(model),
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        # For Claude models, Anthropic uses async streaming response iterator
        # This method is primarily for OpenAI-style sync streaming
        if self.model_type == ModelType.ANTHROPIC:
            # For sync streaming with Claude, we need to handle it differently
            # Return a wrapper that can handle Anthropic streaming format
            from litellm.llms.anthropic.chat.handler import ModelResponseIterator
            return ModelResponseIterator(
                streaming_response=streaming_response,
                sync_stream=sync_stream,
                json_mode=json_mode,
            )
        # For non-Claude models, use OpenAI iterator
        return self._ensure_openai_base().get_model_response_iterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

    # Override for async streaming with Claude models
    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: Any,
    ) -> AsyncIterator:
        """
        For Claude models, return Anthropic streaming iterator.
        For other models, delegate to base OpenAI implementation.
        """
        if self.model_type == ModelType.ANTHROPIC:
            return self._ensure_anthropic_messages_base().get_async_streaming_response_iterator(
                model=_clean_model(model),
                httpx_response=httpx_response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
            )
        # For non-Claude models, use base implementation if available
        raise NotImplementedError("Async streaming not implemented for non-Claude AgentRouter models")

    # ---------- Anthropic Messages (native) ----------
    def _ensure_anthropic_messages_base(self) -> AnthropicMessagesConfig:
        if not isinstance(self.config, AnthropicMessagesConfig):
            raise ValueError("Model is not an Anthropic model; cannot use Anthropic messages flow.")
        return self.config  # type: ignore

    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return self._ensure_anthropic_messages_base().get_supported_anthropic_messages_params(model)

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
        api_base = self._ensure_agentrouter_base(api_base)
        headers, api_base = self._ensure_anthropic_messages_base().validate_anthropic_messages_environment( #type: ignore
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

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params,
        headers: dict,
    ) -> Dict:
        return self._ensure_anthropic_messages_base().transform_anthropic_messages_request(
            model=_clean_model(model),
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ):
        return self._ensure_anthropic_messages_base().transform_anthropic_messages_response(
            model=_clean_model(model), raw_response=raw_response, logging_obj=logging_obj
        )
