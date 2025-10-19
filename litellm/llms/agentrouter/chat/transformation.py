"""
AgentRouter provider – Unified transformation config

Dispatcher that routes to the appropriate specialized config based on model type:
- Claude models → AgentrouterAnthropicConfig
- OpenAI-compatible models → AgentrouterOpenAIConfig

Both configs support cross-format conversions.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import httpx

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import AnthropicStreamWrapper

from .anthropic_config import AgentrouterAnthropicConfig
from .openai_config import AgentrouterOpenAIConfig


def _clean_model(model: str) -> str:
    """Remove provider prefix from model name."""
    if "/" in model:
        return model.split("/", 1)[-1]
    return model


class AgentrouterConfig:
    """
    Dispatcher config for AgentRouter provider.

    Routes requests to the appropriate specialized config based on model type:
    - Claude models (claude-*) → AgentrouterAnthropicConfig
    - OpenAI-compatible models (gpt-*, deepseek-*, grok-*) → AgentrouterOpenAIConfig

    Both configs support cross-format conversions, allowing flexible API usage.
    """

    def __init__(self, model: str):
        """
        Initialize AgentRouter config dispatcher.

        Args:
            model: Model name (with or without provider prefix)
        """
        clean_model_name = _clean_model(model)

        # Detect model type and instantiate appropriate config
        if "claude" in clean_model_name:
            self._config: Union[AgentrouterAnthropicConfig, AgentrouterOpenAIConfig] = AgentrouterAnthropicConfig()
            self._is_anthropic = True
        else:
            # OpenAI-compatible models (GPT, DeepSeek, Grok, etc.)
            self._config = AgentrouterOpenAIConfig(clean_model_name)
            self._is_anthropic = False

    @classmethod
    def get_config(cls):
        """Get config dictionary (for provider-specific config overrides)."""
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
            and not isinstance(
                v,
                (
                    staticmethod,
                    classmethod,
                    property,
                    type(lambda: None),
                ),
            )
        }

    @property
    def is_anthropic_model(self) -> bool:
        """Check if the model is a Claude/Anthropic model."""
        return self._is_anthropic

    @property
    def is_openai_compatible_model(self) -> bool:
        """Check if the model is OpenAI-compatible."""
        return not self._is_anthropic

    # ========== Delegate all methods to the appropriate config ==========

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get complete URL for the request."""
        return self._config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

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
        """Validate environment and setup headers."""
        return self._config.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Transform request."""
        return self._config.transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def translate_developer_role_to_system_role(
        self,
        messages: List[AllMessageValues],
    ) -> List[AllMessageValues]:
        """Translate developer role to system role."""
        result = self._config.translate_developer_role_to_system_role(messages=messages)  # type: ignore
        return result  # type: ignore

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
        """Sign request (OpenAI-compatible models only)."""
        if self.is_openai_compatible_model:
            # Ensure api_base is not None for sign_request
            base_url = api_base or "https://agentrouter.org/v1"
            return self._config.base_config.sign_request(
                headers=headers,
                optional_params=optional_params,
                request_data=request_data,
                api_base=base_url,
                api_key=api_key,
                model=_clean_model(model) if model else None,
                stream=stream,
                fake_stream=fake_stream,
            )
        # Anthropic doesn't need signing, return headers unchanged
        return headers, None

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
        """Transform response."""
        return self._config.transform_response(
            model=model,
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
        """Get model response iterator for streaming."""
        return self._config.get_model_response_iterator(
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
        """
        Get async streaming response iterator.

        For Claude models: Use native Anthropic streaming
        For OpenAI models: Use OpenAI streaming wrapped in Anthropic format (cross-format)
        """
        if self.is_anthropic_model:
            return self._config.get_async_streaming_response_iterator(
                model=model,
                httpx_response=httpx_response,
                request_body=request_body,
                litellm_logging_obj=litellm_logging_obj,
            )

        # For OpenAI models via /v1/messages (cross-format):
        # 1. Get OpenAI streaming iterator from httpx response
        # 2. Filter out invalid chunks (empty choices)
        # 3. Wrap it in Anthropic format
        config = self._config
        openai_stream = config.base_config.get_model_response_iterator(  # type: ignore
                streaming_response=httpx_response.aiter_lines(),
                sync_stream=False,
                json_mode=False,
        )

        # Filter out invalid chunks before wrapping
        async def filtered_stream():
            """Filter out chunks with empty choices array to prevent IndexError."""
            from litellm import verbose_logger
            chunk_count = 0
            yielded_count = 0
            async for chunk in openai_stream:
                chunk_count += 1
                # Debug: log chunk details
                verbose_logger.debug(
                    f"AgentRouter OpenAI→Anthropic: Chunk #{chunk_count}, "
                    f"has_choices={hasattr(chunk, 'choices')}, "
                    f"choices_len={len(chunk.choices) if hasattr(chunk, 'choices') else 'N/A'}, "
                    f"chunk_type={type(chunk).__name__}"
                )

                # Skip chunks with no choices or empty choices array
                if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    yielded_count += 1
                    verbose_logger.debug(
                        f"AgentRouter: Yielding chunk #{yielded_count}, "
                        f"finish_reason={chunk.choices[0].finish_reason if chunk.choices else None}"
                    )
                    yield chunk
                else:
                    verbose_logger.debug(f"AgentRouter: Skipping chunk #{chunk_count} - no choices")

            verbose_logger.info(
                f"AgentRouter stream finished: {yielded_count}/{chunk_count} chunks yielded"
            )

        # Wrap filtered OpenAI stream in Anthropic format for /v1/messages compatibility
        return AnthropicStreamWrapper(
            completion_stream=filtered_stream(),
            model=_clean_model(model)
        ).async_anthropic_sse_wrapper()

    # ========== Anthropic Messages API Support ==========
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        """Get list of supported parameters."""
        if self.is_anthropic_model:
            return self._config.get_supported_anthropic_messages_params(model)
        # For OpenAI models, return OpenAI params (used for cross-format)
        # Type narrowing: we know _config is AgentrouterOpenAIConfig here
        return self._config.get_supported_openai_params(_clean_model(model))

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
        if self.is_anthropic_model:
            return self._config.validate_anthropic_messages_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
                api_base=api_base,
            )
        else:
            # OpenAI model via Anthropic endpoint (cross-format)
            headers = self._config.validate_environment(
                headers=headers,
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                api_key=api_key,
                api_base=api_base,
            )
            return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params,
        headers: dict,
    ) -> Dict:
        """
        Transform request for Anthropic Messages API.

        For Claude models: Native Anthropic transformation
        For OpenAI models: Cross-format conversion (Anthropic → OpenAI)
        """
        return self._config.transform_anthropic_messages_request(
            model=model,
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
        """
        Transform response from Anthropic Messages API.

        For Claude models: Native Anthropic transformation
        For OpenAI models: Cross-format conversion (OpenAI → Anthropic)
        """
        return self._config.transform_anthropic_messages_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

    # ========== Additional Methods ==========

    async def async_transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Async version of transform_request."""
        if self.is_anthropic_model:
            return self._config.transform_request(
                model=model,
                messages=messages,
                optional_params=optional_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        # For OpenAI models, check if async version exists
        return await self._config.async_transform_request(  # type: ignore
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get AgentRouter API key."""
        from litellm.secret_managers.main import get_secret_str
        return api_key or get_secret_str("AGENTROUTER_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """Get AgentRouter API base URL."""
        if api_base:
            return api_base
        from litellm.secret_managers.main import get_secret_str
        return get_secret_str("AGENTROUTER_API_BASE") or "https://agentrouter.org/v1"

    def should_fake_stream(
        self,
        model: str,
        stream: Optional[bool],
        custom_llm_provider: str,
    ) -> bool:
        """Determine if we should fake streaming."""
        # Neither Anthropic nor OpenAI need fake streaming for AgentRouter
        return False

    def get_supported_openai_params(self, model: str) -> list:
        """Get list of supported OpenAI parameters."""
        if self.is_anthropic_model:
            return self._config.get_supported_anthropic_messages_params(model)
        config = self._config
        if hasattr(config, 'base_config') and hasattr(config.base_config, 'get_supported_openai_params'):
            return config.base_config.get_supported_openai_params(_clean_model(model))  # type: ignore
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        """Map OpenAI parameters to provider-specific parameters."""
        config = self._config
        if self.is_anthropic_model:
            if hasattr(config, 'base_config') and hasattr(config.base_config, 'map_openai_params'):
                return config.base_config.map_openai_params(  # type: ignore
                    non_default_params=non_default_params,
                    optional_params=optional_params,
                    model=_clean_model(model),
                    drop_params=drop_params,
                )
            return optional_params
        if hasattr(config, 'base_config') and hasattr(config.base_config, 'map_openai_params'):
            return config.base_config.map_openai_params(  # type: ignore
                non_default_params=non_default_params,
                optional_params=optional_params,
                model=_clean_model(model),
                drop_params=drop_params,
            )
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict[str, Any], httpx.Headers]
    ) -> Any:
        """Get the appropriate error class for the error response."""
        # Both configs use similar error handling
        if self.is_anthropic_model:
            if hasattr(self._config.base_config, 'get_error_class'):
                return self._config.base_config.get_error_class(
                    error_message=error_message,
                    status_code=status_code,
                    headers=headers,
                )
        return self._config.base_config.get_error_class(
            error_message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def should_retry_llm_api_inside_llm_translation_on_http_error(
        self, e: httpx.HTTPStatusError, litellm_params: dict
    ) -> bool:
        """
        Returns True if the model/provider should retry the LLM API on UnprocessableEntityError.

        For both Anthropic and OpenAI models, delegate to base config if available.
        """
        config = self._config
        if hasattr(config, 'base_config') and hasattr(
            config.base_config, 'should_retry_llm_api_inside_llm_translation_on_http_error'
        ):
            return config.base_config.should_retry_llm_api_inside_llm_translation_on_http_error(  # type: ignore
                e=e, litellm_params=litellm_params
            )
        # Default: don't retry
        return False

    def transform_request_on_unprocessable_entity_error(
        self, e: httpx.HTTPStatusError, request_data: dict
    ) -> dict:
        """Transform the request data on UnprocessableEntityError."""
        config = self._config
        if hasattr(config, 'base_config') and hasattr(
            config.base_config, 'transform_request_on_unprocessable_entity_error'
        ):
            return config.base_config.transform_request_on_unprocessable_entity_error(  # type: ignore
                e=e, request_data=request_data
            )
        # Default: return unchanged
        return request_data

    @property
    def max_retry_on_unprocessable_entity_error(self) -> int:
        """
        Returns the max retry count for UnprocessableEntityError.

        Used if `should_retry_llm_api_inside_llm_translation_on_http_error` is True.
        """
        config = self._config
        if hasattr(config, 'base_config') and hasattr(
            config.base_config, 'max_retry_on_unprocessable_entity_error'
        ):
            return config.base_config.max_retry_on_unprocessable_entity_error  # type: ignore
        # Default: no retry
        return 0

    def __getattr__(self, name: str) -> Any:
        """
        Delegate any undefined attributes/methods to the underlying config.

        This ensures compatibility with all base class methods and properties
        without explicitly delegating each one.
        """
        return getattr(self._config, name)
