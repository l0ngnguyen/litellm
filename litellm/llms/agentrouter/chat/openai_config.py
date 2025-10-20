"""
AgentRouter OpenAI Config

Handles OpenAI-compatible models (GPT, DeepSeek, XAI/Grok, etc.) through AgentRouter.
Supports cross-format: can handle Anthropic Messages API requests and convert them.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.xai.chat.transformation import XAIChatConfig

from .format_converters import (
    convert_anthropic_messages_to_openai,
    convert_anthropic_tools_to_openai,
    convert_openai_response_to_anthropic,
)


AGENTROUTER_DEFAULT_BASE = "https://agentrouter.org/v1"


def _clean_model(model: str) -> str:
    """Remove provider prefix from model name."""
    if "/" in model:
        return model.split("/", 1)[-1]
    return model


class OpenAIModelTypes(Enum):
    """Enum for OpenAI-compatible model types supported by AgentRouter."""
    OPENAI = "gpt"
    DEEPSEEK = "deepseek"
    XAI = "grok"


class AgentrouterOpenAIConfig(OpenAIGPTConfig):
    """
    Configuration for OpenAI-compatible models through AgentRouter.

    Supports:
    - GPT models (gpt-4, gpt-5, etc.)
    - DeepSeek models (deepseek-v3.1, deepseek-r1, etc.)
    - XAI models (grok-2, grok-code-fast-1, etc.)
    - Other OpenAI-compatible models

    Also supports cross-format: can handle Anthropic Messages API requests
    and automatically convert to OpenAI format.
    """

    def __init__(self, model: str):
        """
        Initialize OpenAI config for AgentRouter.

        Args:
            model: Model name (with or without provider prefix)
        """
        super().__init__()

        model = _clean_model(model)
        self.model_type: OpenAIModelTypes

        # Detect model type and use appropriate config
        if "deepseek" in model:
            self.base_config: BaseConfig = DeepSeekChatConfig()
            self.model_type = OpenAIModelTypes.DEEPSEEK
        elif "xai" in model or "grok" in model:
            self.base_config = XAIChatConfig()
            self.model_type = OpenAIModelTypes.XAI
        else:
            self.base_config = OpenAIGPTConfig()
            self.model_type = OpenAIModelTypes.OPENAI

    def _ensure_agentrouter_base(self, api_base: Optional[str]) -> str:
        """
        Ensure AgentRouter base URL is properly formatted with /v1 suffix.

        Args:
            api_base: Custom API base URL or None

        Returns:
            Properly formatted API base URL
        """
        if api_base is None:
            return AGENTROUTER_DEFAULT_BASE

        base = api_base.rstrip("/")

        # Ensure /v1 suffix for OpenAI-compatible endpoint
        if not base.endswith("/v1"):
            return f"{base}/v1"
        return base

    # ========== OpenAI Chat Completions Methods ==========

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """Get complete URL for OpenAI chat completions endpoint."""
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
        """Validate environment and set up headers for OpenAI request."""
        headers = self.base_config.validate_environment(
            headers=headers,
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=self._ensure_agentrouter_base(api_base),
        )

        # Add AgentRouter-specific headers
        if not any(h.lower() == "user-agent" for h in headers.keys()):
            headers["user-agent"] = "Kilo-iCode/4.107.0"
            headers["x-title"] = "Kilo Code"
            headers["x-kilocode-version"] = "4.107.0"

        return headers

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """Transform request for OpenAI chat completions."""
        return self.base_config.transform_request(
            model=_clean_model(model),
            messages=messages,
            optional_params=optional_params,
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
        """Transform response from OpenAI chat completions."""
        return self.base_config.transform_response(
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

    # ========== Cross-Format Support: Anthropic → OpenAI ==========

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params,
        headers: dict,
    ) -> Dict:
        """
        Cross-format: Transform Anthropic Messages API request to OpenAI format.

        This allows users to call /v1/messages endpoint with OpenAI models.
        Flow: Anthropic request → OpenAI request → AgentRouter /v1/chat/completions

        Args:
            model: Model name
            messages: Messages in Anthropic format
            anthropic_messages_optional_request_params: Anthropic-specific params
            litellm_params: LiteLLM params
            headers: Request headers

        Returns:
            Request dict in OpenAI format
        """
        # Convert Anthropic messages to OpenAI messages format
        openai_messages = convert_anthropic_messages_to_openai(messages)

        # Convert Anthropic optional params to OpenAI optional params
        optional_params = anthropic_messages_optional_request_params.copy()

        # Map Anthropic-specific params to OpenAI equivalents
        if "stop_sequences" in optional_params:
            # Anthropic: stop_sequences, OpenAI: stop
            optional_params["stop"] = optional_params.pop("stop_sequences")

        if "tools" in optional_params:
            # Convert Anthropic tools to OpenAI tools format
            optional_params["tools"] = convert_anthropic_tools_to_openai(
                optional_params["tools"]
            )

        # Use OpenAI transformation
        return self.transform_request(
            model=model,
            messages=openai_messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> Dict:
        """
        Cross-format: Transform OpenAI response to Anthropic Messages format.

        This converts the OpenAI response back to Anthropic format for the user.
        Flow: OpenAI response → Anthropic response

        Args:
            model: Model name
            raw_response: Raw HTTP response from AgentRouter
            logging_obj: Logging object

        Returns:
            Response dict in Anthropic format
        """
        # Parse OpenAI response
        try:
            response_json = raw_response.json()
        except Exception:
            response_json = {}

        # Convert OpenAI response to Anthropic Messages format
        return convert_openai_response_to_anthropic(response_json)

    # ========== Additional Methods ==========

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """Get AgentRouter API key from environment or parameter."""
        from litellm.secret_managers.main import get_secret_str
        return api_key or get_secret_str("AGENTROUTER_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> str:
        """Get AgentRouter API base URL."""
        from litellm.secret_managers.main import get_secret_str

        base = api_base or get_secret_str("AGENTROUTER_API_BASE") or AGENTROUTER_DEFAULT_BASE

        # Ensure /v1 suffix
        if not base.endswith("/v1"):
            return f"{base.rstrip('/')}/v1"
        return base

    def __getattr__(self, name: str) -> Any:
        """
        Delegate any undefined attributes/methods to the base config.

        This ensures compatibility with all base class methods and properties
        without explicitly delegating each one.
        """
        return getattr(self.base_config, name)
