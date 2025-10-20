"""
Format converters for AgentRouter cross-format support.

Simple wrapper utilities for converting between Anthropic and OpenAI formats.
These are lightweight converters specific to AgentRouter's needs.
"""

from typing import Any, Dict, List
import json
import time


def convert_anthropic_messages_to_openai(anthropic_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Anthropic messages format to OpenAI messages format.
    
    Anthropic: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
    OpenAI: {"role": "user", "content": "Hello"}
    """
    openai_messages = []
    
    for msg in anthropic_messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            
            if text_parts:
                openai_messages.append({"role": role, "content": "".join(text_parts)})
            continue
        
        openai_messages.append({"role": role, "content": content})
    
    return openai_messages


def convert_anthropic_tools_to_openai(anthropic_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Anthropic tools format to OpenAI tools format.
    
    Anthropic: {"name": "...", "input_schema": {...}}
    OpenAI: {"type": "function", "function": {"name": "...", "parameters": {...}}}
    """
    openai_tools = []
    
    for tool in anthropic_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {})
            }
        })
    
    return openai_tools


def convert_openai_response_to_anthropic(openai_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert OpenAI response to Anthropic Messages format.
    
    Maps: choices → content, finish_reason → stop_reason, usage tokens
    """
    choice = openai_response.get("choices", [{}])[0]
    message = choice.get("message", {})
    content = message.get("content", "")
    tool_calls = message.get("tool_calls", [])
    finish_reason = choice.get("finish_reason", "stop")
    usage = openai_response.get("usage", {})
    
    anthropic_content = []
    
    if content:
        anthropic_content.append({
            "type": "text",
            "text": content
        })
    
    for tool_call in tool_calls:
        anthropic_content.append({
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": tool_call.get("function", {}).get("name", ""),
            "input": json.loads(tool_call.get("function", {}).get("arguments", "{}"))
        })
    
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")
    
    return {
        "id": openai_response.get("id", "msg_" + str(int(time.time()))),
        "type": "message",
        "role": "assistant",
        "content": anthropic_content,
        "model": openai_response.get("model", ""),
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
    }


def convert_openai_messages_to_anthropic(openai_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI string content to Anthropic content blocks."""
    anthropic_messages = []
    
    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if isinstance(content, str):
            anthropic_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}]
            })
        else:
            anthropic_messages.append(msg)
    
    return anthropic_messages


def convert_openai_tools_to_anthropic(openai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tools to Anthropic tools format."""
    anthropic_tools = []
    
    for tool in openai_tools:
        func = tool.get("function", {})
        anthropic_tools.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {})
        })
    
    return anthropic_tools


def convert_anthropic_response_to_openai(anthropic_response: Dict[str, Any]) -> Dict[str, Any]:
    """Convert Anthropic response to OpenAI format."""
    content_blocks = anthropic_response.get("content", [])
    
    text_content = ""
    tool_calls = []
    
    for block in content_blocks:
        if block.get("type") == "text":
            text_content = block.get("text", "")
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}))
                }
            })
    
    finish_reason_map = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    finish_reason = finish_reason_map.get(
        anthropic_response.get("stop_reason", "end_turn"),
        "stop"
    )
    
    usage = anthropic_response.get("usage", {})
    
    return {
        "id": anthropic_response.get("id", "chatcmpl_" + str(int(time.time()))),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_response.get("model", ""),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls if tool_calls else None
            },
            "finish_reason": finish_reason
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        }
    }
