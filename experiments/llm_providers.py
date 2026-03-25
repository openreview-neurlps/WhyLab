# -*- coding: utf-8 -*-
"""
LLM Providers for Multi-Model Experiments
==========================================
Provider abstraction for Gemini, OpenAI, and Anthropic.
Each provider implements `call(system_prompt, user_prompt, **kwargs) -> LLMResponse`.
"""
import os
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    """Structured LLM response with metadata."""
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cache_hit: bool = False
    model: str = ""


class LLMProvider(Protocol):
    """Protocol for LLM providers."""
    model: str

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        ...


class GeminiProvider:
    """Google Gemini API provider."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        import google.generativeai as genai

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY or GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel(model)
        self._genai = genai

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
        gen_config = self._genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        seed = kwargs.get("seed", 0)
        try:
            gen_config.seed = seed
        except (AttributeError, TypeError):
            pass

        start = time.time()
        response = self._client.generate_content(full_prompt, generation_config=gen_config)
        latency_ms = (time.time() - start) * 1000

        text = response.text if response.text else ""
        pt = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
        ct = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0

        return LLMResponse(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency_ms, cache_hit=False, model=self.model,
        )


class OpenAIProvider:
    """OpenAI API provider (GPT-5 series)."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        seed = kwargs.get("seed", 0)
        start = time.time()

        # GPT-5+ uses max_completion_tokens; older models use max_tokens
        token_param = {}
        if "gpt-5" in self.model or "gpt-4.1" in self.model:
            token_param["max_completion_tokens"] = self.max_tokens
        else:
            token_param["max_tokens"] = self.max_tokens

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            seed=seed,
            **token_param,
        )
        latency_ms = (time.time() - start) * 1000

        text = response.choices[0].message.content or ""
        usage = response.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0

        return LLMResponse(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency_ms, cache_hit=False, model=self.model,
        )


class AnthropicProvider:
    """Anthropic Claude API provider."""

    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 2048):
        from anthropic import Anthropic

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self._client = Anthropic(api_key=api_key)

    def call(self, system_prompt: str, user_prompt: str, **kwargs) -> LLMResponse:
        start = time.time()
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system_prompt if system_prompt else "",
            messages=[{"role": "user", "content": user_prompt}],
        )
        latency_ms = (time.time() - start) * 1000

        text = response.content[0].text if response.content else ""
        pt = response.usage.input_tokens if response.usage else 0
        ct = response.usage.output_tokens if response.usage else 0

        return LLMResponse(
            text=text, prompt_tokens=pt, completion_tokens=ct,
            latency_ms=latency_ms, cache_hit=False, model=self.model,
        )


def create_provider(model: str, temperature: float = 0.0, max_tokens: int = 2048) -> LLMProvider:
    """Factory: create the right provider based on model name."""
    model_lower = model.lower()
    if "gemini" in model_lower:
        return GeminiProvider(model, temperature, max_tokens)
    elif "gpt" in model_lower or "o1" in model_lower or "o3" in model_lower or "o4" in model_lower:
        return OpenAIProvider(model, temperature, max_tokens)
    elif "claude" in model_lower:
        return AnthropicProvider(model, temperature, max_tokens)
    else:
        raise ValueError(f"Unknown model family: {model}. Prefix should contain gemini/gpt/claude.")
