# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
#
# Permission is granted to use the LLM4AD platform for research purposes.
# All publications, software, or other works that utilize this platform
# or any part of its codebase must acknowledge the use of "LLM4AD" and
# cite the following reference:
#
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang,
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
#
# For inquiries regarding commercial use or licensing, please contact
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------
from __future__ import annotations

import openai
from typing import Any

from llm4ad.base import LLM


class OpenAIAPI(LLM):
    def __init__(self, base_url: str, api_key: str, model: str, timeout=60, **kwargs):
        super().__init__()
        self._model = model
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

        # Pricing per 1M tokens (USD) - o3 models
        self._pricing = {
            'o3': {'input': 1.00, 'output': 4.00},
        }

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate API cost based on token usage and model pricing."""
        if self._model in self._pricing:
            pricing = self._pricing[self._model]
            input_cost = (prompt_tokens / 1_000_000) * pricing['input']
            output_cost = (completion_tokens / 1_000_000) * pricing['output']
            return input_cost + output_cost
        else:
            # Default pricing for o3-mini if model name doesn't match exactly
            input_cost = (prompt_tokens / 1_000_000) * 1.10
            output_cost = (completion_tokens / 1_000_000) * 4.40
            return input_cost + output_cost

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        if isinstance(prompt, str):
            prompt = [{'role': 'user', 'content': prompt.strip()}]
        response = self._client.chat.completions.create(
            model=self._model,
            messages=prompt,
            stream=False,
        )

        # Extract usage information
        if hasattr(response, 'usage') and response.usage:
            self.last_prompt_tokens = response.usage.prompt_tokens or 0
            self.last_completion_tokens = response.usage.completion_tokens or 0
            self.last_total_tokens = response.usage.total_tokens or 0

            # Calculate cost for this request
            self.last_api_cost = self._calculate_cost(self.last_prompt_tokens, self.last_completion_tokens)

            # Update cumulative totals
            self.total_prompt_tokens += self.last_prompt_tokens
            self.total_completion_tokens += self.last_completion_tokens
            self.total_tokens += self.last_total_tokens
            self.total_api_cost += self.last_api_cost

        return response.choices[0].message.content
