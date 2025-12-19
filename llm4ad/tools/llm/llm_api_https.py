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

import http.client
import json
import time
from typing import Any
import traceback
from ...base import LLM


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=60, **kwargs):
        """Https API
        Args:
            host   : host name. please note that the host name does not include 'https://'
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
        """
        super().__init__(**kwargs)
        self._host = host
        self._key = key
        self._model = model
        self._timeout = timeout
        self._kwargs = kwargs
        self._cumulative_error = 0

        # Pricing per 1M tokens (USD) - o3 models
        self._pricing = {
            'o3-mini': {'input': 1.10, 'output': 4.40},
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

        while True:
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 4096),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': prompt
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', '/v1/chat/completions', payload, headers)
                res = conn.getresponse()
                data = res.read().decode('utf-8')
                data = json.loads(data)

                # Extract usage information
                if 'usage' in data:
                    usage = data['usage']
                    self.last_prompt_tokens = usage.get('prompt_tokens', 0)
                    self.last_completion_tokens = usage.get('completion_tokens', 0)
                    self.last_total_tokens = usage.get('total_tokens', 0)

                    # Calculate cost for this request
                    self.last_api_cost = self._calculate_cost(self.last_prompt_tokens, self.last_completion_tokens)

                    # Update cumulative totals
                    self.total_prompt_tokens += self.last_prompt_tokens
                    self.total_completion_tokens += self.last_completion_tokens
                    self.total_tokens += self.last_total_tokens
                    self.total_api_cost += self.last_api_cost

                response = data['choices'][0]['message']['content']
                if self.debug_mode:
                    self._cumulative_error = 0
                return response
            except Exception as e:
                self._cumulative_error += 1
                if self.debug_mode:
                    if self._cumulative_error == 10:
                        raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                                           f'You may check your API host and API key.')
                else:
                    print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
                          f'You may check your API host and API key.')
                    time.sleep(2)
                continue
