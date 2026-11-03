# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generation against official vLLM servers started by `external_launcher`, outside the Ray job.

`HttpServerHandle` stands in for a `vLLMHttpServer` actor handle: ``await handle.generate.remote(...)``
calls vLLM's token-in, token-out route and returns verl's `TokenOutput`, so the load balancer and
`FullyAsyncLLMServerClient` (including resubmitting aborted requests) work unchanged.
"""

import logging
import os
from typing import Any

import aiohttp

from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def to_vllm_sampling_params(
    sampling_params: dict[str, Any], prompt_len: int, prompt_length: int, response_length: int
) -> dict[str, Any]:
    """verl's sampling params as vLLM's, following `vLLMHttpServer.generate`."""
    params = dict(sampling_params)
    max_possible_tokens = prompt_length + response_length - prompt_len
    if "max_tokens" in params:
        max_tokens = params.pop("max_tokens")
    elif "max_new_tokens" in params:
        max_tokens = params.pop("max_new_tokens")
    else:
        max_tokens = min(response_length, max_possible_tokens)
    params["max_tokens"] = max(1, min(max_tokens, max_possible_tokens))
    topk_log_probs = params.pop("topk_log_probs", 0)
    if params.pop("logprobs", False):
        params["logprobs"] = topk_log_probs or 0
    return params


def to_token_output(choice: dict[str, Any]) -> TokenOutput:
    """vLLM's generate choice as verl's `TokenOutput`."""
    finish_reason = choice.get("finish_reason")
    if finish_reason == "abort":
        stop_reason = "aborted"
    elif finish_reason in ("stop", "length"):
        stop_reason = "completed"
    else:
        stop_reason = finish_reason
    content = (choice.get("logprobs") or {}).get("content")
    log_probs = [entry["logprob"] for entry in content] if content else None
    extra_fields: dict[str, Any] = {}
    spans = choice.get("weight_versions") or []
    versions = [int(span["version"]) for span in spans if str(span.get("version", "")).lstrip("-").isdigit()]
    if versions:
        # The version that produced the last token, and the range the output spans.
        extra_fields["global_steps"] = versions[-1]
        extra_fields["min_global_steps"] = min(versions)
        extra_fields["max_global_steps"] = max(versions)
    return TokenOutput(
        token_ids=choice.get("token_ids") or [],
        log_probs=log_probs,
        stop_reason=stop_reason,
        extra_fields=extra_fields,
    )


class _Generate:
    def __init__(self, handle: "HttpServerHandle"):
        self.handle = handle

    async def remote(self, **kwargs) -> TokenOutput:
        return await self.handle.generate_async(**kwargs)


class HttpServerHandle:
    """Picklable stand-in for a vLLM server actor handle, for an external vLLM server.

    Args:
        url: Base URL of the vLLM server.
        prompt_length: Rollout prompt length, to bound max_tokens as the in-job server does.
        response_length: Rollout response length; the default max_tokens.
        timeout: Seconds per request.
    """

    def __init__(self, url: str, prompt_length: int, response_length: int, timeout: float = 3600.0):
        self.url = url.rstrip("/")
        self.prompt_length = prompt_length
        self.response_length = response_length
        self.timeout = timeout

    @property
    def generate(self) -> _Generate:
        return _Generate(self)

    async def generate_async(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Any = None,
        video_data: Any = None,
        audio_data: Any = None,
        priority: int = 0,
        **_ignored,
    ) -> TokenOutput:
        if image_data is not None or video_data is not None or audio_data is not None:
            raise NotImplementedError("External vLLM rollout does not forward multimodal inputs yet")
        body = {
            "request_id": request_id,
            "token_ids": list(prompt_ids),
            "sampling_params": to_vllm_sampling_params(
                sampling_params, len(prompt_ids), self.prompt_length, self.response_length
            ),
        }
        if priority:
            body["priority"] = priority
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            async with session.post(f"{self.url}/inference/v1/generate", json=body) as response:
                if response.status >= 400:
                    raise RuntimeError(f"vLLM at {self.url} returned {response.status}: {await response.text()}")
                return to_token_output((await response.json())["choices"][0])


class ExternalServerManager:
    """The parts of `LLMServerManager` the trainer uses, over external vLLM servers.

    Args:
        config: Trainer config.
        server_urls: Base URLs of the vLLM servers.
    """

    def __init__(self, config, server_urls: list[str]):
        from verl.workers.rollout.router import get_router_handle

        rollout = config.actor_rollout_ref.rollout
        self.config = config
        self.server_addresses = list(server_urls)
        handles = [
            HttpServerHandle(url, prompt_length=rollout.prompt_length, response_length=rollout.response_length)
            for url in self.server_addresses
        ]
        self.global_load_balancer = get_router_handle(
            servers=dict(zip(self.server_addresses, handles, strict=True)),
            router_config_path=getattr(rollout, "router_config_path", None),
            full_determinism=getattr(rollout, "full_determinism", False),
        )

    def get_client(self, client_cls=None, **kwargs):
        from verl.workers.rollout.llm_server import LLMServerClient

        client_cls = client_cls or LLMServerClient
        return client_cls(config=self.config, load_balancer_handle=self.global_load_balancer, **kwargs)

    def get_addresses(self) -> list[str]:
        return self.server_addresses

    def get_replicas(self) -> list:
        # The servers are not verl replicas: verl neither launches nor controls them.
        return []

    def start_profile(self, **kwargs):
        logger.warning("Profiling is not supported for external vLLM servers")

    def stop_profile(self):
        pass
