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
"""HttpServerHandle against a fake vLLM token-in, token-out server."""

import asyncio
import pickle
import socket
import threading
import time

import pytest

from verl.workers.rollout.external_server import (
    HttpServerHandle,
    to_token_output,
    to_vllm_sampling_params,
)


def test_sampling_params_follow_the_in_job_server():
    sampling = {"temperature": 1.0, "logprobs": True, "topk_log_probs": 0}
    params = to_vllm_sampling_params(sampling, prompt_len=100, prompt_length=512, response_length=256)
    assert params == {"temperature": 1.0, "logprobs": 0, "max_tokens": 256}
    # An explicit budget is capped by what is left of the context.
    assert to_vllm_sampling_params({"max_tokens": 900}, 500, 512, 256)["max_tokens"] == 268
    assert "logprobs" not in to_vllm_sampling_params({"logprobs": False}, 10, 512, 256)


def test_token_output_mapping():
    output = to_token_output(
        {
            "finish_reason": "length",
            "token_ids": [5, 6, 7],
            "logprobs": {"content": [{"token_id": t, "logprob": -0.5 * t} for t in (5, 6, 7)]},
            "weight_versions": [{"version": "3", "start": 0, "end": 2}, {"version": "4", "start": 2, "end": 3}],
        }
    )
    assert output.token_ids == [5, 6, 7] and output.log_probs == [-2.5, -3.0, -3.5]
    assert output.stop_reason == "completed"
    assert output.extra_fields == {"global_steps": 4, "min_global_steps": 3, "max_global_steps": 4}

    aborted = to_token_output({"finish_reason": "abort", "token_ids": [], "weight_versions": None})
    assert aborted.stop_reason == "aborted" and aborted.token_ids == [] and "global_steps" not in aborted.extra_fields


@pytest.fixture
def fake_vllm():
    """A fake /inference/v1/generate that records requests and echoes a fixed answer."""
    import uvicorn
    from fastapi import FastAPI, Request

    app, requests = FastAPI(), []

    @app.post("/inference/v1/generate")
    async def generate(request: Request):
        body = await request.json()
        requests.append(body)
        return {
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "token_ids": [1, 2],
                    "logprobs": {"content": [{"token_id": 1, "logprob": -0.1}, {"token_id": 2, "logprob": -0.2}]},
                    "weight_versions": [{"version": "7", "start": 0, "end": 2}],
                }
            ]
        }

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}", requests
    server.should_exit = True
    thread.join(timeout=10)


def test_handle_generates_over_http(fake_vllm):
    url, requests = fake_vllm
    # The handle travels through the load balancer actor, so it must survive pickling.
    handle = pickle.loads(pickle.dumps(HttpServerHandle(url, prompt_length=512, response_length=128)))
    output = asyncio.run(
        handle.generate.remote(
            request_id="r1",
            prompt_ids=[9, 9, 9],
            sampling_params={"temperature": 0.7, "logprobs": True},
            session_id="s1",
        )
    )
    assert output.token_ids == [1, 2] and output.log_probs == [-0.1, -0.2]
    assert output.extra_fields["global_steps"] == 7
    assert requests == [
        {
            "request_id": "r1",
            "token_ids": [9, 9, 9],
            "sampling_params": {"temperature": 0.7, "logprobs": 0, "max_tokens": 128},
        }
    ]

    with pytest.raises(NotImplementedError):
        asyncio.run(handle.generate.remote(request_id="r2", prompt_ids=[1], sampling_params={}, image_data=[object()]))
