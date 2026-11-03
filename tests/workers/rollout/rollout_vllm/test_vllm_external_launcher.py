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
"""An official `vllm serve`, started by verl's external launcher, follows published weight versions.

The test process runs the weight registry server and publishes three versions: the original weights,
weights with the final norm zeroed (all logits 0, so greedy decoding repeats one token), and the original
weights again. Greedy outputs from vLLM's own `/inference/v1/generate` show which weights it serves, and
vLLM's per-token weight version spans show the label the sidecar set.

Usage:
    pytest tests/workers/rollout/rollout_vllm/test_vllm_external_launcher.py -v -s
"""

import asyncio
import os
import socket
import subprocess
import sys
import time

import aiohttp
import pytest
import torch

MODEL_PATH = os.path.expanduser(os.getenv("WEIGHT_UPDATE_AGENT_TEST_MODEL", "~/models/Qwen/Qwen2.5-0.5B-Instruct"))


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _versions():
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)
    original = [(name, tensor.detach().clone()) for name, tensor in model.state_dict().items()]
    broken = [(name, torch.zeros_like(t) if name == "model.norm.weight" else t) for name, t in original]
    return [original, broken, original]


async def _generate(url, prompt_ids, max_tokens=16):
    body = {"token_ids": prompt_ids, "sampling_params": {"temperature": 0.0, "max_tokens": max_tokens}}
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{url}/inference/v1/generate", json=body) as response:
            assert response.status == 200, await response.text()
            choice = (await response.json())["choices"][0]
            return choice["token_ids"], choice.get("weight_versions")


async def _wait_for_replica_version(registry, version, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        replicas = await registry.list_replicas()
        if replicas and all(info.version >= version for info in replicas.values()):
            return replicas
        await asyncio.sleep(0.5)
    raise TimeoutError(f"replica did not reach version {version}: {await registry.list_replicas()}")


@pytest.mark.skipif(not os.path.isdir(MODEL_PATH), reason=f"{MODEL_PATH} not found")
def test_external_vllm_follows_published_versions():
    pytest.importorskip("nixl")
    from verl.checkpoint_engine.transport.http_registry import RegistryServer

    server = RegistryServer(host="127.0.0.1", port=0)
    registry_url = f"http://127.0.0.1:{server.port}"
    vllm_port = _free_port()
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "verl.workers.rollout.vllm_rollout.external_launcher",
            "--registry-url",
            registry_url,
            "--port",
            str(vllm_port),
            "--advertise-host",
            "127.0.0.1",
            "--poll-interval",
            "0.2",
            "--",
            MODEL_PATH,
            "--gpu-memory-utilization",
            "0.5",
            "--max-model-len",
            "2048",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]},
    )
    try:
        asyncio.run(_run(registry_url, f"http://127.0.0.1:{vllm_port}", launcher))
    finally:
        launcher.terminate()
        launcher.wait(timeout=120)
        server.stop()


async def _run(registry_url, vllm_url, launcher):
    from transformers import AutoTokenizer

    from verl.checkpoint_engine.transport.http_registry import HttpWeightRegistry
    from verl.checkpoint_engine.transport.publisher import WeightPublisher
    from verl.utils.tokenizer import normalize_token_ids

    registry = HttpWeightRegistry(registry_url)
    # The sidecar registers once vllm serve is healthy, before any version exists.
    deadline = time.monotonic() + 1800
    while not await registry.list_replicas():
        assert launcher.poll() is None, f"launcher exited with {launcher.returncode}"
        assert time.monotonic() < deadline, "the launcher never registered"
        await asyncio.sleep(2)
    replica = next(iter((await registry.list_replicas()).values()))
    assert replica.url == vllm_url and replica.version == -1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    prompt_ids = normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Name three primary colors."}], add_generation_prompt=True, tokenize=True
        )
    )
    reference, _ = await _generate(vllm_url, prompt_ids)

    publisher = WeightPublisher(registry, bucket_size=256 << 20, device="cpu", backend="nixl_native")
    versions = _versions()

    await publisher.publish(iter(versions[0]), version=0)
    await _wait_for_replica_version(registry, 0)
    tokens, spans = await _generate(vllm_url, prompt_ids)
    assert tokens == reference
    assert spans and {span["version"] for span in spans} == {"0"}

    await publisher.publish(iter(versions[1]), version=1)
    await _wait_for_replica_version(registry, 1)
    tokens, spans = await _generate(vllm_url, prompt_ids)
    assert tokens != reference and len(set(tokens)) == 1, "all-zero logits should repeat one token"
    assert {span["version"] for span in spans} == {"1"}

    await publisher.publish(iter(versions[2]), version=2)
    await _wait_for_replica_version(registry, 2)
    tokens, spans = await _generate(vllm_url, prompt_ids)
    assert tokens == reference
    assert {span["version"] for span in spans} == {"2"}
    await registry.close()
