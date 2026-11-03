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
"""The weight registry over HTTP: serialization, every route, and a publish/pull round trip."""

import asyncio
import time

import pytest
import torch

from tests.checkpoint_engine.test_transport_pull_on_cpu import _FakeTransport, _weights
from verl.checkpoint_engine.transport import (
    BucketDesc,
    LeaseError,
    TensorSpec,
    WeightManifest,
    manifest_from_dict,
    manifest_to_dict,
)
from verl.checkpoint_engine.transport.http_registry import HttpWeightRegistry, RegistryServer
from verl.checkpoint_engine.transport.publisher import WeightPublisher
from verl.checkpoint_engine.transport.puller import WeightPuller


def _manifest(version):
    return WeightManifest(
        version=version,
        endpoint=b"\x00agent-metadata\xff",
        total_bytes=4096,
        tensors=[
            TensorSpec(name="w", shape=torch.Size([16, 64]), dtype=torch.bfloat16, offset=0),
            TensorSpec(name="ids", shape=torch.Size([3]), dtype=torch.int64, offset=2048),
        ],
        buckets=[BucketDesc(offset=0, size=4096, remote_buffer=b"\x01\x02descriptor")],
        extra={"backend": "nixl_native"},
    )


@pytest.fixture
def server():
    server = RegistryServer(host="127.0.0.1", port=0)
    url = f"http://127.0.0.1:{server.port}"
    # uvicorn starts in a background thread.
    for _ in range(100):
        try:
            asyncio.run(HttpWeightRegistry(url).list_replicas())
            break
        except Exception:
            time.sleep(0.05)
    yield url
    server.stop()


def test_manifest_round_trips_through_json():
    manifest = _manifest(3)
    assert manifest_from_dict(manifest_to_dict(manifest)) == manifest


def test_every_route(server):
    async def main():
        registry = HttpWeightRegistry(server)
        assert await registry.latest_ready() is None
        await registry.publish(_manifest(1))
        await registry.publish(_manifest(2))
        assert await registry.latest_ready() == _manifest(2)
        assert await registry.get_manifest(1) == _manifest(1)
        assert await registry.get_manifest(9) is None
        with pytest.raises(ValueError):
            await registry.publish(_manifest(2))

        lease = await registry.acquire_lease(2, ttl=60)
        assert lease.version == 2 and await registry.is_lease_valid(lease.lease_id)
        assert (await registry.renew_lease(lease.lease_id, ttl=60)).lease_id == lease.lease_id
        await registry.retire(2)
        with pytest.raises(LeaseError):
            await registry.acquire_lease(2, ttl=60)
        with pytest.raises(KeyError):
            await registry.retire(7)
        assert not await registry.try_reclaim(2)
        await registry.release_lease(lease.lease_id)
        assert await registry.try_reclaim(2)
        with pytest.raises(LeaseError):
            await registry.renew_lease(lease.lease_id, ttl=60)

        await registry.report_replica("r0", "http://10.0.0.5:8000", -1)
        await registry.report_replica("r0", "http://10.0.0.5:8000", 1)
        replicas = await registry.list_replicas()
        assert set(replicas) == {"r0"} and replicas["r0"].version == 1 and replicas["r0"].url.endswith(":8000")
        await registry.close()

    asyncio.run(main())


def test_publish_and_pull_through_http(server):
    async def main():
        registry = HttpWeightRegistry(server)
        publisher = WeightPublisher(
            registry, bucket_size=4096, device="cpu", rollout_dtype=None, transport_factory=_FakeTransport
        )
        puller = WeightPuller(registry, device="cpu", transport_factory=_FakeTransport, poll_interval=0.01)
        for step in (1, 2, 3):
            await publisher.publish(iter(_weights(step)), version=step)
            pulled = await puller.fetch(await puller.wait_for_version(step, timeout=5))
            received = dict(pulled.named_tensors())
            for name, expected in _weights(step):
                assert torch.equal(received[name], expected), name
        await registry.close()

    asyncio.run(main())
