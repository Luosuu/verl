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
"""The weight registry over HTTP, for rollout processes outside the trainer's Ray job.

The server exposes an `InMemoryWeightRegistry`; `HttpWeightRegistry` is the matching client. Each
`WeightRegistry` method is one POST route taking and returning JSON. The service is unauthenticated:
serve it on a network only the trainer and rollout nodes can reach.
"""

import asyncio
import socket
import threading
import time
from dataclasses import asdict
from typing import Annotated, Any

import aiohttp

from .registry import (
    InMemoryWeightRegistry,
    Lease,
    LeaseError,
    ReplicaInfo,
    WeightManifest,
    WeightRegistry,
    manifest_from_dict,
    manifest_to_dict,
)

_LEASE_ERROR_STATUS = 409


def build_registry_app(registry: InMemoryWeightRegistry):
    """A FastAPI app serving `registry`."""
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI()

    async def call(fn, *args):
        try:
            return await fn(*args)
        except LeaseError as e:
            raise HTTPException(status_code=_LEASE_ERROR_STATUS, detail=str(e)) from e
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    def manifest_or_none(manifest: WeightManifest | None):
        return None if manifest is None else manifest_to_dict(manifest)

    @app.post("/publish")
    async def publish(body: Annotated[dict, Body()]):
        await call(registry.publish, manifest_from_dict(body["manifest"]))

    @app.post("/latest_ready")
    async def latest_ready():
        return manifest_or_none(await registry.latest_ready())

    @app.post("/get_manifest")
    async def get_manifest(body: Annotated[dict, Body()]):
        return manifest_or_none(await registry.get_manifest(body["version"]))

    @app.post("/acquire_lease")
    async def acquire_lease(body: Annotated[dict, Body()]):
        return asdict(await call(registry.acquire_lease, body["version"], body["ttl"]))

    @app.post("/renew_lease")
    async def renew_lease(body: Annotated[dict, Body()]):
        return asdict(await call(registry.renew_lease, body["lease_id"], body["ttl"]))

    @app.post("/release_lease")
    async def release_lease(body: Annotated[dict, Body()]):
        await registry.release_lease(body["lease_id"])

    @app.post("/is_lease_valid")
    async def is_lease_valid(body: Annotated[dict, Body()]):
        return await registry.is_lease_valid(body["lease_id"])

    @app.post("/retire")
    async def retire(body: Annotated[dict, Body()]):
        await call(registry.retire, body["version"])

    @app.post("/try_reclaim")
    async def try_reclaim(body: Annotated[dict, Body()]):
        return await registry.try_reclaim(body["version"])

    @app.post("/report_replica")
    async def report_replica(body: Annotated[dict, Body()]):
        await registry.report_replica(body["replica_id"], body["url"], body["version"])

    @app.post("/list_replicas")
    async def list_replicas():
        return {replica_id: asdict(info) for replica_id, info in (await registry.list_replicas()).items()}

    return app


class RegistryServer:
    """Serves an `InMemoryWeightRegistry` over HTTP from a background thread.

    Run it as a Ray actor so it lives on a node the rollout servers can reach:
    `ray.remote(RegistryServer).remote(port=...)`, then pass `address()` to every client.

    Args:
        host: Interface to listen on.
        port: Port to listen on; 0 picks a free one.
        timeout_keep_alive: Seconds the server keeps an idle client connection open.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 0, timeout_keep_alive: int = 5):
        import uvicorn

        if port == 0:
            with socket.socket() as probe:
                probe.bind((host, 0))
                port = probe.getsockname()[1]
        self.registry = InMemoryWeightRegistry()
        config = uvicorn.Config(
            build_registry_app(self.registry),
            host=host,
            port=port,
            log_level="warning",
            timeout_keep_alive=timeout_keep_alive,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        self.port = port
        # Return only once clients can connect.
        deadline = time.monotonic() + 30
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError(f"Weight registry server failed to start on {host}:{port}")
            time.sleep(0.01)

    def address(self) -> str:
        """`http://<node ip>:<port>`, as seen from other nodes."""
        import ray

        return f"http://{ray.util.get_node_ip_address()}:{self.port}"

    def stop(self):
        self._server.should_exit = True
        self._thread.join(timeout=10)


class HttpWeightRegistry(WeightRegistry):
    """`WeightRegistry` client for a `RegistryServer`.

    Args:
        url: Base URL of the registry server, e.g. `http://10.0.0.2:8900`.
        timeout: Seconds per request.
    """

    def __init__(self, url: str, timeout: float = 30.0):
        self.url = url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def _post(self, route: str, body: dict[str, Any] | None = None) -> Any:
        # A session is bound to the event loop it was created on.
        loop = asyncio.get_running_loop()
        if self._session is None or self._loop is not loop:
            # No pooled connections: calls are seconds apart, and a pooled connection the server closed
            # for being idle fails the next call with ServerDisconnectedError.
            self._session = aiohttp.ClientSession(
                timeout=self.timeout, connector=aiohttp.TCPConnector(force_close=True)
            )
            self._loop = loop
        async with self._session.post(f"{self.url}/{route}", json=body or {}) as response:
            if response.status == _LEASE_ERROR_STATUS:
                raise LeaseError((await response.json()).get("detail"))
            if response.status == 404:
                raise KeyError((await response.json()).get("detail"))
            if response.status == 400:
                raise ValueError((await response.json()).get("detail"))
            response.raise_for_status()
            return await response.json()

    async def publish(self, manifest: WeightManifest):
        await self._post("publish", {"manifest": manifest_to_dict(manifest)})

    async def latest_ready(self) -> WeightManifest | None:
        data = await self._post("latest_ready")
        return None if data is None else manifest_from_dict(data)

    async def get_manifest(self, version: int) -> WeightManifest | None:
        data = await self._post("get_manifest", {"version": version})
        return None if data is None else manifest_from_dict(data)

    async def acquire_lease(self, version: int, ttl: float) -> Lease:
        return Lease(**await self._post("acquire_lease", {"version": version, "ttl": ttl}))

    async def renew_lease(self, lease_id: str, ttl: float) -> Lease:
        return Lease(**await self._post("renew_lease", {"lease_id": lease_id, "ttl": ttl}))

    async def release_lease(self, lease_id: str):
        await self._post("release_lease", {"lease_id": lease_id})

    async def is_lease_valid(self, lease_id: str) -> bool:
        return await self._post("is_lease_valid", {"lease_id": lease_id})

    async def retire(self, version: int):
        await self._post("retire", {"version": version})

    async def try_reclaim(self, version: int) -> bool:
        return await self._post("try_reclaim", {"version": version})

    async def report_replica(self, replica_id: str, url: str, version: int):
        await self._post("report_replica", {"replica_id": replica_id, "url": url, "version": version})

    async def list_replicas(self) -> dict[str, ReplicaInfo]:
        data = await self._post("list_replicas")
        return {replica_id: ReplicaInfo(**info) for replica_id, info in data.items()}

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None
