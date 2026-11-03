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
"""Launch an official vLLM server as a rollout replica of a decoupled_async trainer.

Runs ``vllm serve`` with verl's pull engine selected, then acts as the replica's sidecar: it keeps the
server on a recent published weight version and reports the server's URL and version to the trainer's
weight registry. Everything after ``--`` goes to ``vllm serve`` unchanged::

    python -m verl.workers.rollout.vllm_rollout.external_launcher \\
        --registry-url http://<trainer node>:8900 --port 8000 -- \\
        ~/models/Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 2

The server runs with ``VLLM_SERVER_DEV_MODE=1`` (vLLM's weight update routes need it), which also exposes
vLLM's unauthenticated development routes: keep it on a network only the trainer can reach.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import aiohttp

from verl.checkpoint_engine.transport.http_registry import HttpWeightRegistry
from verl.checkpoint_engine.transport.registry import WeightRegistry
from verl.checkpoint_engine.transport.update_agent import WeightUpdateAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

PLUGIN_NAME = "verl_transport_pull"


class HttpReplicaUpdateAgent(WeightUpdateAgent):
    """`WeightUpdateAgent` for a vLLM server driven over its HTTP weight update routes.

    Applying a version: ``/pause`` (abort; clients resubmit cut requests), ``/start_weight_update``,
    ``/update_weights``, ``/finish_weight_update`` (which also labels the new version), ``/resume``. The
    pull happens inside ``/update_weights``, so generation is paused for the whole transfer. On every poll
    the agent reports the server's URL and loaded version to the registry, which is also its heartbeat.

    Args:
        server_url: Base URL of the vLLM server.
        replica_id: Name the replica reports under.
        registry: The trainer's weight registry.
        init_info: Init info of the server's ``verl_transport_pull`` engine.
        min_version_lag: Apply once the newest version is this many versions ahead.
        poll_interval: Seconds between registry polls.
        request_timeout: Seconds per HTTP call; an update can take as long as a full pull.
    """

    def __init__(
        self,
        server_url: str,
        replica_id: str,
        registry: WeightRegistry,
        init_info: dict,
        min_version_lag: int = 1,
        poll_interval: float = 0.5,
        request_timeout: float = 1800.0,
    ):
        super().__init__(
            servers=[], workers=[], registry=registry, min_version_lag=min_version_lag, poll_interval=poll_interval
        )
        self.server_url = server_url.rstrip("/")
        self.replica_id = replica_id
        self.init_info = init_info
        self.request_timeout = aiohttp.ClientTimeout(total=request_timeout)

    async def _post(self, route: str, body: dict | None = None, params: dict | None = None):
        async with aiohttp.ClientSession(timeout=self.request_timeout) as session:
            async with session.post(f"{self.server_url}/{route}", json=body, params=params) as response:
                if response.status >= 400:
                    raise RuntimeError(f"POST /{route} failed with {response.status}: {await response.text()}")

    async def initialize(self, registry_timeout: float = 3600.0):
        """Point the server's pull engine at the registry and register; call once before `run`.

        Waits for the registry to come up, so servers can be started before the trainer.
        """
        await self._post("init_weight_transfer_engine", {"init_info": self.init_info})
        deadline = time.monotonic() + registry_timeout
        while True:
            try:
                await self._heartbeat()
                return
            except (aiohttp.ClientError, OSError) as e:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Weight registry unreachable for {registry_timeout}s") from e
                logger.info(f"Waiting for the weight registry: {e}")
                await asyncio.sleep(5)

    async def _heartbeat(self):
        await self.registry.report_replica(self.replica_id, self.server_url, self.version)

    async def _prefetch(self, version: int) -> bool:
        # The engine pulls inside /update_weights; a separate prefetch needs a background pull in the engine.
        return True

    async def _apply(self, version: int):
        await self._post("pause", params={"mode": "abort", "clear_cache": "true"})
        await self._post("start_weight_update")
        await self._post("update_weights", {"update_info": {"version": version}})
        await self._post("finish_weight_update", {"weight_version": str(version)})
        await self._post("resume")


def _advertised_host() -> str:
    """This node's address as other nodes reach it."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("8.8.8.8", 80))  # no packet is sent; picks the outbound interface
            return probe.getsockname()[0]
        except OSError:
            return socket.gethostbyname(socket.gethostname())


def build_vllm_command(vllm_args: list[str], host: str, port: int) -> tuple[list[str], dict[str, str]]:
    """The ``vllm serve`` command line and environment for a rollout replica."""
    env = dict(os.environ)
    plugins = [name for name in env.get("VLLM_PLUGINS", "").split(",") if name]
    env["VLLM_PLUGINS"] = ",".join(plugins + [PLUGIN_NAME] if PLUGIN_NAME not in plugins else plugins)
    env["VLLM_SERVER_DEV_MODE"] = "1"
    # Tools vLLM runs at startup (ninja for kernel builds) live next to the interpreter in a venv,
    # which is not on PATH unless the venv was activated.
    env["PATH"] = os.pathsep.join([os.path.dirname(sys.executable), env.get("PATH", "")])
    command = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        *vllm_args,
        "--host",
        host,
        "--port",
        str(port),
        "--weight-transfer-config",
        json.dumps({"backend": PLUGIN_NAME}),
        # Token-in, token-out generation (/inference/v1/generate) is what verl's rollout clients call.
        "--enable-scale-out",
    ]
    return command, env


async def _wait_until_healthy(url: str, server: subprocess.Popen, timeout: float):
    deadline = time.monotonic() + timeout
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"vllm serve exited with code {server.returncode} before becoming healthy")
            try:
                async with session.get(f"{url}/health") as response:
                    if response.status == 200:
                        return
            except aiohttp.ClientError:
                pass
            await asyncio.sleep(2)
    raise TimeoutError(f"vllm serve at {url} did not become healthy within {timeout}s")


async def _serve(args, vllm_args: list[str]) -> int:
    advertise_host = args.advertise_host or _advertised_host()
    url = f"http://{advertise_host}:{args.port}"
    command, env = build_vllm_command(vllm_args, args.host, args.port)
    logger.info(f"Starting {' '.join(command)}")
    server = subprocess.Popen(command, env=env)

    def terminate(*_):
        if server.poll() is None:
            server.terminate()

    signal.signal(signal.SIGTERM, terminate)
    try:
        await _wait_until_healthy(f"http://127.0.0.1:{args.port}", server, args.startup_timeout)
        registry = HttpWeightRegistry(args.registry_url)
        agent = HttpReplicaUpdateAgent(
            server_url=url,
            replica_id=args.replica_id or f"{advertise_host}:{args.port}",
            registry=registry,
            init_info={
                "registry_url": args.registry_url,
                "transport": args.transport,
                "device": args.mirror_device,
                "lease_ttl": args.lease_ttl,
            },
            min_version_lag=args.min_version_lag,
            poll_interval=args.poll_interval,
        )
        await agent.initialize()
        logger.info(f"Rollout replica {agent.replica_id} at {url} registered with {args.registry_url}")
        agent_task = asyncio.create_task(agent.run())
        while server.poll() is None and not agent_task.done():
            await asyncio.sleep(1)
        if agent_task.done():
            # A failed update leaves the server paused; stop it rather than keep a dead replica around.
            agent_task.result()
            return 1
        agent.stop()
        await agent_task
        return server.returncode
    finally:
        terminate()
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--" not in argv:
        raise SystemExit("Pass vllm serve arguments after `--`, starting with the model path")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry-url", required=True, help="Weight registry of the decoupled_async trainer")
    parser.add_argument("--host", default="0.0.0.0", help="Interface vLLM listens on")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--advertise-host", default=None, help="Address the trainer uses to reach this server")
    parser.add_argument("--replica-id", default=None, help="Defaults to <advertise host>:<port>")
    parser.add_argument("--transport", default="nixl_native", help="Must match the trainer's transport_pull transport")
    parser.add_argument("--mirror-device", default="cpu", help="Device of each worker's copy of a pulled version")
    parser.add_argument("--lease-ttl", type=float, default=60.0)
    parser.add_argument("--min-version-lag", type=int, default=1)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--startup-timeout", type=float, default=1800.0)
    args = parser.parse_args(argv[:split])
    logging.basicConfig(level=logging.INFO)
    return asyncio.run(_serve(args, argv[split + 1 :]))


if __name__ == "__main__":
    sys.exit(main())
