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
"""vLLM weight transfer engine that pulls published weight versions from verl's registry.

Runs inside each vLLM worker of an official vLLM build, loaded through the plugin in
`vllm_plugin.py` and selected with `--weight-transfer-config '{"backend": "verl_transport_pull"}'`.
A caller (verl's launcher sidecar) drives it over vLLM's HTTP routes:

- ``/init_weight_transfer_engine`` with ``{"registry_url": ...}`` once;
- per update: ``/start_weight_update``, ``/update_weights`` with ``{"version": v}``,
  ``/finish_weight_update``.
"""

import asyncio
import logging
import os
import time
from dataclasses import dataclass

import torch
from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

from verl.checkpoint_engine.transport.http_registry import HttpWeightRegistry
from verl.checkpoint_engine.transport.puller import WeightPuller

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

BACKEND_NAME = "verl_transport_pull"


@dataclass
class TransportPullInitInfo(WeightTransferInitInfo):
    registry_url: str
    """Base URL of verl's weight registry server."""
    transport: str = "nixl_native"
    """Transport of the publisher: `nixl_native`, or a torch.distributed._transport backend."""
    device: str = "cpu"
    """Device of this worker's local mirror of a published version."""
    lease_ttl: float = 60.0


@dataclass
class TransportPullUpdateInfo(WeightTransferUpdateInfo):
    version: int
    """Published weight version to load."""


class TransportPullWeightTransferEngine(WeightTransferEngine[TransportPullInitInfo, TransportPullUpdateInfo]):
    """Pulls a whole published version under a lease, then loads it as a checkpoint-format reload.

    The layerwise reload starts only after the pull succeeded, so a failed or torn pull leaves the
    model untouched and the update can simply be retried.
    """

    init_info_cls = TransportPullInitInfo
    update_info_cls = TransportPullUpdateInfo
    supports_draft_weight_update = False

    def __init__(self, config, vllm_config, device: torch.device, model: torch.nn.Module) -> None:
        super().__init__(config, vllm_config, device, model)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._registry: HttpWeightRegistry | None = None
        self._puller: WeightPuller | None = None
        self._reload_started = False

    def _run(self, coro):
        return self._loop.run_until_complete(coro)

    def init_transfer_engine(self, init_info: TransportPullInitInfo) -> None:
        self.shutdown()
        # vLLM calls engines synchronously; the puller and registry client are asyncio.
        self._loop = asyncio.new_event_loop()
        self._registry = HttpWeightRegistry(init_info.registry_url)
        self._puller = WeightPuller(
            self._registry, device=init_info.device, backend=init_info.transport, lease_ttl=init_info.lease_ttl
        )

    def start_weight_update(self) -> None:
        # The reload is started in receive_weights, once the pull has succeeded.
        self._reload_started = False

    def receive_weights(self, update_info: TransportPullUpdateInfo) -> None:
        if self._puller is None:
            raise RuntimeError("init_transfer_engine must be called before receiving weights")
        start = time.perf_counter()
        manifest = self._run(self._registry.get_manifest(update_info.version))
        if manifest is None:
            raise RuntimeError(f"Weight version {update_info.version} is not published (or already reclaimed)")
        pulled = self._run(self._puller.fetch(manifest))
        fetched = time.perf_counter()

        from vllm.model_executor.model_loader.mtp_validation import disable_mtp_completeness_check
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        initialize_layerwise_reload(self.model)
        self._reload_started = True
        # One tensor at a time on the device, rather than a second full copy of the model.
        weights = ((name, tensor.to(self.device, non_blocking=True)) for name, tensor in pulled.named_tensors())
        with disable_mtp_completeness_check():
            self.model.load_weights(weights)
        logger.info(
            f"Loaded weight version {update_info.version}: pull {fetched - start:.2f}s, "
            f"load {time.perf_counter() - fetched:.2f}s"
        )

    def finish_weight_update(self) -> None:
        if not self._reload_started:
            return
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        finalize_layerwise_reload(self.model, self.model_config)
        self._reload_started = False

    def shutdown(self) -> None:
        if self._puller is not None:
            self._puller.close()
            self._puller = None
        if self._registry is not None:
            self._run(self._registry.close())
            self._registry = None
        if self._loop is not None:
            self._loop.close()
            self._loop = None

    @staticmethod
    def trainer_send_weights(iterator, trainer_args) -> None:
        raise NotImplementedError("verl publishes weights through its registry; this engine only receives them")
