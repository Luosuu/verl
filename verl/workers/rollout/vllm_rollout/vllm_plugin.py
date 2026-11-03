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
"""vLLM general plugin: registers verl's weight transfer engines with an official vLLM build.

Installed as the ``verl_transport_pull`` entry point of the ``vllm.general_plugins`` group, so vLLM
calls `register` in each of its processes when ``VLLM_PLUGINS`` names it. Registration is lazy: the
engine module is imported only when a server selects the backend.
"""


def register() -> None:
    from vllm.distributed.weight_transfer import WeightTransferEngineFactory

    name = "verl_transport_pull"
    if name not in WeightTransferEngineFactory._registry:
        WeightTransferEngineFactory.register_engine(
            name, "verl.workers.rollout.vllm_rollout.transport_pull_engine", "TransportPullWeightTransferEngine"
        )
