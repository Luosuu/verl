# External vLLM Rollout: Official vLLM Builds as Rollout Replicas

**Authors:** Luosuu

**Last updated:** 2026-11-03

## Background

The `decoupled_async` trainer ([design](decoupled_weight_sync.md)) no longer controls rollout: it
publishes weight versions and enqueues prompts, and each replica decides when to load new weights. The
replicas themselves are still launched by the trainer inside its Ray job, as verl's `vLLMHttpServer`
actors with a verl worker extension, plus one checkpoint engine worker per GPU that hands pulled weights
to vLLM over CUDA IPC.

This design runs rollout as **official vLLM servers** started by a small verl launcher, outside the
trainer's Ray job. verl contributes only a vLLM plugin (a pip-installed weight transfer engine) and a
sidecar process; the vLLM build is unmodified.

## What vLLM already provides

Checked against vLLM 0.32, verl's pinned version:

| Need | vLLM | Note |
| --- | --- | --- |
| Load our pull engine in every worker | Plugins in the `vllm.general_plugins` entry point group; `WeightTransferEngineFactory.register_engine`; engine selected with `--weight-transfer-config` | `VLLM_PLUGINS` selects which plugins load |
| Update weights from outside | HTTP routes `/pause`, `/resume`, `/init_weight_transfer_engine`, `/start_weight_update`, `/update_weights`, `/finish_weight_update`, `/update_weight_version` | Mounted only with `VLLM_SERVER_DEV_MODE=1`, which also enables development endpoints |
| Token-in, token-out generation | `POST /inference/v1/generate`: `token_ids` in; `token_ids`, per-token logprobs, `finish_reason` and `weight_versions` spans out | `finish_reason="abort"` marks requests cut by `/pause` |
| Checkpoint-format reload | `initialize_layerwise_reload` / `model.load_weights` / `finalize_layerwise_reload` | The pattern vLLM's own IPC engine uses |

## Architecture

```
trainer Ray job                                           rollout node (no verl Ray)
-----------------------------------------------           ------------------------------------------------
WeightPublisher --publish--> registry HTTP server  <----  launcher sidecar: poll, report replica + version
                              (versions, leases,               | /pause /start /update /finish /resume
                               replica directory)              v
PromptDispatcher -- claim --> agent loop workers  --HTTP-->  vllm serve (official build)
  (gates on replica versions     (HttpServerHandle)             + verl_transport_pull engine in each worker
   from the registry)                                              --one-sided READ--> trainer host slots
```

1. **Registry over HTTP.** The trainer serves the existing `InMemoryWeightRegistry` over HTTP. The same
   service is a replica directory: each sidecar reports its server URL and loaded version, with a
   heartbeat. `HttpWeightRegistry` implements `WeightRegistry` for the publisher, the vLLM plugin and
   the sidecar; manifests travel as JSON (bytes base64-encoded, dtypes as names).
2. **vLLM plugin.** `verl_transport_pull`, a `WeightTransferEngine` registered through the
   `vllm.general_plugins` entry point. `init_transfer_engine` creates a `WeightPuller` against the
   registry URL; `start_weight_update` starts a layerwise reload; `receive_weights({"version": v})`
   pulls version `v` into the worker's host mirror under a lease and loads it with
   `model.load_weights`; `finish_weight_update` finalizes the reload. The checkpoint engine worker and
   the CUDA IPC hop disappear.
3. **Launcher.** `python -m verl.workers.rollout.vllm_rollout.external_launcher` sets `VLLM_PLUGINS`
   and `VLLM_SERVER_DEV_MODE`, starts `vllm serve` with `--weight-transfer-config
   '{"backend": "verl_transport_pull"}'` plus the user's vLLM arguments, waits for it to be healthy,
   and runs the sidecar.
4. **Sidecar update agent.** The same policy as `WeightUpdateAgent`, driving vLLM's HTTP routes instead
   of Ray actors: when a newer version is ready, `/pause` (abort) → `/start_weight_update` →
   `/update_weights` → `/finish_weight_update` → `/update_weight_version` → `/resume`, then report the
   new version to the registry.
5. **Generation client.** An `HttpServerHandle(url)` stands in for a vLLM server actor handle: its
   `generate.remote(...)` calls `/inference/v1/generate` and returns verl's `TokenOutput`
   (`finish_reason="abort"` → `stop_reason="aborted"`, so `FullyAsyncLLMServerClient` resubmits it; the
   last weight version span → `global_steps`). The load balancer and client are unchanged.
6. **Trainer wiring.** `trainer.v1.decoupled_async.external_rollout.enable` makes the trainer start the
   registry server instead of launching replicas, wait for `min_replicas` sidecars to register, and build
   the LLM client from their URLs. The prompt dispatcher reads replica versions from the registry.

Agent loops and the dispatcher stay in the trainer's Ray job for now: they are CPU actors, and keeping
them there means TransferQueue does not need to be reachable from outside.

## Limitations

- **Development endpoints.** `VLLM_SERVER_DEV_MODE=1` exposes unauthenticated control routes; the
  rollout servers must sit on a network only the trainer can reach.
- **No prefetch.** The plugin pulls inside `/update_weights`, so generation is paused for the whole
  transfer. Prefetching needs either a background pull in the plugin or a vLLM route that runs the
  engine outside the pause.
- **Fixed replica set.** The trainer uses the replicas registered at startup; replicas joining or
  leaving during training are follow-ups.
- **Text-only.** The HTTP client sends token IDs and sampling parameters; multimodal inputs and
  routed-expert capture are not forwarded.
- **vLLM API drift.** The plugin targets vLLM 0.32's `WeightTransferEngine` interface.
