# Public runtime

The released model is named `Refrain`, implemented by `RefrainTransformer` in
`hymm/models/modules/Refrain.py`. `configs/train.yaml` and `configs/infer.yaml`
record the architecture, training stages, SP sizes and reference-control settings.
The VAE is configured at `weights/vae_3d/refrain_vae/`. Refrain weights are not
released (see [CHECKPOINTS.md](CHECKPOINTS.md)); public HunyuanVideo-1.5 checkpoints
are not interchangeable with them.

## Distributed execution

Sequence parallelism uses public PyTorch DeviceMesh and collective operations,
following HunyuanVideo-1.5. Video tokens are exchanged between sequence shards and
attention-head shards. Output all-gather sums gradients across consumers, including
uneven token shards. CUDA uses NCCL; the small CPU checks use Gloo.

The dense runtime supports data parallelism, sequence parallelism, and FSDP2.
All training stages use AdamW and TP=PP=EP=1. MoE, expert parallelism and
pipeline-parallel engines are not part of this runtime.

Arrow readers are bundled from public HunyuanDiT source with local sidecar and
resolution access. The active video dataset continues to read its existing Arrow
format. Sampler offsets count consumed examples per data-parallel rank; epoch
shuffling happens before that offset is applied. No separate sampler package is required.

## Checkpoints

Full inference weights keep the same tensor keys and can be supplied as a single
`.safetensors` file. Load weights before FSDP wrapping, as in the existing inference
entrypoint. Checkpoint names stored in the file name do not select the architecture;
`model_type: Refrain` in the configuration does.

The public engine writes the following training checkpoint layout:

```
global_step-<step>/
  distributed/       # torch.distributed.checkpoint model and optimizer state
  trainer.pt         # training counters and scheduler state
  rng_rank_<rank>.pt # per-rank RNG state
```

These directories can be loaded by the public engine and inference entrypoint.
Resume with optimizer state also restores scheduler and RNG state. The retained
stage-transition training call loads model and counters without optimizer state,
as in the original stage-transition code.

Other distributed checkpoint layouts require conversion to a model safetensors
file or to the training-state layout above. For a weights-only start, supply the converted safetensors
checkpoint and set `resume: null` in the configuration. Exact training
continuation requires compatible optimizer, scheduler and RNG states.
