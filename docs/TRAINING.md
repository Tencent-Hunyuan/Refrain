# Training configuration

| Stage | Cumulative steps | Learning rate | Recorded SP size | Input |
|---|---:|---:|---:|---|
| T2V | 0–19,000 | 1e-5 | 2 | Text and target video |
| Reference reconstruction | 19,000–36,000 | 1e-5 | 2 | References matched to target duration |
| Variable-duration reconstruction | 36,000–45,000 | 2e-6 | 8 | References of variable duration |

All three stages live in [`configs/train.yaml`](../configs/train.yaml): a `shared`
block plus one override block per stage. Values come from the saved arguments of
the main run; anything not listed uses the defaults in `hymm/config.py`.

```bash
tools/train.sh text_to_video
tools/train.sh reference_training           # resumes outputs/text_to_video/global_step-19000
tools/train.sh variable_duration_training   # resumes outputs/reference_training/global_step-36000
```

`--resume DIR` and `--output DIR` override the configured paths. Multi-node runs use
the usual `NNODES`, `NODE_RANK`, `MASTER_ADDR` and `MASTER_PORT` variables.
The first stage starts from `weights/initialization/model.safetensors`, which is
not released; supply a compatible DiT of your own.

The second and third stages load model weights and training counters from their
preceding stage checkpoints. The stage-transition call resets optimizer state.
A standalone safetensors file supplies model weights without training counters;
see [Runtime](RUNTIME.md) for the checkpoint format.

## Data contract

The loader consumes an Arrow index JSON with `arrow_list`, `cache_folder_list`,
`aspect_ratio_index_list`, `aspect_ratios_nums_list`, and temporal bucket metadata.
`configs/train.yaml` specifies the exact caption, latent and condition column
names. Prepare these columns using your own dataset; no training CSV or Arrow
inventory is included in this repository.

Reference records retain their selected source start and destination interval.
`segment_conditions` describes per-shot edited references. The original video
remains the reconstruction target. Reference editing changes the conditioning
input; it does not relabel the target. Text-only sampling and examples without
selected references maintain the T2V mode.

At the final stage, `range_condition_temporal_pack_mode=reference_timeline` places native
reference latents without temporal interpolation. A target interval may be longer
or shorter than its reference. Preserve target duration, source start, and image
anchor coordinates. Do not force equal lengths or pad a static reference tail.

Training raw media, reference generation, and annotation credentials are not part
of the code release. To inspect the resolved training settings without loading any
model, run `python tools/train.py text_to_video --dry-run`.
