# Evaluation

Use a separate `.venv-eval` environment for evaluation. After installing the
base dependencies from [INSTALL.md](INSTALL.md), install `transformers==4.48.0`
and `peft==0.14.0` in that environment, together with the documented ViCLIP
dependency. The generation environment continues to use its own version pins.
`tools/evaluate.sh` selects `.venv-eval/bin/python`; `EVAL_PYTHON` can override it. Official weight sources and remaining release artifacts
are listed in [CHECKPOINTS.md](CHECKPOINTS.md).

The evaluator computes MAE, CLIP-T, TA, three inter-shot consistency measures,
three reference consistency measures, and SDC. It accepts arbitrary method names
and existing videos; no manuscript score files or baseline installation are used.

## Evaluation environment

Create this environment separately from generation, using the same Python and
PyTorch/CUDA versions in the installation guide:

```bash
python3.12 -m venv .venv-eval
.venv-eval/bin/python -m pip install -r requirements.txt
.venv-eval/bin/python -m pip install transformers==4.48.0 peft==0.14.0
.venv-eval/bin/python -m pip install --no-deps vbench==0.1.5
```

Install the compiled kernels required by VideoAlign following the installation
guide, using `.venv-eval/bin/python`. Place all evaluation models and the fixed
The SDC bank is already in the repository (`evaluation/assets/sdc_bank/`). Place the
evaluation models in the paths listed in [CHECKPOINTS.md](CHECKPOINTS.md).

## Output manifest

Create a small JSON list, for example:

```json
[
  {"method":"Refrain", "case":2, "video_path":"refrain/2.mp4"},
  {"method":"StoryMem", "case":2, "video_path":"storymem/2.mp4", "mae_applicable":false}
]
```

Paths are relative to the output manifest. Case IDs are those in the benchmark.
Set `mae_applicable=false` for shotwise-concatenated generation. The optional
`submitted_cut_seconds` field contains internal cut times only, in seconds, when
an API/model used a submitted timeline different from the benchmark timeline.
Content metrics continue to use the fixed benchmark intervals. Set
`reference_metrics_applicable=false` when the evaluation protocol omits a method's
reference scores; this is explicit metadata and does not filter by score.

```bash
tools/evaluate.sh --videos outputs/videos.json --output metrics
```

Run an individual stage with `python3 -m evaluation.run worker --stage
reference_semantic --videos outputs/videos.json --output metrics`.
Workers support `--shards N --rank K`. Use different ranks for different workers,
then run `python3 -m evaluation.run summarize --output metrics` after all
workers finish. Never launch two workers with the same shard assignment/output.
Finished measurements are reused; failed units are retried. Use a fresh output
folder after changing model weights, input videos, or scoring code.

## Definitions

- **MAE:** TransNetV2 threshold 0.8; order-preserving cut matching, one-second
  missing/extra-cut penalty, normalized by the larger cut count. Videos with no
  detected cut are inapplicable under the manuscript protocol and retain that
  status in the output.
- **CLIP-T:** frame/short-shot-description CLIP compatibility. Caption chunks preserve
  text beyond CLIP's context window. Shot scores are averaged per video.
- **TA:** VideoAlign's published normalized text-alignment reward. Evaluation uses
  original long shot descriptions plus shared style, including for referenced
  shots whose inference description was replaced by a generic control instruction.
- **Inter-Shot Semantic:** ViCLIP cosine over the fixed annotated comparable pairs,
  eight frames per requested shot. Different-person cuts are not automatically
  treated as identity-preservation requirements.
- **Inter-Shot Subject/Scene:** masked DINOv2-small features on recurring entities
  and shared scenes only. GroundingDINO/SAM provide foreground boxes/masks.
- **Reference Subject/Background:** matched references and destination intervals,
  with eligibility specified in the benchmark. Subject/material edits are omitted
  from subject comparisons; background/layout/material edits from background
  comparisons. Ambiguous or missing detections are recorded as failures.
- **Reference Semantic:** eight frames sampled independently from each complete
  native reference clip and requested target shot. ViCLIP video embeddings are
  compared with cosine similarity, averaged over references per video and then
  across videos. All video-reference edit categories are scored; PNG anchors are
  excluded. This is overall semantic correspondence, not an isolated motion score.
- **SDC:** TransNetV2-detected shots, whole-video exclusion if any detected shot has fewer than eight frames,
  three-frame boundary guards, up to four
  distinct frames per usable shot. Content features use DINOv2-small, style uses
  CSD's `module.last_layer_style` and VGG16 AdaIN moments (layers 3, 8, 15, 22).
  The CSD extractor returns **style first**, content second. Energy contrasts,
  robust residual correction, conditional ranks, three-group averaging and 2:1
  geometric CSD/VGG pooling reuse the numerical implementation. Query-source folds
  are excluded using the input source identity in the benchmark.

`summary.csv` reports video-weighted means and measurement coverage. Per-video
JSON files preserve scores, invalid units, and sampled indices. Do not turn missing
or inapplicable values into zero.

## Building an SDC bank

The bundled bank (see [CHECKPOINTS.md](CHECKPOINTS.md#sdc-real-video-bank)) reproduces
the paper's numbers. Pass `--bank DIR` to `tools/evaluate.sh` to use another one;
both the full array layout and the compact text layout are accepted. A bank fitted on a different real-video corpus produces a new
calibration and potentially different scores.

To construct a bank from local real videos, provide a JSON list with `source_id`
(a unique 32-character hexadecimal source identity) and `video_path` for each video:

```bash
python3 -m evaluation.calibrate extract --manifest train/real_videos.json --output outputs/calibration_features
python3 -m evaluation.calibrate fit --features outputs/calibration_features --output weights/sdc_full_bank
python3 -m evaluation.style_consistency.bank.compact_bank weights/sdc_full_bank outputs/my_sdc_bank
```

Extraction detects real-video shots with TransNetV2, stores float32 CSD style/content,
DINO and VGG embeddings, and preserves exact sampled coordinates. Fitting uses
balanced within-shot splits with equal video/shot mass, three groups and five
source folds. The fit command is an in-memory reference implementation; for the
full multi-million-video setting use the included array-cache builder under
`evaluation/style_consistency/bank/build_bank.py`, with an externally prepared
cache matching its documented schema. Cached embeddings and corpus videos are not
distributed.

## Fixed eligibility

`configs/metric_eligibility.json` fixes the TA and inter/reference
subject, scene and semantic eligibility units used in aggregation. Generation
prompts and evaluation descriptions serve different roles; changing generation
text does not rewrite evaluation annotations. Summary means average eligible
units within each video, then average videos.
