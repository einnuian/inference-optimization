# Baseline methods and experiment tools

The completed trial evaluates EAGLE-3, DFlash, and the H-Spec authors' P-EAGLE reproduction. SGLang and vLLM ModelRunnerV1 provide their matching autoregressive (AR) references. The pilot adds the original 100%, 75%, and 50% feed-forward checkpoints and a Transformers AR reference. Together, these nine configurations produce 810 timed responses from 30 prompts and three repetitions.

The [project README](../README.md) presents the objectives, metrics, measured findings, and current reproduction commands. The [evaluation report](../measurements/report/report.pdf) contains the full method-by-benchmark tables. Compute cost means GPU time per output token. Memory and whole-job allocation totals retain their separate scopes.

## Current measurement scope

| Method | Measured execution path | Matching AR reference |
|---|---|---|
| EAGLE-3 | SGLang 0.5.19 uses its converted released checkpoint. | The target-only SGLang server supplies the reference. |
| DFlash | SGLang 0.5.19 uses the released UltraChat checkpoint. | The target-only SGLang server supplies the reference. |
| P-EAGLE | The H-Spec reproduction uses patched vLLM 0.25.1 and ModelRunnerV1. | The target-only ModelRunnerV1 server supplies the reference. |
| Pilot FFN 100% | The custom four-layer drafter uses the imported original checkpoint. | The active AR uses another physical GPU, so paired cells remain empty. |
| Pilot FFN 75% and 50% | The custom drafter uses the two imported narrowed checkpoints. | The complete Transformers AR reference uses the same physical GPU. |

Every primary configuration has 90 completed requests. EAGLE-3 also has 90 saved diagnostic responses. DFlash has four diagnostic blocks containing 12 responses. Further diagnostics remain paused. DSpark and its vLLM ModelRunnerV2 reference remain deferred.

The initial four-prompt setup checks passed for the prepared methods. The completed 30-prompt trial reveals output differences from the matching AR references. Exact agreement covers 11 unique prompts for EAGLE-3, 13 for DFlash, and 14 for P-EAGLE. Each narrowed pilot matches on 12. Their response lengths can also differ. The speed results describe these observed responses, and their cause remains unresolved. Benchmark task accuracy requires a separate scoring procedure.

## Run, inspect, and reproduce

| Guide or tool | Purpose |
|---|---|
| [Measurement runbook](measurement-runbook.md) | The guide explains client clocks, saved response viewing, manual prompts, and resource accounting. |
| [Pilot runbook](team-peagle-runbook.md) | The guide documents import, adapter qualification, primary evaluation, and optional future training commands. |
| [Report update procedure](../measurements/report/README.md) | The procedure assembles immutable evidence, exports metrics, compiles the report, and prepares publication. |
| [Initial sequence controller](trial_sequence.py) | The controller qualifies and measures the selected released-method configurations sequentially. |
| [Pilot sequence controller](team_peagle_sequence.py) | The controller imports checkpoints and runs the selected pilot configurations. |
| [Streaming recorder](trial_recorder.py) | The recorder preserves exact inputs, native token IDs, output text, and monotonic timestamps. |
| [Metric exporter](trial_export.py) | The exporter validates measurements, paired references, native counts, and empty-cell reasons. |
| [Metric catalogue snapshot](metric-catalogue.ts) | The snapshot supplies the definitions through the exporter's `--catalogue` argument. |
| [Resource collection](trial_resources.py) | The tool samples GPU memory and records final Slurm allocations. |
| [Site recording packager](trial_site_assets.py) | The packager preserves original JSON in the ZIP and publishes compressed browser assets with checksums. |
| [W&B publisher](trial_wandb.py) | The publisher validates selected records and preserves versioned result artifacts. |

The tracked [benchmark selection](../benchmarks/manifest.json) contains the exact trial inputs and their attribution. The original notebook and its saved HTML preserve the earlier pilot. Current results use imported original weights. New optimization experiments remain planned.

## Environment and preparation records

The following documents preserve setup and artifact history. Their dated setup outcomes identify the evidence available at that stage. The project README and final report describe the completed measurement state.

| Record | Contents |
|---|---|
| [Completed installation and output checks](setup-report.md) | The record identifies the prepared environments and initial qualification observations. |
| [Core methods](core-methods.md) | The record lists pinned source revisions, checkpoints, target compatibility, and implementation boundaries. |
| [Insomnia setup](insomnia.md) | The guide explains access, private storage, scheduling, and the initial Python environment. |
| [SGLang setup](sglang-setup.md) | The record preserves the pinned CUDA 12.9 container and initial EAGLE-3 and DFlash checks. |
| [Patched vLLM setup](hspec-setup.md) | The record preserves the H-Spec patches, model runners, and loader audit. |
| [PARD-2 setup](pard2-setup.md) | The record preserves the exploratory eager adapter and initial output checks. |
| [Notebook review](notebook-review.md) | The review explains the historical custom-drafter implementation and saved pilot results. |
| [Tracking preparation](tracking.md) | The record documents the earlier setup runs and synchronization procedure. |
| [Recent paper artifacts](recent-artifacts.md) | The catalogue records DSpark and related released sources. |
| [Additional paper artifacts](other-artifacts.md) | The catalogue records the wider speculative-decoding literature and available artifacts. |

The prepared P-EAGLE reproduction uses five target feature layers; the original paper describes three. The deferred DSpark reproduction uses a fixed proposal length and skips its confidence head. H-Spec and PARD-2 remain preparation history and related work. Their preparation records preserve those distinctions.
