# Speculative decoding and drafter feed-forward compression

This Columbia High Performance Machine Learning project studies how to generate language-model responses faster with less GPU time per output token. The experiment compares EAGLE-3, DFlash, and P-EAGLE with ordinary autoregressive (AR) generation. A separate pilot evaluates a custom P-EAGLE-style drafter at 100%, 75%, and 50% feed-forward width.

The completed trial contains **30 unique prompts, three repetitions, nine configurations, and 810 timed responses**. The published catalogue contains **2,056 measured cells**, including timing, acceptance, response agreement, resource use, and eligible diagnostic observations. Each run uses one NVIDIA L40S GPU.

## Project resources

| Resource | Contents |
|---|---|
| [Project hub](https://hpml-workspace.ayomideadetayo.chatgpt.site/) | The site contains method-by-benchmark tables, source-paper context, prompt recordings, and downloads. |
| [Evaluation report](measurements/report/report.pdf) | The report explains the procedure, results, compute accounting, and research implications. |
| [Presentation](measurements/report/HPML_Midpoint_Report_All_Categories.pptx) | The eleven-slide presentation includes Overleaf result excerpts for every benchmark category. |
| [Compact measured results](measurements/results/trial-first30-20261122/) | The saved exports, configuration summaries, allocations, and provenance preserve the completed trial. |
| [Report source](measurements/report/report.tex) | The editable LaTeX source includes generated measurement tables. |
| [Final W&B run](https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22/runs/48b0a0ffa383382fa011) | The publication preserves the catalogue, recording archive, report, and presentation. |
| [W&B project](https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22) | The project preserves separate configuration runs and earlier evidence versions. |
| [GitHub repository](https://github.com/einnuian/hpml-project) | The repository contains the notebook, experiment code, tests, input selection, and report source. |

## Research questions

The first question concerns the released methods: which recorded implementation delivers the highest throughput and lowest compute cost under the common procedure? The second question concerns compression: does a narrower drafter reduce execution cost enough to improve response generation? The third question concerns acceptance: how many proposed tokens survive target verification, and how does that progress relate to speed?

A feed-forward network transforms each hidden vector inside a transformer layer. The pilot reduces the network's intermediate width while preserving its input and output dimensions. Each pilot drafter has four layers and proposes eight tokens per verification round. The target verifies those proposals and supplies a correction or bonus token.

## Evaluated configurations

AR runs the target model one token at a time. Each speculative method uses an AR reference with the corresponding inference engine. This pairing keeps the serving implementation consistent within the speedup calculation.

| Displayed method | Configuration ID | Execution path and matching reference |
|---|---|---|
| AR · SGLang | `ar-sglang` | The SGLang target-only server provides the EAGLE-3 and DFlash reference. |
| EAGLE-3 | `eagle3` | SGLang uses the converted released drafter and the SGLang AR reference. |
| DFlash | `dflash` | SGLang uses the released UltraChat drafter and the SGLang AR reference. |
| AR · vLLM ModelRunnerV1 | `ar-vllm-v1` | The target-only vLLM ModelRunnerV1 server provides the P-EAGLE reference. |
| P-EAGLE | `peagle` | The H-Spec authors' Llama reproduction uses vLLM ModelRunnerV1. |
| AR · Transformers (pilot) | `team-ar` | The notebook's Transformers decoder provides the pilot reference. |
| Pilot FFN 100% | `team-ffn-100` | The original full-width checkpoint uses the custom Transformers adapter. |
| Pilot FFN 75% | `team-ffn-75` | The original checkpoint retains 75% of the feed-forward intermediate width. |
| Pilot FFN 50% | `team-ffn-50` | The original checkpoint retains 50% of the feed-forward intermediate width. |

DSpark and its vLLM ModelRunnerV2 reference remain deferred. Their preparation history remains available in the [baseline records](baselines/README.md).

The pilot evaluation imports the teammate's original checkpoints. The supplied data contains 200 training examples and zero held-out examples. The notebook specifies one epoch and 50 optimizer updates per width. Import records preserve verified weight hashes, model dimensions, vocabulary, and available provenance. The original training seed, target revision, and complete training history remain unresolved.

## Evaluation procedure

The trial uses the first three prepared prompts from each benchmark in the saved input order. Every prompt contains one user turn. The [benchmark manifest](benchmarks/manifest.json) preserves source indices, prompt hashes, input-file hashes, and attribution paths.

| Task family | Benchmarks |
|---|---|
| Mathematics | GSM8K, MATH-500, and AIME 2025. |
| Code | HumanEval, MBPP, and LiveCodeBench. |
| Conversation and instruction following | MT-Bench, Alpaca, and Arena-Hard. |
| Summarization | CNN/DailyMail. |

The target is **Meta Llama-3.1-8B-Instruct**, revision `0e9e39f249a16976918f6564b8830bc894c89659`. Generation uses temperature zero, one active request, natural stopping, and a 2,048-token output limit. The prepared context limit is 3,328 tokens. Each configuration has three separate warmups and three timed repetitions of all 30 prompts.

The measurement client and model server run on the same compute node through its loopback connection. Python's monotonic nanosecond clock records submission and streamed arrivals. The browser displays saved observations. Browser refresh time lies outside the recorded intervals.

The five initial configurations and Pilot FFN 100% use one physical L40S on `ins061`. Pilot FFN 75%, Pilot FFN 50%, and their complete AR reference use another L40S on `ins060`. The 100% width comparison therefore describes observed runs across devices. Its active paired speedup and agreement cells remain empty. The two narrower widths have complete same-device AR comparisons.

### Measurements and aggregation

| Metric | Definition and purpose |
|---|---|
| Output tokens per second (OTPS) | Divide delivered output tokens by the complete three-prompt benchmark interval. Higher throughput means faster output delivery. |
| Paired speedup | Divide a method's OTPS by its matching AR OTPS for the same benchmark and repetition. Values above one show higher throughput. |
| Allocated GPU time per million output tokens | Multiply the benchmark interval by its allocated GPU count, divide by delivered tokens, and multiply by one million. Lower values show lower compute cost. |
| Mean acceptance length, τ | Divide committed draft and target tokens by verification rounds before final output trimming. Larger values indicate more progress per round. |
| Draft acceptance | Divide accepted draft tokens by proposed draft tokens. Position-level and tree-prefix measurements describe where acceptance ends. |
| Time to first token (TTFT) | Measure submission through the first received chunk containing text. A chunk can contain several tokens. |
| Completion and streaming latency | Measure complete request time and gaps between received text chunks. The report includes median and 95th-percentile summaries. |
| Exact output agreement | Compare delivered token IDs, output lengths, and stopping reasons with matching AR responses. |
| GPU memory and allocation totals | Measure eligible memory peaks and count each parent job's GPU-hours and CPU-hours once. |

Each benchmark repetition contains three sequential requests. The tables report the mean, sample standard deviation, and eligible repetition count. Paired speedup first forms one ratio per repetition, then summarizes those ratios. Three execution repetitions reuse the same inputs. Broader benchmark claims require a larger prompt sample.

Compute cost uses **GPU time per output token**. Its timed interval includes client gaps inside the benchmark block. Startup, warmups, and other allocation time have separate whole-job accounting. Per-prompt GPU-seconds equal request completion seconds multiplied by the allocated GPU count.

EAGLE-3 proposes 15 candidate tree nodes per round, including alternative branches. DFlash proposes nine linear tokens, P-EAGLE proposes seven, and the pilot proposes eight. These proposal structures give acceptance fractions different denominators. The report keeps EAGLE-3's tree-prefix diagnostic separate from linear-position acceptance.

## Measured findings

The following ranges cover the ten benchmark means. Each benchmark mean summarizes three timed repetitions. The report preserves every benchmark's value, standard deviation, and evidence.

| Method | Mean OTPS range, tokens/s | Paired mean speedup range |
|---|---:|---:|
| AR · SGLang | 42.88–43.44 | Reference. |
| EAGLE-3 | 94.66–116.71 | 2.189–2.690× |
| DFlash | 97.67–213.37 | 2.271–4.931× |
| AR · vLLM ModelRunnerV1 | 45.72–46.47 | Reference. |
| P-EAGLE | 82.75–157.94 | 1.782–3.411× |
| AR · Transformers (pilot) | 39.12–40.17 | Reference. |
| Pilot FFN 100% | 32.42–39.02 | The active reference uses another physical GPU. |
| Pilot FFN 75% | 31.92–39.02 | 0.798–0.971× |
| Pilot FFN 50% | 31.56–37.59 | 0.807–0.936× |

DFlash leads the five initial configurations in OTPS on six benchmarks. P-EAGLE leads on MATH-500 and AIME 2025. EAGLE-3 leads on MT-Bench and CNN/DailyMail. The timed GPU-cost ranking has the same per-benchmark leaders. DFlash is the strongest current released-method speed reference on these selected prompts and recorded outputs.

Narrowing provides no consistent observed speed or compute-cost advantage in the pilot. The 100% width has the highest observed throughput on six benchmarks, and 75% leads on four. The 50% width leads on none. Within the same-device comparison, 75% exceeds 50% on seven benchmarks. Both narrower widths are slower than their matched AR reference on all ten benchmarks. Pilot acceptance length remains approximately 1.1–1.25 tokens per round despite eight proposals.

Exact response preservation remains unresolved. EAGLE-3 matches its AR reference on 11 of 30 unique prompts, DFlash on 13, and P-EAGLE on 14. Each narrower pilot matches its reference on 12. Each configuration repeats its own responses consistently across three executions. Speed measurements describe these observed outputs, whose lengths also differ. The cause of the response differences remains unresolved. Benchmark task accuracy remains unmeasured.

EAGLE-3 has 90 saved diagnostic responses. DFlash has four saved diagnostic blocks containing 12 responses. Further diagnostics remain paused. The continuation memory sampler starts during AR evaluation, so the 75% and 50% device-memory cells remain empty. Every empty catalogue cell retains its evidence reason.

## Reproduce the experiment

The commands below describe a new primary measurement sequence. The commands require an allocated Insomnia L40S, the pinned environments, and accessible model weights. The completed observations remain preserved in their publication archive.

### Prepare inputs and runtimes

1. Clone this repository with its history and enter the checkout.

```bash
git clone https://github.com/einnuian/hpml-project.git
cd hpml-project
```

2. Follow the [Insomnia setup guide](baselines/insomnia.md) and prepare the [SGLang runtime](baselines/sglang-setup.md) and [patched vLLM runtime](baselines/hspec-setup.md).

SGLang uses release 0.5.19 at source revision `0bcd822377da7b5718e674eaf9c870d349424dd1` in the pinned CUDA 12.9 container. P-EAGLE uses patched vLLM 0.25.1 with ModelRunnerV1. The pilot runtime uses Python 3.10, PyTorch 2.6.0 with CUDA 12.4, and Transformers 4.57.6. The setup scripts record actual packages, driver, host, job, and GPU identity. Model access and checkpoint acquisition require the corresponding account permissions.

3. Inside the assigned compute node, set the project paths and identify its allocated GPU.

```bash
export HPML_ROOT="/insomnia001/depts/edu/users/$USER/hpml-baselines"
export HF_HOME="$HPML_ROOT/cache/huggingface"
export HPML_PREPARED="$PWD/benchmarks/inputs/trial-tokenized.json"
export HPML_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
export HPML_REPLAY_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export HPML_TRIAL="$HPML_ROOT/results/replay-$HPML_REPLAY_STAMP/trial-first30-$(date -u +%Y%m%d)"
python3 baselines/preflight.py --mode gpu
python3 baselines/trial_recorder.py validate benchmarks/inputs/trial-first-three.jsonl
```

The GPU variable must identify the single L40S assigned to the allocation. The preflight and runtime checks verify the compute-node context. The prepared bundle preserves exact input token IDs and stopping rules. Its SHA-256 is `8d3302b67672b22bda85c929b464c56c8935d823609776fa96da3d091cfe6e1e`.

### Measure the released methods

1. In a second shell on the same allocated node, set the same trial path and GPU UUID, then start the passive memory recorder.

```bash
python3 baselines/trial_resources.py monitor \
  --device "$HPML_GPU_UUID" \
  --output "$HPML_TRIAL/gpu-memory.jsonl"
```

2. In the main shell, run the five configurations sequentially.

```bash
python3 baselines/trial_sequence.py \
  --prepared "$HPML_PREPARED" \
  --output "$HPML_TRIAL" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --configurations ar-sglang eagle3 dflash ar-vllm-v1 peagle \
  --run
```

The controller starts one server, checks one-token and natural-stop streaming, records warmups and 90 timed responses, then stops that server. Saved records preserve token IDs, text, event times, engine settings, and native acceptance counts. A repeated invocation validates completed configurations and preserves incomplete attempts before retrying them.

3. After inference finishes, stop the memory recorder with **Control+C** and preserve its completed log.
4. After the allocation ends, save its final Slurm record.

```bash
python3 baselines/trial_resources.py allocation "$HPML_JOB_ID" \
  --output "$HPML_TRIAL/allocation.json"
```

Set `HPML_JOB_ID` to the allocation's recorded job ID. The [measurement runbook](baselines/measurement-runbook.md) explains the clocks, manual prompt interface, and project-wide resource accounting.

### Measure the original pilot checkpoints

1. In an allocated compute node with the prepared target cache, create a fresh pilot run and prepare its runtime.

```bash
export HPML_TEAM_RUN="$HPML_ROOT/runs/pilot-replay-$HPML_REPLAY_STAMP"
export HPML_TEAM_SHARED="/insomnia001/depts/edu/users/ahn2130/peagle_ffn_pilot"
bash baselines/setup_team_peagle.sh
export HPML_TEAM_PYTHON="$HPML_ROOT/envs/team-peagle-cu124-py310/bin/python"
```

2. Run the primary pilot sequence with its matching AR reference.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_sequence.py \
  --output "$HPML_TEAM_RUN" \
  --prepared "$HPML_PREPARED" \
  --hf-home "$HF_HOME" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --import-from "$HPML_TEAM_SHARED" \
  --configurations team-ar team-ffn-100 team-ffn-75 team-ffn-50 \
  --run
```

The shared directory must contain the three original checkpoints and `train_data.jsonl`. The controller exports the notebook from pinned commit `64b24c42f1859e25b0ffb69293e7e9f793a697d3`. It checks checkpoint hashes, vocabulary, model dimensions, input overlap, and source-equivalent generation before primary measurement. The default phase is primary. The command preserves the imported weights and evaluates the four configurations sequentially on the assigned GPU. The [pilot runbook](baselines/team-peagle-runbook.md) documents individual adapter commands and optional future training.

### Rebuild tables from saved observations

1. Download and extract the final recording ZIP from the project hub or final W&B artifact.
2. Set `HPML_SAVED_TRIAL` to the extracted `trial-first30-20261122` directory and choose a fresh export directory.

```bash
export HPML_EXPORT="$(mktemp -d)/trial-export"
python3 baselines/trial_export.py \
  --trial "$HPML_SAVED_TRIAL" \
  --catalogue baselines/metric-catalogue.ts \
  --output "$HPML_EXPORT" \
  --memory-samples gpu-memory-snapshot-20261123T013545Z.jsonl \
  --additional-memory-samples gpu-memory-snapshot-20261123T015408Z.jsonl \
  --report-output measurements/report/generated
```

The catalogue snapshot makes this command independent of the Site checkout. The exporter validates evidence, preserves unavailable-cell reasons, and regenerates the five report fragments. The [report procedure](measurements/report/README.md) covers LaTeX compilation, Site packaging, and W&B publication. The historical partial AR attempt remains available as a separate archive.

## Repository contents

| Path | Purpose |
|---|---|
| `baselines/` | The directory contains setup scripts, servers, recorders, analysis, publication tools, and technical guides. |
| `benchmarks/` | The directory preserves the selected prompts, tokenized bundle, source catalogues, and attribution notices. |
| `tests/` | The tests check streaming, stopping, pairing, acceptance, accounting, and publication validation. |
| `measurements/report/` | The directory contains the editable report, generated tables, and reviewed PDF. |
| `measurements/results/` | The tracked selection preserves the final catalogue, evidence hashes, configuration summaries, and allocation records. |
| `peagle_ffn.ipynb` | The teammate's notebook preserves the original pilot implementation. |
| `peagle_ffn_insomnia_run.html` | The saved notebook output preserves the earlier pilot observations. |

Raw response archives, checkpoints, model caches, local environments, and generated build files have separate storage. The project hub and W&B artifact preserve the full recording archive. The original proposal remains in the surrounding project's `proposal` directory.

## Next research steps

1. Verify response preservation and establish a stronger full-width drafter with documented training provenance.
2. Compare conditioning, layer count, and proposal length under fixed training and evaluation settings.
3. Repeat the 100% and 75% comparison on one physical GPU, then test whether improved acceptance offsets drafting and verification costs.
4. Measure speed and GPU time per output token against matched AR, EAGLE-3, DFlash, and P-EAGLE.
5. Expand the held-out prompt sample and training seeds, and add task-appropriate accuracy scoring.

These steps target a measurable speed and compute-cost advantage over ordinary generation and the released methods. New optimization experiments remain planned. The completed trial supplies their reference observations.
