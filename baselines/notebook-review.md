# Saved P-EAGLE notebook: cell-by-cell review

The saved notebook records a custom drafter training pilot on one NVIDIA L40S. It builds three independently trained drafters, then measures isolated greedy generation. Its output shows lower draft time at narrower FFN widths, a small change in full cycle time, and token divergence on two of four prompts at every width. Released pretrained checkpoints and exact-output qualification provide the starting point for the current inference-only study.

The review covers all 33 cells in [peagle_ffn.ipynb](../peagle_ffn.ipynb), all saved outputs in [peagle_ffn_insomnia_run.html](../peagle_ffn_insomnia_run.html), and the historical [EAGLE setup guide](../SETUP_EAGLE.md) and [notebook setup guide](../SETUP_NOTEBOOK.md). The notebook and HTML belong to team source commit `c8af4638d2b7ba046765a31425be5e7fda80dc6f`. Cell numbers here count every Markdown and code cell from the top, starting at one. The review reads the saved work; it executes no notebook cell.

## What the recorded run establishes

| Item | Recorded evidence | Interpretation |
|---|---|---|
| Hardware | Cell 5 prints `cuda NVIDIA L40S`. | The recorded measurements describe an L40S run. Cell 1's A100 durations describe the notebook author's estimates for another setting. |
| Runtime | Cell 4 prints a teammate's `ea_env` interpreter; cell 5 asserts Transformers 4.57.6. | The notebook environment differs from the newly prepared EAGLE inference environment, which uses Transformers 4.53.1. |
| Data | Cell 13 reports 200 examples and mean answer length 434 tokens. | These are pilot settings. The source permits 3,000 examples in its full setting. |
| Draft vocabulary | Cell 15 reports 12,048 tokens and 100% coverage of the saved answers. | The configured 32,000 is an upper limit on frequent tokens. Coverage describes those training answers. |
| Drafter widths | Cells 7 and 17 specify FFN dimensions 14,336, 10,752 and 7,168. | The experiment changes the actual intermediate dimensions of newly constructed drafters. |
| Training | Cell 21 reports 50 AdamW updates for each width. | Each width has separately learned weights. The current study uses already released weights. |
| Internal view check | Cell 25 reports 8/8 matching guesses for one example/anchor at each width. | That example agrees across the full-sequence and cached drafting paths. |
| Output check | Cell 28 reports `same, differs at 67, differs at 6, same` for all widths. | Exact greedy token identity passes on two of four prompts. The recorded check identifies divergence positions and leaves their cause unresolved. |
| Timing | Cell 30 reports 41.4 output tokens/s for ordinary generation and 38.3, 38.3 and 38.0 for the three custom drafters. | These saved custom configurations take longer per returned token than the ordinary path on this small prompt set. |

The HTML contains the same 33 cells. Eighteen of its 19 code-cell sources match the notebook exactly after trimming surrounding whitespace. Cell 3 differs: the HTML retains an executable `!pip install` command, while the notebook contains terminal-install comments. Execution counts also repeat and appear out of order. These artifacts preserve results from an interactive session; a clean sequential execution from the current saved source remains unestablished.

The preserved file hashes are:

- Notebook SHA-256: `f86753deccaee877ab34eeada6a5b45a8de3d818366a8909613f7fd61cf7b8e2`.
- HTML SHA-256: `735616a42557753f63d7f57c63c3bfc3f375a16fea9ffc27cbb73b5890da3f37`.

## How the custom drafter relates to the released P-EAGLE reproduction

A feed-forward network (FFN) transforms a token's feature vector through an intermediate set of channels and projects the result back to the model width. The notebook uses a gated FFN: `down(silu(gate(h)) * up(h))`. Changing the intermediate dimension changes the shapes of all three FFN matrices. The notebook keeps four decoder layers at every width.

The prepared baseline set includes the [H-Spec authors' released Llama P-EAGLE reproduction](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/tree/0db2ae706f7a314286516835006183efdcb5bb39), alongside EAGLE-3, DFlash, H-Spec, PARD-2 and DSpark. Its [configuration](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/blob/0db2ae706f7a314286516835006183efdcb5bb39/config.json) and the pinned [Speculators P-EAGLE implementation](https://github.com/vllm-project/speculators/blob/3251ed020cfc954717c1d075d83dca8721713f0c/src/speculators/models/peagle/core.py) identify the specific released system available for comparison.

| Component | Saved notebook | Released reproduction |
|---|---|---|
| Weight origin | `Drafter(width)` initializes new parameters; cell 21 trains every width. | The prepared inference path loads the completed pretrained checkpoint and keeps its weights fixed. |
| Target conditioning | Each real slot uses the target's final hidden vector, of width 4,096, and the next token embedding. | The released configuration requests target feature layers `[1,8,15,22,30]` and uses its released feature-combination path. |
| Input projection | The notebook normalizes and concatenates one target feature with a token embedding, then projects `2D` to `D`. | Speculators first projects concatenated target features and constructs its EAGLE-derived layer input with the token embedding. The released implementation defines the compatible dimensions and normalization. |
| Placeholder representation | The notebook learns a 4,096-dimensional `h_shared` and a separate `e_mask`. | The release uses its configured mask token and learned feature-mask representation. Those tensors come from the checkpoint. |
| Vocabulary | The pilot constructs 12,048 entries from its own answer frequencies, then clones selected rows of the target's output head. | The released checkpoint specifies a 32,000-entry draft vocabulary and its own mappings/head weights. Loading and preserving that mapping is part of qualification. |
| Attention/cache | The notebook implements causal attention, manual key/value concatenation and cropping, repeated key/value heads, and a local rotary-position function. | The released P-EAGLE route uses the pinned patched vLLM runner and its cache/attention implementation. Its configuration also includes a 2,048-token sliding window. |
| Proposal count | Each notebook cycle proposes eight new tokens and can append a ninth target token. | The documented H-Spec reproduction launch uses seven speculative positions. Each implementation needs its proposal-count convention recorded explicitly. |
| Serving | The notebook executes one prompt at a time in a Python loop. | The baseline study also evaluates streaming and predetermined concurrent arrivals through a qualified server. |

The notebook's table labels use “P-EAGLE” for its custom construction. Preserve that wording when quoting the saved output and label the configuration as the custom notebook drafter in new comparisons. Shared architecture ideas establish research relevance; checkpoint, feature, vocabulary and implementation identities establish a specific baseline.

## Cells 1–11: scope, settings and target

### Cell 1 — Markdown: question and five-step outline

The cell asks whether 75% and 50% FFN widths retain enough accepted tokens to remain fast. It outlines target-generated data, vocabulary selection, training, correctness checks and timing. The stated A100 durations and full settings provide the author's planned workflow. The saved pilot outputs supply the observed run conditions. The current work establishes the performance of unchanged released methods under shared conditions.

### Cell 2 — Markdown: cluster and authentication instructions

The cell requests a compute node, selects the historical `ea_env` kernel and introduces model access. Its generic `--gres=gpu:1` leaves GPU type open. Its eight-CPU/64G example also needs reconciliation with the currently observed `short` partition memory allowance. The current [Insomnia instructions](insomnia.md) specify an explicit GPU type, a sufficient CPU allocation and private scratch storage. Authentication belongs to the existing personal model-hub sign-in.

### Cell 3 — Code: package preparation

The notebook now contains comments for installing Transformers 4.57.6, datasets 3.6.0, Hugging Face Hub below 1.0, W&B and Jupyter tools. Its saved output contains a pip upgrade notice. The HTML's corresponding cell contains `!pip install`, which explains why the current commented source and saved output differ. The historical notebook package set stays separate from released-method environments; the current setup scripts preserve each method's selected dependencies.

### Cell 4 — Code: interpreter identity

`sys.executable` prints the interpreter used by the session. The saved path belongs to the teammate's home-directory `ea_env`. This identifies the Python environment location; a complete environment record also needs package versions, code revisions and CUDA details. The new setup records supply those fields for the prepared baseline environments.

### Cell 5 — Code: imports, allocator configuration and device selection

The cell enables expandable CUDA allocator segments, imports the numerical/model libraries, asserts Transformers 4.57.6 and selects CUDA when available. The output identifies an NVIDIA L40S. The CPU fallback makes the cell executable in another environment, while the saved performance result specifically concerns CUDA on the L40S. The cell records no exact PyTorch, driver or attention-backend version.

### Cell 6 — Markdown: pilot setting explanation

The cell introduces `PILOT=True` as the initial setting. Its role is explanatory. The next cell determines the actual data, update and evaluation sizes. The source's instruction to run the historical pilot remains part of the artifact being reviewed.

### Cell 7 — Code: experiment constants

The cell selects Llama-3.1-8B-Instruct, eight draft proposals, widths `[1.0,0.75,0.5]`, a 32,000-token vocabulary limit, 200 pilot training prompts, one epoch, four-example gradient accumulation, four evaluation prompts and 128 output tokens per evaluation request. The pilot therefore schedules `200/4=50` optimizer updates. It enables online W&B logging and defaults `OUT` to a home-directory pilot folder. The target and datasets use names without fixed revisions. A reproducible pretrained run records exact revisions, prompt IDs, generation settings and scratch paths.

### Cell 8 — Code: sign-in and output directory

The cell prompts for Hugging Face authentication, creates `OUT`, and signs in to W&B when enabled. The saved W&B output records an existing teammate account. This cell performs account and filesystem operations; it produces no model-quality or speed measurement. The current baseline preparation uses private scratch caches and existing personal authorization. The current [W&B tracking setup](tracking.md) records released-method setup and correctness evidence locally, then supports synchronization to your verified account.

### Cell 9 — Markdown: frozen-target interface

The cell defines a target feature vector, vocabulary logits and a key/value cache. A key/value cache stores attention state from prior positions so later forwards process only new input positions. The explanation identifies a 4,096-dimensional final feature. That feature choice is an architectural input to the custom drafter. The released P-EAGLE reproduction collects features from the multiple layers identified above.

### Cell 10 — Code: repeated Hugging Face login

The cell calls `login()` again. It introduces no new model or configuration. Repeated authentication is unnecessary once the intended cached sign-in is established. The review preserves the source and carries no credentials into the report.

### Cell 11 — Code: target loading and ordinary forward helper

The cell loads the target and tokenizer by repository name, moves the target to BF16 on the selected device, sets evaluation mode and freezes its parameters. The recorded dimensions are hidden width 4,096, FFN width 14,336, 32 attention heads and eight key/value heads. The target's generation configuration supplies stop IDs `{128008,128001,128009}`.

`target_step` calls the target backbone, applies the target output head and returns features, float32 logits and the cache. `chat_ids` uses the tokenizer's user-only chat template with a generation prompt. The implementation uses one target for ordinary generation and speculative verification. Exact target revision, template bytes, dtype, backend and stop policy still need a fixed record before comparing implementations.

## Cells 12–21: generated data and custom drafter training

### Cell 12 — Markdown: target-generated answers

The cell explains why the target writes the answer tokens used to teach the custom drafter. It identifies UltraChat first user turns and resumable JSONL storage. This is the historical data-preparation rationale. The current benchmark starts with released weights and a shared, recorded evaluation prompt set.

### Cell 13 — Code: create or reuse answer data

The cell shuffles UltraChat with seed zero, filters prompt length, left-pads batches and asks the target for greedy answers up to 512 tokens. It stores each prompt's token IDs and answer IDs through the first EOS. The saved output reports 200 examples with mean answer length 434 tokens.

Resume behavior checks the number of rows already saved. It does not verify that existing rows share the current target revision, template, generation configuration or dataset revision. The saved result therefore documents the reused data file's observed contents at a coarse level. New pretrained benchmarks require their own frozen prompt record and avoid this training-data generation path.

### Cell 14 — Markdown: draft vocabulary rationale

The cell describes a frequency-selected draft vocabulary and a frozen subset of target output-head rows. The numerical explanation uses a configured 32,000-entry vocabulary. The following cell reports the actual pilot vocabulary, which is smaller because the 200 answers contain fewer distinct tokens.

### Cell 15 — Code: vocabulary mappings and output-head subset

The cell counts answer-token frequency, selects up to 32,000 token IDs and adds missing EOS IDs. `D2T` maps draft IDs to target IDs; `T2D` maps target IDs back to draft IDs and uses `-1` for absent tokens. `HEAD` clones the corresponding frozen target output rows. The saved 12,048-entry vocabulary covers every token in those saved training answers. That coverage provides no direct coverage measurement for new prompts or a released checkpoint's vocabulary.

### Cell 16 — Markdown: real slots and placeholders

The cell defines a real slot as target feature `i` plus the embedding of token `i+1`, with a prediction for token `i+2`. Placeholder slots supply learned vectors for later positions. Four decoder layers process all slots together. The statement “Only `ffn` changes” describes the constructor's architectural parameter; the subsequent runs also learn separate parameter values for each width.

### Cell 17 — Code: custom four-layer drafter

The cell implements rotary positions, RMS normalization, grouped key/value attention, a gated FFN, feature/embedding projection, placeholders and the output projection. Width rounding produces intermediate dimensions 14,336, 10,752 and 7,168. The saved counts are approximately 906M, 730M and 554M trainable parameters. Those counts exclude the shared frozen target embedding and cloned frozen vocabulary head.

`Drafter(width)` initializes its parameters afresh. The different matrix shapes consume different random-number sequences, and later updates alter each model independently. Identical initial values in shared channels would require a direct tensor comparison across the three models. The saved width comparison therefore reflects the shapes, independently learned weights and runtime behavior of three custom drafters.

### Cell 18 — Markdown: full-sequence and cached views

The cell describes the attention mask for real slots and groups of placeholders, then explains cache reuse during decoding. The full-sequence view permits each placeholder group to attend to the context for its own anchor. The cached view appends current real slots and drops placeholders after the draft pass. Agreement between these views is a useful internal implementation check.

### Cell 19 — Code: masks, labels and cached drafting

`training_view` builds real slots plus up to 64 groups of seven placeholders. It constructs a dense Boolean attention mask, position IDs and target-token labels, then maps those labels into the draft vocabulary. The source masks prompt positions and unavailable target tokens out of the training loss.

The cache helpers store four layers of keys and values. `draft` appends real slots and seven placeholders, runs the drafter, removes placeholder cache entries and returns eight target-vocabulary guesses. This is the custom inference implementation that cells 25–30 inspect and time. Its Python concatenation, attention mask and vocabulary projection are part of the measured system.

### Cell 20 — Markdown: training procedure

The cell introduces cross-entropy learning from the target's answer tokens, shared data order and periodic checkpoints. Those controls describe the historical learning experiment. The current study treats learned released checkpoints as inputs and contains no optimizer-update stage.

### Cell 21 — Code: train each width

The cell creates an AdamW optimizer, samples anchor positions, computes the target's frozen features, backpropagates the custom drafter loss, clips gradients and writes checkpoints containing both drafter and optimizer state. The saved pilot performs 50 updates at each width. Final reported loss/first-slot accuracy/placeholder accuracy are 6.681/0.21/0.06 at full width, 6.691/0.20/0.05 at 75%, and 6.798/0.19/0.05 at 50%.

The recorded training peaks are 33.9 GB, 41.4 GB and 41.4 GB. The cell uses `torch.cuda.max_memory_allocated()` without resetting the peak for each width, so later values can retain an earlier peak. They describe training-session allocation and do not supply isolated inference memory or cost. The current workflow retains this cell's training and checkpoint recovery as historical context. Its new W&B records cover pretrained setup, correctness checks and future inference measurements.

## Cells 22–28: internal and target-output checks

### Cell 22 — Markdown: correctness expectations

The cell introduces agreement between full-sequence and cached drafting, then agreement between speculative and ordinary target output. It attributes a “rare late difference” to BF16 rounding. The saved checks establish the locations of differences. Establishing a numerical cause requires a controlled comparison of target logits and numerical paths, alongside cache, mask, position and stop handling. The current study records every mismatch and resolves its cause before qualifying an exact-output configuration.

### Cell 23 — Code: one memory reading

The cell prints current PyTorch allocated bytes divided by one billion. The saved value is 20.539515904 GB. It is a single current allocation reading, with the live session's objects still present. A per-method peak requires a reset and a complete observation interval. A target-only memory control requires a process containing the target and its required runtime state.

### Cell 24 — Code: load all three custom drafters

`load_drafter` constructs a new model, loads the complete saved checkpoint with `map_location=DEV`, copies the drafter tensors, converts to BF16 and selects evaluation mode. The dictionary keeps all three drafters resident. The temporary checkpoint includes optimizer state, and construction begins with FP32 parameters. These details explain why loading can create a transient memory peak. The historical notebook guide describes a CPU-loading variant. Released baseline qualification loads one selected drafter with its target in the corresponding isolated process.

### Cell 25 — Code: compare full-sequence and cached guesses

The cell selects the first saved data example and one anchor ten tokens into its answer. It compares eight guesses from a full-sequence pass with guesses from two cached chunks. The output reports 8/8 agreement for every width. The printed `OK` condition is `same >= K-1`, so the source would also print `OK` for seven matches. An exact check should require the specified complete equality, and qualification needs multiple positions and cache lengths.

### Cell 26 — Markdown: speculative cycle

The cell describes drafting eight tokens, verifying the previous committed token plus those proposals, and committing an accepted prefix plus a target token. The extra target token is a correction after a rejection or a bonus after full acceptance. Those categories need separate counts when interpreting acceptance and output progress.

### Cell 27 — Code: ordinary and speculative generation

`standard` obtains a first target token from prefill, then generates one target token per loop iteration until the output cap or EOS. `speculative` also starts with target prefill, uses the custom drafter, verifies an eight-token proposal block, accepts matching prefix tokens and appends the target's next choice. It crops the target cache and updates the drafter's real-slot inputs for the next cycle.

The speculative loop records accepted proposals before final output clipping. Its loop continuation checks the most recently appended token. An EOS inside a longer accepted block can therefore coexist with later appended tokens, and the loop can perform additional work before the final returned list is clipped at the first EOS. The function finally enforces `max_new` on its returned list. Exact returned output length, pre-clipping work and accepted-token counters therefore need separate interpretation.

The optional `clock` synchronizes CUDA around draft, verify and other work. This supports diagnosis while adding synchronization overhead. The first prefill lies outside those component intervals. A full request timer and separate profiles are required to connect cycle measurements to response latency.

### Cell 28 — Code: compare target outputs on four prompts

The cell loads the first four MT-Bench prompts, keeps each prompt's first turn, computes ordinary references and compares speculative output lists. All widths produce the same saved pattern: equality on prompts one and four; first divergence at zero-based positions 67 and 6 on prompts two and three. The cell prints the beginning of the ordinary answer for prompt one.

The comparison checks list length as well as token pairs. A pure length mismatch would print `differs at None`, which needs a clearer classification in a reusable harness. The source records the first mismatch, while a qualification record also needs complete token IDs, prompt identity, target logits around the divergence, stop reason and configuration. The shared divergence pattern across widths is useful evidence for investigation; it establishes no unique cause.

## Cells 29–33: timing and interpretation

### Cell 29 — Markdown: timing protocol and column definitions

The cell specifies two warmups, three timed passes and one separate component-profile pass. It defines `C` as one plus mean accepted guesses and `a1` through `a8` as the fraction of cycles that accept an entire prefix. Those `a` values are prefix-survival probabilities. Conditional acceptance at position `j` uses cycles whose earlier positions passed as its denominator. `C` describes committed progress before the function's final output clipping.

### Cell 30 — Code: measure isolated generation

`tokens_per_second` runs two warmups on the first prompt. For each of three repeats, it times all four prompts sequentially, sums final returned list lengths and divides by the full repeat's elapsed time. Each repeat includes target prefill and Python control work. The table reports the arithmetic mean of those three repeat rates, and speedup divides each mean by the ordinary mean.

The `spread` field is maximum repeat rate minus minimum repeat rate. The printed `±` label therefore denotes a range, and the displayed 0.0 reflects rounding to one decimal place. It supplies no confidence interval or evidence of zero run-to-run variation. The accepted-token statistics pool cycles across three repeats, so requests with more cycles contribute more observations.

The component-profile pass runs separately and divides each component's accumulated duration by its cycle count. Every ordinary and speculative timing still shares a process containing all three loaded drafters. That resident memory affects available cache capacity. The notebook records no client TTFT, streaming chunks, concurrent arrivals, per-request completion-latency distribution, allocated GPU time or actual GPU charges.

### Cell 31 — Markdown: cycle-time equation

The cell interprets time per token as `(draft + verify + other) / C` and compares relative cycle-time reduction with relative progress loss. The equation is an informative cycle-level model when numerator and denominator cover the same cycles and output boundary. Complete-request inference additionally includes prefill, final clipping, scheduler behavior and transport. The current [metrics definitions](metrics.md) keep those boundaries explicit.

### Cell 32 — Code: calculate relative changes

The cell sums the three profiled component means, compares each width against the full-width custom drafter and predicts `1,000 × C / cycle_ms`. Its saved output is:

| Custom notebook drafter | Profiled cycle, ms | Cycle change | Draft-time change | `C` change | Predicted tokens/s |
|---|---:|---:|---:|---:|---:|
| 100% width | 31.42 | 0.0% | 0.0% | 0.0% | 37.5 |
| 75% width | 31.06 | −1.2% | −8.7% | −0.2% | 37.9 |
| 50% width | 30.90 | −1.7% | −12.1% | −1.1% | 37.7 |

Verification accounts for about 85% of the full-width profiled cycle; drafting accounts for about 14%. The narrow widths therefore affect a smaller share of total cycle time. The predictions and measured request-level rates have different timing/count boundaries. The saved output supports component profiling and a direct test of returned tokens per elapsed second. It supplies no improvement over the recorded ordinary target rate. The cell also uploads a W&B table when logging is enabled.

### Cell 33 — Markdown: requested outputs

The final cell asks for the throughput table, the two correctness checks and the relative-cycle calculation. It links conceptually to a separate guide. The current report identifies the saved evidence directly and adds the missing qualification boundaries. The next work follows the current pretrained baseline plan.

## Use the notebook's evidence in the shared baseline comparison

1. Preserve the notebook and HTML as historical source artifacts, including their differing cell 3 and recorded output mismatches.
2. Review the completed checkpoint, environment and four-prompt qualification records in [the setup report](setup-report.md). Keep the canonical target's tokenizer, generation configuration and stop rules with its pinned weights.
3. Read the [method explanations](https://hpml-workspace.ayomideadetayo.chatgpt.site/tutorials#methods-overview) and [paper measurement procedures](https://hpml-workspace.ayomideadetayo.chatgpt.site/tutorials#papers-overview). Identify each released checkpoint, runtime, original metric and ordinary control before preparing the common comparison.
4. Agree on the shared prompts, generation settings, arrival traces and [metric definitions](metrics.md). Record native generation, final returned output and stop reasons separately. If inference settings require selection, declare development and reserved prompt sets before choosing those settings.
5. Qualify the complete workload on the selected physical GPU, loading one configuration at a time. Save full greedy token IDs and investigate every mismatch before interpreting an exact-output speed result.
6. Measure complete requests and returned tokens for each unchanged released method and its matched ordinary control. Collect component profiles in separate runs. Use a clean target-only process for ordinary memory/resource accounting.
7. Summarize the measured configurations by workload and load level, including output agreement, failed requests, latency, throughput, GPU allocation time and memory. Record actual charges when attributable billing evidence becomes available.

The notebook's training-data creation, vocabulary fitting, optimizer updates and recovery checkpoints remain descriptions of the saved pilot. The current study uses existing pretrained artifacts throughout.
