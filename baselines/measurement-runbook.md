# Run and inspect the benchmark trial

The Mac holds the source, prepared prompts, and saved results. Insomnia runs the model server and measurement client on the same allocated compute node. The client connects through that node's loopback address. Safari shows the prompts, responses, and measurements. Terminal logs preserve the detailed execution record.

The five initial configurations have completed 450 timed requests. The configurations are AR SGLang, EAGLE-3, DFlash, AR vLLM ModelRunnerV1, and the H-Spec P-EAGLE reproduction. All five passed streaming and stopping qualification. EAGLE-3 completed its 90 diagnostic requests. DFlash saved four diagnostic benchmark blocks before the sequence paused. The pilot evaluation adds 360 completed responses, bringing the total to 810. Further diagnostics remain paused. DSpark and its vLLM ModelRunnerV2 reference remain deferred.

Exact output agreement remains unresolved. EAGLE-3 matches its AR reference on 11 of 30 unique prompts. DFlash matches on 13, and P-EAGLE matches on 14. Each configuration repeats its own outputs consistently across three repetitions. The speed measurements describe those recorded responses. Benchmark task accuracy requires a separate scoring procedure.

## Use the fixed inputs

The trial contains the first three prepared prompts from each of ten benchmarks: GSM8K, MATH-500, AIME 2025, HumanEval, MBPP, LiveCodeBench, MT-Bench, Alpaca, Arena-Hard, and CNN/DailyMail. Every prompt contains one user turn. The records preserve each source index, exact message, and input token sequence.

The target is Meta Llama-3.1-8B-Instruct at revision `0e9e39f249a16976918f6564b8830bc894c89659`. The longest input contains 1,010 tokens. Each initial server uses a context limit of 3,328 tokens. Generation uses temperature zero, natural stopping, and a limit of 2,048 new tokens. Each configuration serves one request at a time, runs three separate warmups, and completes three timed repetitions.

The local prepared file is `../benchmarks/inputs/trial-tokenized.json`, relative to `hpml-project`. The cluster copy is `$HPML_ROOT/source/benchmark-inputs/trial-tokenized.json`. Its SHA-256 is `8d3302b67672b22bda85c929b464c56c8935d823609776fa96da3d091cfe6e1e`.

## Follow saved results in Safari

1. From the local `hpml-project` directory, start the read-only trial viewer.

```bash
python3 baselines/trial_view.py \
  --prepared ../benchmarks/inputs/trial-tokenized.json \
  --results measurements/trial-first30-20261122 \
  --port 31101
```

2. In a fresh Safari window, open `http://127.0.0.1:31101` and the [project hub](https://hpml-workspace.ayomideadetayo.chatgpt.site/).
3. On the local overview, open each prompt's link to inspect its text, submission status, responses, and measurements.

The viewer prints all 30 prompt URLs. Full response text appears when its three-request benchmark block closes. A browser refresh reads saved files and sends no inference request. The viewer runs on the Mac. Long-running servers and measurement programs run inside a Slurm compute allocation.

Terminal windows remain available for inspection. The execution controller can also save process logs while Safari provides the visual progress view. The published Site preserves completed recordings after the local viewer closes.

## Enter your own prompts

1. From the local `hpml-project` directory, open the prompt interface in preview mode.

```bash
.venv-playground/bin/python baselines/prompt_playground.py \
  --preview \
  --prepared ../benchmarks/inputs/trial-tokenized.json \
  --usage measurements/trial-first30-20261122/compute-usage.json
```

2. In Safari, open `http://127.0.0.1:7860`.
3. On **Trial prompts**, select a prompt to read its exact text.
4. On **Cluster compute**, inspect the saved allocation totals and observation time.

To submit your own prompt, start the selected model inside an allocated compute node. Forward its loopback port 31000 to Mac port 31100. The SSH forwarding connection must terminate on the compute node that hosts the server. Use the node and connection established for that allocation.

5. After the forwarding connection is ready, stop preview mode with **Control+C** and start the connected interface.

```bash
.venv-playground/bin/python baselines/prompt_playground.py \
  --configuration eagle3 \
  --model-url http://127.0.0.1:31100/v1 \
  --prepared ../benchmarks/inputs/trial-tokenized.json \
  --usage measurements/trial-first30-20261122/compute-usage.json
```

6. In the browser, submit a message and read the streamed response.

Replace `eagle3` with the running initial configuration. The interface checks that the server exposes `hpml-target`. Manual prompts use one user message, greedy decoding, and the shared stop tokens. The interface prints manual prompts and responses in its log. The benchmark recorder separately creates the formal trial records. Run manual tests after a timed block has finished, and preserve their log when recording compute usage.

The interface adapts vLLM's repository example at `external/vllm/examples/applications/chatbot/gradio_openai_chatbot_webserver.py`. Its dependencies are recorded in `baselines/playground-requirements.txt`.

## Understand the automatic clock

The recorder reads Python's `time.perf_counter_ns()` immediately before request submission. The recorder reads the same clock when each complete streaming event arrives, before interpreting its JSON. The first event containing generated text establishes the first-output timestamp. The `[DONE]` event establishes completion.

Time to first token (TTFT) subtracts submission time from the first-text timestamp. A request submitted at 10.000 seconds whose first text arrives at 10.420 seconds has a TTFT of 420 milliseconds. A streamed chunk can contain several model tokens. TTFT describes the first text delivered to this client. Browser refresh time lies outside this interval.

Output tokens per second (OTPS) divides all delivered tokens in a three-prompt benchmark block by its complete elapsed time. That interval runs from the first submission to the final completion and includes client-side gaps. Each benchmark has three block repetitions. The tables report the mean of the three estimates and their sample standard deviation. Paired speedup first divides each method's OTPS by the matching AR block's OTPS, then summarizes the three ratios.

## Read the metric collection map

| Metric | Measurement procedure |
|---|---|
| Exact output agreement with AR | The exporter compares delivered token IDs, output length, and stopping reason. Native IDs and raw events preserve the stream evidence. |
| End-to-end latency and TTFT | The recorder subtracts the corresponding monotonic timestamps. The exporter calculates each repetition's median and p95. |
| Request output rate and TPOT | Request output rate divides tokens by complete latency. TPOT divides latency after first text by the output count minus one, for responses with at least two tokens. |
| Inter-chunk latency | The recorder measures gaps between nonempty text chunks. The catalogue separates the pooled p95 from the p95 of each request's longest gap. |
| Draft-token acceptance rate | Accepted draft tokens are divided by proposed draft tokens. EAGLE-3 proposes 15 candidate tree nodes per round. DFlash proposes nine linear tokens, and P-EAGLE proposes seven. Alternative tree branches contribute to EAGLE-3's denominator. |
| Accepted draft tokens per round | Accepted drafts are divided by all verification rounds, including rounds with zero accepted proposals. |
| Mean acceptance length, τ | Committed draft and target tokens are divided by verification rounds. Counts precede final output trimming. Validated primary SGLang counters supply this measure. P-EAGLE requires retained-target evidence from diagnostic rounds. |
| Position and tree-depth acceptance | Linear proposals use their position and prefix counts. EAGLE-3 uses a separate tree-prefix diagnostic with validated paths and eligible-depth counts. |
| Sampled GPU memory | The NVIDIA monitor samples the assigned GPU every 100 milliseconds. The exporter selects samples within each timed benchmark interval. |
| Framework memory | Diagnostic hooks reset and read allocated and reserved memory peaks for each request. The exporter requires coverage of all three requests in a block. |
| Draft and verification latency | CUDA events measure the named GPU operations during separate diagnostic executions. |
| Full-round latency | Host-clock boundaries cover drafting through the scheduler's final state update. The exporter joins worker and scheduler observations for the same request and round. |
| Allocation time and compute cost | Whole-job GPU-hours and CPU-core-hours count each parent allocation once. Timed GPU-seconds per output token use each benchmark interval and its final allocation count. Peak memory has its own measured scope. |

Percentiles use linear interpolation. Three prompts provide a small procedure check for each benchmark. The evidence distinguishes that prompt sample from its three execution repetitions.

## Resume diagnostic measurements when required

The five initial configurations share allocation `14056875` and its L40S GPU. The first diagnostic startup stopped before prompt submission because storage was full. The subsequent pass completed EAGLE-3 and saved partial DFlash results. Further diagnostic execution is paused. The commands below support a later resumption. Primary result directories remain the source of throughput and response-time measurements.

1. On the assigned compute node, enter the cluster project checkout and set the trial paths.

```bash
export HPML_TRIAL="$HPML_ROOT/results/trial-first30-20261122"
export HPML_PREPARED="$HPML_ROOT/source/benchmark-inputs/trial-tokenized.json"
export HPML_GPU_UUID="GPU-2b339976-866e-636d-c1de-fbc4789c6d88"
```

The GPU value identifies the initial allocation's device. A new allocation requires its actual assigned UUID. The current exporter requires a diagnostic run and its primary run to have the same physical GPU. A different device requires a fresh matched measurement group or a separately reviewed comparison rule.

2. Confirm that the model port is free and that the verified packages exist at `$HPML_TRIAL/diagnostic-runtime/sglang` and `$HPML_TRIAL/diagnostic-runtime/vllm`.
3. On the same allocation, start or resume the diagnostic sequence.

```bash
python3 baselines/trial_sequence.py \
  --prepared "$HPML_PREPARED" \
  --output "$HPML_TRIAL/diagnostics" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --configurations eagle3 dflash peagle ar-sglang ar-vllm-v1 \
  --diagnostic-runtime "$HPML_TRIAL/diagnostic-runtime" \
  --probe-sglang-logits \
  --run
```

The controller validates and skips completed configurations. It preserves partial attempts before rerunning a configuration. Each attempt saves qualification, launch settings, logs, and traces. The score probe records a separate HumanEval request before SGLang diagnostic warmups. The probe examines target scores at the known early divergence. The current evidence leaves the cause of response differences unresolved.

SGLang full-round latency starts before drafting and ends after the scheduler completes its state update. That update includes output append, stopping checks, streaming, and completion of the corresponding cache operations. P-EAGLE links the actual draft producer to its later verification and scheduler update. P-EAGLE can prepare its next draft before that update ends, so adjacent full-round intervals can overlap. The exporter preserves that overlap. Synchronization and trace construction add work to these separate diagnostic executions.

The [team pilot runbook](team-peagle-runbook.md) describes the separate Transformers group. The current procedure imports the teammate's original checkpoints. The supplied data contains 200 training examples and zero held-out examples. Original training revision and seed fields remain unknown where evidence is unavailable.

## Refresh the allocation record

1. In the connected cluster terminal, save a parent-allocation snapshot.

```bash
python3 baselines/compute_usage.py --since 2026-11-01 \
  --include-job 14056242 \
  --output "$HPML_ROOT/results/trial-first30-20261122/compute-usage.json"
```

2. From the Mac project directory, copy the record through your authenticated SSH connection.

```bash
scp aa5886@insomnia.rcs.columbia.edu:/insomnia001/depts/edu/users/aa5886/hpml-baselines/results/trial-first30-20261122/compute-usage.json \
  measurements/trial-first30-20261122/compute-usage.json
```

3. In the prompt interface, select **Cluster compute**, then select **Reload the saved record**.

The project total includes job names beginning `hpml-` and the explicitly included preparation job. Each parent allocation contributes once. GPU-hours multiply allocated GPUs by elapsed hours. CPU-core-hours multiply allocated cores by elapsed hours. Completed, cancelled, and failed jobs contribute their actual elapsed allocation time. A running job contributes the elapsed time available at the snapshot's observation time.

The [report update procedure](../measurements/report/README.md) gives commands for final per-job accounting, immutable snapshots, result export, Site packaging, and W&B validation. Whole-job totals include setup, warmups, qualification, inference, diagnostics, and idle time. Timed benchmark blocks retain their own scope.

## Trace the implementation

The request recorder is `baselines/trial_recorder.py`. The engine launcher is `baselines/trial_server.py`. Diagnostic collection uses `trial_phase_hooks.py` and `trial_round_hooks.py`. The exporter validates these records in `trial_export.py`.

The pinned SGLang revision is `0bcd822377da7b5718e674eaf9c870d349424dd1`. Its tokenizer manager defines stream construction. The pinned vLLM completion protocol and serving module define delta token IDs. Run manifests retain the actual package revisions and source hashes.

- [Python's clock documentation](https://docs.python.org/3/library/time.html#time.perf_counter_ns) defines the elapsed-time clock.
- [NVIDIA's query documentation](https://docs.nvidia.com/deploy/nvidia-smi/) defines device memory observations.
- [PyTorch's memory documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management) defines framework counters.
- [Slurm's accounting documentation](https://slurm.schedmd.com/sacct.html) defines allocation records.
