# Common baseline metrics and controls

The selected central benchmark methods are **P-EAGLE, EAGLE-3, DSpark and DFlash**. EAGLE-3 uses the prepared SGLang path for the central serving comparison; its original research implementation remains a secondary reference. H-Spec and PARD-2 remain documented as preparation history and related work. The central comparison requires ordinary SGLang, ordinary vLLM Model Runner V1 and ordinary vLLM Model Runner V2.

The current comparison measures the unchanged released pretrained methods under common conditions. It asks which qualified implementation returns the same target output fastest, and which implementation uses the fewest GPU resources and actual GPU charges per million returned output tokens while preserving response speed and output quality. Every result records the target, engine, model runner, draft checkpoint, workload and arrival rate. The [common benchmark tutorial](https://hpml-workspace.ayomideadetayo.chatgpt.site/tutorials#benchmark-overview) proposes the workload and explains the remaining measurement work.

## Define the token counts

| Count | Definition and collection |
|---|---|
| Input tokens | Target-tokenizer IDs supplied to the target after the chat template. Record uncached and cached input counts separately when prefix caching operates. |
| Proposed draft tokens | Every speculative candidate position produced by the drafter. For trees, count all candidate nodes and retain tree depth separately. Exclude an existing anchor token from newly proposed positions. |
| Accepted draft tokens | Draft proposals accepted by target verification. Record total proposals, accepted proposals, proposed length and accepted-prefix length for each cycle. Define per-position denominators as described after the table. |
| Correction or bonus tokens | Tokens supplied by the target after a rejection or a fully accepted proposal sequence. Record these separately from accepted draft tokens. |
| Committed target-output tokens | Tokens appended to the verified output sequence before final length and stop handling. Preserve the complete IDs for correctness diagnosis. |
| Returned output tokens | Committed target tokens exposed to the client after the agreed stop and length rules. Count each returned token once, including streamed partial output from failed requests in the all-requests accounting. Also report returned tokens from successful requests separately. |
| Provider-reported completion tokens | The serving API's own usage count. Save this count and reconcile it with the returned IDs, including special tokens, hidden output and clipping. |

For a chain or parallel block, position `j` is eligible in cycles that propose at least `j` tokens. Prefix survival at `j` is the fraction of those eligible cycles that accept through position `j`. Conditional acceptance at `j` uses the eligible cycles whose preceding positions all passed as its denominator. Record both denominators, especially when proposed length varies. A zero denominator gives an undefined rate. For trees, retain candidate-node counts and depth, and identify the verified branch before reporting prefix statistics.

Store original target token IDs when the implementation exposes them. Re-tokenizing concatenated text can change token boundaries; label any text-derived count explicitly. A transport chunk can contain several tokens. Record each chunk's arrival time and token count so streaming pauses remain visible.

Keep the released generator's stopping behavior separate from the final returned-output policy. The EAGLE and PARD-2 setup wrappers preserve raw generated IDs, apply the same first-stop and strict-length rule after generation finishes, and report both the normalized comparison and raw boundary compliance. A passing normalized comparison establishes that post-generation transformation on the tested prompts. A streaming adapter requires its own implementation and output checks before client timing and concurrent service. Native stopping compliance also requires the generation function to end at the agreed boundary. Any tokens computed after that boundary still contribute to elapsed time and GPU resources. The returned-token denominator includes only the output exposed under the agreed policy.

For SGLang, the recorded native token sequence comes from its server response after scheduler stopping. A passing native check establishes that response boundary. Count internal proposals, accepted tokens and discarded work through separate engine instrumentation.

Input length affects prefill and cache memory. Draft and verification work affect GPU time even when the proposed tokens never appear in the response. Returned output tokens form the normalization denominator requested by the study. Under continuous serving, concurrency, queueing, padding, prefix reuse, idle time, retries and allocation duration affect the resources needed for that output. Record those serving conditions beside each rate.

## Measure the common outcomes

Use one client monotonic clock for request submission, first returned content, subsequent content chunks and completion. Record server-side timing separately. Let `N` be returned output tokens in the selected interval and `W` be its elapsed seconds.

| Metric | Definition | Research role |
|---|---|---|
| Greedy output and stopping match | Compare original token IDs, returned length and stop reason with the target-only control under the same generation settings. Record the first divergence and full outputs for each failure. | Required output-preservation condition for both questions. |
| Time to first token (TTFT) | First returned content timestamp minus request submission timestamp. Include transport and queueing; an initial empty streaming event supplies no token. | Measures initial response speed. |
| Streaming pause | Time between consecutive nonempty content chunks. Report p50, p95, p99 and maximum, with tokens per chunk. Record the number of requests with at least two nonempty chunks. Freeze whether summaries pool all gaps or summarize each request first; label both when both are reported. An empty set of eligible gaps gives an undefined result. | Measures interruptions a reader experiences. |
| Time per output token (TPOT) | For requests returning more than one token, `(last_token_time - first_token_time) / (N_request - 1)`. Report the averaging method. Chunk timestamps constrain the precision of token timing. | Describes average decode delivery speed. |
| Completion latency | Completion timestamp minus request submission timestamp. Preserve timeouts, cancellation and error outcomes. | Measures the complete response. |
| Per-request returned-token rate | Returned tokens divided by complete request latency. Report the request distribution; keep separately labeled decode-only rates when available. | Compares request-level speed. |
| Aggregate returned-token throughput | Total returned tokens across all requests divided by the common measurement interval `W`. | Measures serving capacity at each arrival rate. |
| Request outcomes | Submitted, completed, failed, timed out and cancelled counts, with queue depth or waiting time when exposed. | Exposes overload and incomplete service. |
| Allocated GPU-seconds | Sum, across all assigned GPUs, of their allocation duration for the stated interval. Include idle allocated time. | Measures the GPU resource requirement. |
| Allocated GPU-seconds per million returned tokens | `1,000,000 × allocated_GPU_seconds / N`. A zero denominator gives an undefined rate. | Primary normalized resource result. |
| Actual GPU charges per million returned tokens | `1,000,000 × attributable_GPU_charges / N`, with currency, billing source and charged interval. | Primary monetary cost result. Missing billing evidence leaves the result unmeasured. |
| Peak GPU memory | Maximum per-device process allocation and reservation from the framework, plus the GPU-level observed peak where available. Record observation method and sampling period. | Explains feasibility and capacity; supports the cost interpretation. |

Report p50, p95 and p99 latency across requests, each repeat's aggregate throughput, and uncertainty across repeated runs. Keep raw observations. The response-speed requirement for the cost question must be fixed before the final comparison: specify maximum permitted TTFT, completion latency and streaming pauses, or a preregistered non-inferiority tolerance against the selected reference. Freeze an acceptable completion/success requirement as well, so a lower resource result represents equivalent successful service. Record the chosen thresholds with the workload and report successful-request token totals beside all-request totals.

For sampled generation, record temperature, top-p, top-k, seeds and the evaluation used to establish output quality or distribution preservation. Greedy equality establishes the tested greedy configuration's behavior. Positive-temperature claims require their own evidence. A zero returned-token count makes every per-token resource or charge result undefined.

Allocated GPU time, CUDA kernel duration and GPU utilization describe separate quantities. CUDA traces diagnose computation; scheduler allocation records support resource accounting. Preserve both complete-job resource usage and the marked steady measurement interval. If several allocations overlap, sum each GPU's time once. Use the same interval's returned tokens for each normalized result.

Actual charges require attributable billing evidence. A multiplication by an advertised hourly rate produces an estimated charge and carries that label. A Slurm `edu` allocation establishes account membership and resource access; the allocation alone provides no actual monetary charge.

## Translate the released measurements

The [paper measurement tutorials](https://hpml-workspace.ayomideadetayo.chatgpt.site/tutorials#papers-overview) explain the authors' procedures and the [core-method audit](core-methods.md) identifies the pinned artifacts. A paper's metric name alone leaves its aggregation, timer and token boundary unresolved; preserve the following source-specific definitions alongside the common results.

| Released measurement | Meaning in the inspected implementation | Common measurement to add |
|---|---|---|
| EAGLE evaluation wall time and generated-token count | Generation interval includes prefill and CUDA synchronization. Its ordinary script keeps a loaded drafter resident. Both Llama evaluation scripts parse an output-limit argument and omit it from the generation call; the native default and block-level stopping determine the actual boundary. | Clean target-only memory/resource control, client TTFT, streaming events, complete stop/length checks and identical arrivals. |
| EAGLE `speed.py` | Divides mean per-conversation rates. The speculative side uses recorded generated counts; the ordinary side retokenizes cleaned text with a hard-coded Llama-2 tokenizer. | Recalculate both sides from preserved target output IDs and the same boundary; retain a separately labeled author-formula result when reproducing it. |
| DFlash local decode time and `acceptance_lengths` | Decode timer begins after the first token. Its divisor is the function's clipped output-ID count, including that first token and a retained stop ID. `acceptance_lengths` records clipped newly produced positions per cycle and can include a correction/bonus token. A stop inside the accepted prefix can exclude that token. | Separate TTFT, complete-request latency, accepted-draft counts, proposal counts and returned-token counts. |
| DFlash local summary | Displayed throughput is the reciprocal of the mean request-level time per output token. Its acceptance summary averages each request's cycle mean, then averages across requests. | Aggregate returned tokens over a shared interval and cycle-pooled acceptance, each with its own label. |
| DFlash OpenAI benchmark throughput | Nonstreaming requests maintain a selected active concurrency; throughput uses API completion counts over a shared interval. | Streamed chunk timestamps and an externally timed arrival trace, with reconciled returned-token counts. |
| PARD-2 native `token_per_sec` | The timer resets after the first target verification/prefill pass. In target-dependent mode, the preceding target-feature pass and cache reset also occur before that origin. The numerator counts the full internally generated suffix, and the elapsed interval includes final text decoding. | Outer full-request timing, separately marked prefill/feature preparation, and returned-token throughput. |
| PARD-2 `accept_length` | Accepted proposals plus the target correction/bonus token, recorded before final stopping checks. | Accepted drafts, clipped returned tokens, stop reasons and request completion. |
| PARD-2 `acc` | Comparisons examined through the first rejection, with later proposed positions absent from the list. | Total proposals and accepted proposals with explicit denominators. |
| H-Spec paper acceptance, ITL speedup and serving throughput | Paper-specific single-request and fixed-concurrency settings; main sampling settings differ from greedy qualification. | Matching greedy IDs, matched ordinary runs for each model runner, identical arrivals and client streaming timing. |
| P-EAGLE paper Table 10 output throughput | Aggregate output tokens/s across concurrent requests. Reported speedup uses the best tested autoregressive EAGLE-3 draft length as the reference. | Ordinary-generation throughput in the same runner and an explicitly named speedup denominator. |
| P-EAGLE reproduction counters | Depend on the pinned patched vLLM runner and its token conventions. | Trace one request through proposed, accepted, bonus, committed and returned counts before aggregating. |
| DSpark DeepSpec offline `accept_len` and `verify_rate` | `accept_len` divides summed cycle advancement by proposal count. `verify_rate` divides that advancement by proposed draft positions plus one position per cycle. The evaluator reports agreement statistics without a service wall-time result. | Full request and streaming timing, resource accounting, accepted-draft counts and returned-token counts. |
| Prepared Llama DSpark | H-Spec authors' reproduction in patched vLLM uses fixed-length drafting and skips the unconnected confidence-head weights. | Record fixed proposal length and reproduction identity; the original production confidence scheduler has a separate reproduction requirement. |

## Collect separate diagnostic profiles

| Diagnostic | Required observation | Decision supported |
|---|---|---|
| Complete draft time | Elapsed proposal-generation interval, with actual block or tree shape. | How much cycle time the released drafter requires. |
| Drafter feed-forward network (FFN) time | Time in the intermediate and output projections, with kernel names and executed shapes. | How much elapsed draft computation comes from the feed-forward network. |
| Draft attention and vocabulary projection | Component durations, cache growth and draft-vocabulary work. | Whether another draft component dominates. |
| Target verification | Elapsed verification time, verified positions and block/tree shape. | How proposal length changes target work. |
| Accepted drafts per proposal | Accepted draft count divided by all proposed positions, plus prefix survival and conditional acceptance with the eligible denominators defined above. | How checkpoint, method settings and workload affect useful proposals. |
| Returned tokens per cycle | Returned tokens produced during verification cycles divided by the number of those cycles, with final-cycle clipping recorded. When prefill produces the first output token, count that token separately from cycle output. | How cycle duration relates to useful output progress. |
| Scheduling and other work | CPU dispatch, queueing, synchronization, transfers and gaps between kernels. | Whether serving overhead limits improvements. |

Profile in separate runs because tracing can alter timings. GPU kernels can overlap; summed kernel durations can exceed elapsed cycle time. Use a trace that identifies overlap and records the complete cycle interval. Associate every profile with its prompt, token counts and configuration.

## Freeze the comparison controls

1. Record target, tokenizer, chat template, generation configuration, model revisions and code revisions.
2. Fix precision, device placement, attention backend, output cap, stop-token policy, context limit and deterministic greedy settings for qualification.
3. Hold the physical GPU, GPU count, CPU allocation, host memory and resource limits constant across the first shared baseline comparison. Record the node and GPU UUID. Report resource-allocation sweeps as separate comparisons with every assigned GPU included in resource and charge accounting.
4. Separate warmup and compilation from measured inference, and record cold-start measurements separately when included.
5. Fix prefix caching, batching, CUDA-graph, draft and scheduler settings, and preserve the exact server command.
6. Replay the same prompt order and externally scheduled arrival timestamps for each serving configuration.
7. Freeze the measurement start/end, workload-draining rule, repeat count and acceptable request-success requirement. Preserve failed and unfinished requests and apply the same rules to every configuration.
8. If method settings require selection, define development and reserved prompt sets before that selection. Preserve the unchanged checkpoint and record every evaluated inference setting.

If engines produce different greedy token sequences under matched settings, investigate the discrepancy and retain engine-specific output controls. Cross-engine results then state the exact output/quality condition established. A complete-system ranking compares the combined engine, scheduler, kernels and model configuration. The current comparison measures these released configurations with fixed weights. Its results establish which systems perform well under each workload and which measurements need further investigation.
