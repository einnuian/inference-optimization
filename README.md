# Inference optimization through multi-token prediction with speculative decoding

This Columbia High Performance Machine Learning project investigates how a language model can generate responses faster while using less GPU time per output token. The study compares released speculative-decoding methods and evaluates feed-forward compression in a custom P-EAGLE-style drafter. The research goal is to improve generation speed and compute efficiency while preserving the target model's response behavior.

In our initial evaluation, Nine configurations produce 810 timed responses from 30 selected prompts and three repetitions. DFlash leads six of the ten benchmarks in observed throughput and GPU-time efficiency. The 75% and 50% pilot widths remain slower than their matching ordinary-generation reference on every benchmark. These findings establish the next research priority: improve proposal acceptance and reduce drafting cost, then test whether the resulting architecture delivers a consistent advantage.

## Research motivation

An autoregressive (AR) language model generates one token at a time. Each successive token requires another target-model evaluation. Speculative decoding introduces a smaller drafter that proposes several tokens, which the target verifies together. Acceleration depends on how much progress the target accepts and how much time drafting and verification consume.

A drafter's feed-forward network transforms each hidden vector within a transformer layer. Reducing its intermediate width reduces matrix sizes while preserving the layer's input and output dimensions. A narrower network can also change proposal quality. The project examines the relationship between the compute saved by compression and the accepted progress produced during verification.

## Research questions and hypotheses

The study addresses three connected questions:

1. Which released implementation provides the highest throughput and lowest GPU time per output token under a shared evaluation procedure?
2. Can a narrower feed-forward network reduce drafting cost while retaining enough accepted tokens to improve generation speed?
3. Which changes to conditioning, drafter depth, and proposal length improve the balance between execution cost and accepted progress?

The compression hypothesis predicts a speed and compute-cost benefit when reduced drafting time exceeds any additional verification work caused by lower acceptance. The architecture hypothesis predicts that better proposal quality can increase committed tokens per verification round. The next experiments test these hypotheses with fixed training conditions, matching AR references, and response-preservation checks.

## Methods and experimental scope

The evaluation separates the released-method comparison from the custom drafter pilot. Each speculative method has an AR reference using the corresponding inference engine. Paired speedup therefore uses the serving implementation associated with that method.

| Method or study | Role in the project | Evaluated implementation |
|---|---|---|
| EAGLE-3 | Feature-conditioned speculative decoding provides a released speed reference. | SGLang uses the converted released drafter and its target-only AR reference. |
| DFlash | Block-diffusion drafting provides a second released speed reference. | SGLang uses the released UltraChat drafter and the same SGLang AR reference. |
| P-EAGLE | Parallel drafting provides the closest released comparison for the pilot. | The H-Spec authors' Llama reproduction uses vLLM ModelRunnerV1 and a matching AR reference. |
| Pilot FFN 100%, 75%, and 50% | Feed-forward compression tests whether a smaller custom drafter improves efficiency. | A four-layer P-EAGLE-style drafter proposes eight tokens per round and uses a Transformers AR reference. |

The three released methods, three pilot widths, and three AR references form the nine completed configurations. The pilot uses the team's original imported checkpoints, whose weight hashes and available training provenance remain recorded. The supplied training file contains 200 examples. The notebook specifies one epoch and 50 optimizer updates per width. DSpark and its vLLM ModelRunnerV2 reference remain deferred.

## Evaluation design

The trial takes the first three prepared prompts from each of ten benchmarks. Each prompt uses one user turn and repeats three times per configuration. The experiment therefore contains 30 unique prompts and 90 timed responses per configuration.

| Task family | Benchmarks |
|---|---|
| Mathematics | GSM8K, MATH-500, and AIME 2025. |
| Code | HumanEval, MBPP, and LiveCodeBench. |
| Conversation and instruction following | MT-Bench, Alpaca, and Arena-Hard. |
| Summarization | CNN/DailyMail. |

Every configuration uses **Meta Llama-3.1-8B-Instruct**, the same prepared input token IDs, greedy decoding at temperature zero, one active request, and a 2,048-token output limit. Each run uses one NVIDIA L40S GPU on Columbia's Insomnia cluster. The benchmark manifest and run records preserve the exact prompts, model revision, checkpoint identities, software settings, and hardware.

The measurement client runs on the compute node and records submission and streamed arrivals with a monotonic clock. Each timed benchmark block contains three sequential requests. Tables report the mean, sample standard deviation, and eligible repetition count. Repeated execution of the same prompts estimates timing variation. A larger prompt sample is required to evaluate generalization across a benchmark.

### Primary measurements

| Metric | Definition | Research purpose |
|---|---|---|
| Output tokens per second (OTPS) | Delivered output tokens divided by the complete benchmark-block interval. | Higher OTPS indicates faster output delivery. |
| Paired speedup over AR | A method's OTPS divided by its matching AR OTPS for the same benchmark and repetition. | Values above one indicate acceleration over ordinary generation. |
| Allocated GPU time per million output tokens | Benchmark-block seconds multiplied by the allocated GPU count, divided by delivered tokens, and multiplied by one million. | Lower values indicate lower compute cost per unit of generated output. |
| Mean acceptance length, τ | Committed draft and target tokens divided by verification rounds, before final output trimming. | Larger values indicate more generation progress per verification round. |

Supporting measurements include time to first token, completion and streaming latency, draft acceptance, exact output agreement, GPU memory, and allocation totals. Exact agreement requires identical output token IDs, output lengths, and stopping reasons. Task-specific accuracy requires separate scoring of mathematics answers, code tests, summaries, and conversational responses.

Compute cost means GPU time per output token. The timed interval includes client gaps within each benchmark block. Whole-job accounting records model startup, warmups, and other allocation time separately. Acceptance fractions retain each method's proposal structure: EAGLE-3 counts candidate tree nodes, while the linear methods count proposed sequence positions.

## Initial findings

### Released-method comparison

DFlash has the highest observed mean OTPS and lowest mean GPU time per output token on six benchmarks. P-EAGLE and EAGLE-3 each lead on two. The category results identify where each implementation performs best under the recorded procedure.

| Task family | Observed speed and GPU-time leaders |
|---|---|
| Mathematics | DFlash leads GSM8K. P-EAGLE leads MATH-500 and AIME 2025. |
| Code | DFlash leads HumanEval, MBPP, and LiveCodeBench. |
| Conversation and instruction following | EAGLE-3 leads MT-Bench. DFlash leads Alpaca and Arena-Hard. |
| Summarization | EAGLE-3 leads CNN/DailyMail. |

These rankings describe the selected prompts and recorded responses. The complete report preserves each benchmark's measured values, uncertainty, matching reference, and evidence.

### Feed-forward compression pilot

The 75% pilot achieves **0.798–0.971×** its matching AR throughput, and the 50% pilot achieves **0.807–0.936×**. Both narrowed pilots therefore require more GPU time per output token than AR on all ten benchmarks. Within the comparison of the two narrowed widths on the same GPU, 75% has higher throughput on seven benchmarks.

Across the observed width means, 100% leads on six benchmarks and 75% leads on four. The full-width run uses a different physical L40S, which limits conclusions about small width effects. The current observations show no consistent speed or compute-cost advantage from narrowing.

Pilot acceptance length remains approximately **1.1–1.25 tokens per verification round**, despite eight draft proposals. Accepted drafts therefore add limited progress beyond the retained target token. This result motivates experiments that improve proposal acceptance and measure whether the improvement compensates for drafting and verification time.

## Interpretation and limitations

The completed evaluation establishes baseline measurements and a first compression result. The published catalogue contains 2,056 measured cells, and the saved records preserve all 810 timed responses. The current pilot identifies concrete targets for the next architecture study: stronger proposal quality, lower drafting time, and a controlled comparison of feed-forward widths.

**Response preservation remains unresolved.** EAGLE-3 matches its AR reference on 11 of 30 unique prompts, DFlash on 13, and P-EAGLE on 14. Each narrowed pilot matches its reference on 12. Outputs remain stable within each configuration across the three repetitions, while response lengths and stopping can differ between configurations. The speed measurements therefore describe the observed responses. An exact-preservation claim requires resolving those differences, and task accuracy remains unmeasured.

The sample contains three prompts per benchmark. The five initial configurations and the full-width pilot run on one physical L40S. The narrowed pilots and their complete AR reference run on another physical L40S. The full-width pilot's paired speedup and agreement cells remain empty. The original training seed and complete training history remain unresolved. Partial memory and diagnostic coverage also limit resource comparisons, and unavailable measurements remain empty in the tables.

## Next research steps

1. Establish a stronger full-width drafter with documented training data, target revision, updates, and seeds, and verify its response behavior against AR.
2. Test conditioning, layer count, and proposal length to improve accepted progress while reducing drafting and verification time.
3. Compare full and narrowed feed-forward widths on one physical GPU under fixed training and evaluation conditions.
4. Measure each candidate against matched AR, EAGLE-3, DFlash, and P-EAGLE using the same throughput, GPU-time, acceptance, and response checks.
5. Expand the held-out prompt sample, repeat training with multiple seeds, and add task-specific accuracy scoring.
6. Extend the study to Qwen and Inkling target models after confirming runtime support and compatible drafters.

A successful optimization must demonstrate higher throughput and lower GPU time per output token alongside the required response-preservation and task-quality evidence. The next phase tests that outcome through controlled architecture experiments.

## Reproducibility

The repository preserves the evaluation code, exact selected prompts, tokenized inputs, metric definitions, configuration summaries, allocation records, and report source. The [reproduction guide](baselines/reproducing-the-trial.md) documents runtime preparation, measurement commands, and table regeneration. The [benchmark manifest](benchmarks/manifest.json) records input hashes and source attribution.

The full response archive is available in the Team-visible W&B project. The compact results and report files are included in this repository. Model weights and runtime environments retain their own storage and access requirements.

## Project resources

| Resource | Contents |
|---|---|
| [Evaluation report](measurements/report/report.pdf) | The paper presents the research objective, evaluation procedure, complete results, interpretation, and limitations. |
| [Presentation](measurements/report/HPML_Midpoint_Report_All_Categories.pptx) | The eleven-slide presentation includes result excerpts for mathematics, code, conversation, and summarization. |
| [Report source](measurements/report/report.tex) | The editable LaTeX source includes generated measurement tables. |
| [Compact measured results](measurements/results/trial-first30-20261122/) | The exports preserve measured cells, evidence hashes, configuration summaries, allocations, and provenance. |
| [Final W&B run](https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22/runs/48b0a0ffa383382fa011) | The Team-visible run preserves the complete recording archive and report artifacts. |
| [W&B project](https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22) | The Team-visible project preserves individual configuration runs and earlier evidence versions. |
| [Reproduction guide](baselines/reproducing-the-trial.md) | The guide provides the commands and prerequisites for repeating the evaluation. |
| [Baseline implementation records](baselines/README.md) | The technical records describe source revisions, runtime preparation, and implementation boundaries. |
| [Pilot implementation](peagle_ffn.ipynb) | The notebook contains the team's original custom drafter and width experiment. |
