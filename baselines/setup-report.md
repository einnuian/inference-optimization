# Completed installation and output checks

The selected central benchmark methods are **P-EAGLE, EAGLE-3, DSpark and DFlash**. EAGLE-3 uses the prepared SGLang path for the central serving comparison; its original research implementation remains a secondary reference. H-Spec and PARD-2 remain documented as preparation history and related work. The central comparison requires ordinary SGLang, ordinary vLLM Model Runner V1 and ordinary vLLM Model Runner V2.

The baseline preparation uses Meta Llama-3.1-8B-Instruct as the common target and released pretrained drafters for EAGLE-3, DFlash, P-EAGLE, H-Spec, PARD-2 and DSpark. Every target file comes directly from Meta. The current task is to understand these implementations and measure their unchanged pretrained configurations under shared conditions.

A drafter proposes candidate future tokens. Speculative decoding uses the target model to verify those proposals. Multi-token prediction (MTP) predicts several future positions from the available context. The completed work establishes software installation and four-prompt output behavior. The shared performance comparison remains to be prepared.

The Insomnia account, private project copy, seven public source repositories and seven released drafter snapshots are prepared. The complete Meta target passes every content-hash check. Original EAGLE-3 and PARD-2 load the target and match all four outputs after applying the shared stopping and length policy. Their native generation functions produce extra tokens beyond some output boundaries, as recorded below. SGLang DFlash and EAGLE-3 each pass all four native output and stopping comparisons. P-EAGLE, H-Spec and DSpark also pass all four comparisons against their respective ordinary vLLM controls. Performance benchmarks follow the broader correctness and measurement protocol described below.

## Baseline readiness

| Method or control | Released artifacts | Environment and GPU evidence | Target generation and output evidence |
|---|---|---|---|
| Ordinary Meta generation | All thirteen canonical Meta files verified in job `13929459`; final direct-Meta provenance saved. | Each ordinary control uses a clean target-only process. | All five paths—EAGLE, PARD-2, SGLang and vLLM Model Runners V1/V2—produce the same input IDs, output IDs, caps and stopping results on all four prompts. |
| Original EAGLE-3 | Official `yuhuili/EAGLE3-LLaMA3.1-Instruct-8B` snapshot verified. | Python 3.10.22, Torch 2.6.0+cu124 and Transformers 4.53.1 installed. EAGLE class imports and L40S BF16 operation pass. | All fifteen saved tensors and the target-sourced embedding match. All four adapted outputs match; native cap overshoot occurs on two prompts. |
| SGLang EAGLE-3 | `jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B` snapshot verified. | SGLang 0.5.19 CUDA 12.9 container, GPU kernels and server imports pass. | Job `13930206` loads ordinary and EAGLE-3 servers separately. All four native token sequences, returned texts, stopping reasons and caps match. |
| DFlash | Official `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` snapshot verified. | Uses the same qualified SGLang deployment. | Job `13930206` loads DFlash and matches ordinary generation on all four native output and stopping checks. Canonical Llama input and stop IDs are supplied explicitly. |
| P-EAGLE | H-Spec authors' released Llama reproduction verified. | Patched vLLM 0.25.1 CUDA 12.9 dependencies, GPU operations and sampling kernels pass. P-EAGLE uses Model Runner V1. | Job `13930616` passes all four native token, text, stopping and cap comparisons against ordinary V1. |
| H-Spec | Authors' released Llama snapshot verified. | Shares the pinned patched vLLM environment. H-Spec's hybrid scan passes. H-Spec uses Model Runner V2. | Job `13930616` passes all four native token, text, stopping and cap comparisons against ordinary V2. |
| PARD-2 | Official `amd/PARD2-Llama-3.1-8B` backbone and `warp_model.bin` verified. | Python 3.10.22, Torch 2.6.0+cu124 and Transformers 4.51.3 installed. A compiled 32×32 BF16 matrix operation, StaticCache creation/reset, all 147 backbone tensor names and adapter-value equality after BF16 conversion pass. | Eager loading and all four adapted outputs pass; native stop/cap overshoot occurs on all four. Compiled full-model execution and concurrent serving require separate checks. |
| DSpark | H-Spec authors' released Llama reproduction verified. | Shares the qualified patched vLLM environment and Model Runner V2. | Job `13930616` passes all four native token, text, stopping and cap comparisons against ordinary V2. |

Small setup checks establish that a named component executes under the recorded conditions. Four short prompts support conclusions about those four observed outputs. The initial checks use several L40S nodes. A shared performance comparison requires a recorded physical GPU allocation, a larger frozen workload and a common measurement procedure.

An adapted output applies the common stop-token and maximum-length policy to the generation function's raw token sequence. The current EAGLE and PARD-2 wrappers apply that policy after generation finishes. Original EAGLE-3 emits two extra tokens on the length-cap prompt and one on the code prompt. PARD-2 emits 1, 13, 1 and 3 extra tokens across the four prompts. Both methods match the ordinary reference after adaptation. A streaming adapter still needs implementation and testing before first-token timing, streaming pauses and concurrent service can be compared. Future timing includes all executed work, including work that produces discarded tokens.

SGLang's native boundary is its server response after the scheduler enforces stopping. Both SGLang comparisons pass at that boundary. Internal proposals and any discarded computation require separate instrumentation; the response-boundary result establishes which tokens the server returns.

The vLLM checks likewise use the engine's output after scheduler stopping. All three methods pass at that boundary, and ordinary Model Runner V1 and V2 also agree. The loader audit observes every target and drafter checkpoint input key and successful loader completion. Exact loaded-value comparison remains a separate check. Source inspection finds explicit loader skip rules: the prepared DSpark and H-Spec integrations skip their confidence heads. Their fixed seven-token proposal paths execute the parallel/hybrid backbones and sequential Markov heads; H-Spec also uses the target KV cache. The source guides identify the explicit loader skip rules for each model. Input-key coverage records which parameters reach the loader; component execution requires inspection of the inference path. The [P-EAGLE, H-Spec and DSpark guide](hspec-setup.md) preserves those distinctions and the actual runtime settings.

## Project locations and reproducibility records

| Item | Location |
|---|---|
| Mac project | `/Users/ayo/Documents/Columbia/FALL 26/HIGH PERFORMANCE ML/PROJECT/hpml-project` |
| Insomnia project root | `/insomnia001/depts/edu/users/aa5886/hpml-baselines` |
| Private team source copy | `HPML_ROOT/source/hpml-project` |
| Isolated environments | `HPML_ROOT/envs` |
| Model cache | `HPML_ROOT/cache/huggingface` |
| Model provenance | `HF_HOME/hpml-provenance` |
| Setup and output evidence | `HPML_ROOT/runs` and each environment's `hpml-*.json` records |
| Offline W&B records | `HPML_ROOT/tracking/wandb` |

The team source starts at commit `c8af4638d2b7ba046765a31425be5e7fda80dc6f`. The new `baselines/` files, `tests/` regression checks and README links accompany that checkout as uncommitted work. The original notebook, saved HTML and teammate setup guides remain preserved. Each external source checkout has an exact revision, and each model has a pinned snapshot. [Core methods](core-methods.md) identifies every source and checkpoint by name and revision.

## Access, storage and software decisions

Columbia VPN and authenticated SSH access work. The account is `aa5886`, the Slurm account is `edu`, and the private scratch root is the directory listed above. Login Bash initialization supplies the scheduler configuration. Insomnia's default noninteractive shell initially omits that initialization, so remote commands enter `bash -l` before using Slurm.

The inspected L40S has 46,068 MiB of GPU memory, compute capability 8.9 and NVIDIA driver 550.54.14. Its `nvidia-smi` output reports CUDA 12.4. The CUDA module supplies toolkit 12.3; installed PyTorch packages and containers supply their own selected runtime libraries. Every GPU run records the assigned node and driver because another node can have different software.

The EAGLE and PARD-2 environments use CUDA 12.4 PyTorch builds. The SGLang and patched vLLM environments use explicit CUDA 12.9 artifacts and pass both their recorded basic GPU checks and target/drafter generation on driver 550.54.14. The inspected SGLang 0.5.20 dependencies use CUDA 13. The selected SGLang 0.5.19 CUDA 12.9 container supplies the qualified SGLang deployment.

The patched vLLM environment requires the official CPU TorchCodec package and the matching FlashInfer CUDA 12.9 precompiled sampling-kernel cache. Its executable directory must appear in `PATH`. Transformers 5.13 also requires an explicit `return_dict=False` in the check's chat-template call to return token IDs. The reusable scripts include those resolved settings. Job `13930616` completes all five ordinary/speculative stages on `ins059` with exit code `0`.

Apptainer extracts the SGLang container into the allocation's node-local temporary directory because the cluster lacks the `squashfuse` helper. The setup reuses that extraction for all processes in the allocation and removes the temporary runtime on exit. Cleanup removes approximately 100.36 GiB of disposable container assembly files while preserving the final container, exact image identities, package records, kernel cache and model evidence.

P-EAGLE's reproduction path preserves the H-Spec authors' patched vLLM 0.25.1. Current vLLM main removes Model Runner V1, which that P-EAGLE path uses. H-Spec uses Model Runner V2. An ordinary-generation control under each runner identifies that runner's starting behavior. A P-EAGLE/H-Spec comparison therefore describes complete deployments. The shared benchmark should retain an ordinary control for each runner and identify the runner beside every result.

The direct Meta browser route downloads and transfers one weight file at a time. The importer checks each file's byte size and content hash before accepting the file into the canonical cache. A final provenance record requires all four weights plus Meta's original configuration, tokenizer and documentation files. The completed setup uses the verified canonical Meta files throughout.

The authenticated downloader and browser importer now share the same thirteen-file verification. Repeated setup preserves the original acquisition details and publishes each updated record atomically. Twelve regression checks cover fresh downloads, repeated verification, browser imports, partial imports, incomplete source coverage, file corruption and interrupted record publication. EAGLE and PARD-2 setup scripts also enforce the exact Python and package versions used here before future model loading.

## Repeat the setup

1. Connect through Columbia's required network path and sign in to Insomnia with your own UNI and Duo.

   Follow [Insomnia setup](insomnia.md#confirmed-access-and-current-connection) for the verified procedure. The university account and scratch path belong to the individual researcher.

2. In a login Bash shell, select the private project and cache directories.

   ```bash
   bash -l
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   export UV_CACHE_DIR="$HPML_ROOT/cache/uv"
   cd "$HPML_ROOT/source/hpml-project"
   ```

   The prepared checkout already exists at that location. For a fresh workspace, first create private storage using [the storage procedure](insomnia.md#create-project-storage-and-isolated-environments), then copy the current Mac checkout into its `source/` directory. Include `baselines/` and `tests/`, which are currently uncommitted. From the Mac, the copy command for Ayo's approved destination is:

   ```bash
   scp -r '/Users/ayo/Documents/Columbia/FALL 26/HIGH PERFORMANCE ML/PROJECT/hpml-project' \
     aa5886@insomnia.rcs.columbia.edu:/insomnia001/depts/edu/users/aa5886/hpml-baselines/source/
   ```

   Use that command for a fresh destination. Preserve the existing prepared checkout when resuming this setup. Each teammate substitutes their own UNI and approved scratch location.

3. Verify the pinned source repositories before rebuilding an environment.

   ```bash
   python3 baselines/fetch_sources.py --verify
   ```

   If the fresh checkout has no `external/` source directories, run `python3 baselines/fetch_sources.py --fetch` first. The fetch command obtains the exact reviewed revisions.

4. Inside a CPU allocation, run the installation script for the selected implementation.

   | Implementation | Script and complete instructions |
   |---|---|
   | Original EAGLE-3 | [setup_eagle.sh](setup_eagle.sh) and [EAGLE environment procedure](insomnia.md#prepare-the-eagle-inference-environment). |
   | PARD-2 | [setup_pard2.sh](setup_pard2.sh) and [PARD-2 setup](pard2-setup.md). |
   | SGLang DFlash/EAGLE-3 | [setup_sglang.sh](setup_sglang.sh), its [OCI image helper](prepare_sglang_oci.py), and [SGLang setup](sglang-setup.md). |
   | P-EAGLE/H-Spec | [setup_hspec.sh](setup_hspec.sh) and [P-EAGLE/H-Spec setup](hspec-setup.md). |
   | W&B tracking | [setup_tracking.sh](setup_tracking.sh) and [tracking instructions](tracking.md). |

   Start with the EAGLE installation for a fresh workspace. Its script creates the shared isolated `uv 0.13.8` package installer and managed Python 3.10.22 interpreter used by the later setup scripts. Each method guide specifies its allocation, exact command, supported resume behavior and output records. Preserve the separate environments because the methods require different Transformers, PyTorch and engine versions.

5. Retrieve and verify the pinned model snapshots through [the artifact procedure](insomnia.md#obtain-the-model-artifacts).

   Use Meta's approved repository for every target file. Each method's authors supply its released drafter and adapter. The downloader and browser-import paths preserve exact revisions and provenance.

6. Inside one L40S allocation, execute the selected method's small GPU and model-loading checks.

   The reusable EAGLE and PARD-2 wrappers are [qualify_eagle.py](qualify_eagle.py) and [qualify_pard2.py](qualify_pard2.py). Each wrapper saves ordinary and speculative outputs from separate processes. The engine guides specify the SGLang and patched vLLM checks.

7. Record setup and output results with the [W&B tracking tool](tracking.md), retaining the original evidence on Insomnia.

   The verified W&B team is `ayoadetayo-columbia-university`. Eleven records are synchronized to [hpml-pretrained-baselines](https://wandb.ai/ayoadetayo-columbia-university/hpml-pretrained-baselines): four setup records and seven correctness comparisons. API read-back confirms private project access and all eleven finished runs. Original EAGLE-3 and PARD-2 retain their partial native-boundary results. SGLang DFlash/EAGLE-3 and vLLM P-EAGLE/H-Spec/DSpark retain their passed results. Performance measurements remain empty. Credentials remain outside project files and source control.

## What the saved notebook establishes

The [cell-by-cell notebook review](notebook-review.md) covers all 33 cells in `peagle_ffn.ipynb` and the saved outputs in `peagle_ffn_insomnia_run.html`. The notebook constructs and trains three custom four-layer drafters. Each width receives 50 pilot training updates over 200 examples. Those historical results belong to the earlier team pilot. The present comparison uses the authors' released pretrained checkpoints.

| Saved notebook observation | Measurement lesson for the released-method comparison |
|---|---|
| Exact greedy IDs match on two of four prompts at every width. Divergence begins at token 67 and token 6 on the other prompts. | Save original token IDs, stopping behavior and the first differing position. Investigate each mismatch before claiming exact output agreement. |
| Ordinary generation records 41.4 tokens/s; the custom drafters record 38.3, 38.3 and 38.0 tokens/s. | Measure every released method against a named ordinary control under the same conditions. |
| Verification occupies about 85% of the profiled full-width cycle; drafting occupies about 14%. | Collect draft, verification and remaining elapsed time in separate diagnostic runs. |
| All three custom drafters remain resident during ordinary generation. | Start each ordinary memory control in a fresh process containing the target and its required engine state. |
| The notebook counts committed progress before final output clipping. | Preserve proposed, accepted, committed and returned token counts separately. |
| The notebook and HTML differ in cell 3, and saved execution counts repeat or appear out of order. | Preserve both originals and record the exact source revision for each future run. |

## What remains before a shared performance comparison

The [common metrics and controls](metrics.md) defines response timing, returned-token throughput, output agreement, GPU resources, actual attributable charges and memory. The next work is to agree on a comparison protocol for the unchanged released configurations.

1. Select and save a common prompt set, canonical input token IDs, greedy settings, stop policy and output cap.
2. Choose one physical GPU allocation and record the node, GPU UUID, driver, CPU and host-memory allocation.
3. Record one supported inference preset per configuration and its matched ordinary engine/runner control.
4. Implement a common measurement client and the method adapters needed to return token IDs, stopping results and streaming events.
5. Extend output checks to the selected workload, context capacity and serving settings.
6. Measure isolated requests with stated warmup, repetitions and timing boundaries.
7. Define a concurrent-arrival trace and replay the same scheduled requests through each qualified service.
8. Collect diagnostic profiles separately and explain each result with the observed component times and token counts.

EAGLE research code and PARD-2 currently return complete sequences through wrappers that apply stopping after generation. Their streaming adapters remain to be implemented. SGLang has a checked HTTP generation path. The vLLM checks use the Python `LLM` interface; its selected HTTP streaming configuration needs its own output check. Every timing comparison must identify the interface and include adapter overhead consistently.

A performance record needs the workload, exact configuration and measurement interval. Actual GPU charges require attributable billing evidence. W&B currently contains setup and correctness records; its automatic `_runtime` field describes logging duration. The shared performance measurements remain unmeasured.

The project has yet to select a later architecture intervention. The present deliverable establishes how to understand, run and compare the released methods.
