# Prepare P-EAGLE, H-Spec and DSpark on Insomnia

Checked on November 16, 2026. The setup uses the H-Spec authors' released Llama P-EAGLE and DSpark reproductions and their H-Spec implementation. All three methods use existing pretrained weights. The shared environment contains the authors' patches to vLLM 0.25.1 and Speculators. P-EAGLE uses Model Runner V1; H-Spec and DSpark use Model Runner V2. Each runner needs its own ordinary-generation control in the later benchmark.

## Recorded setup state

Model job `13930616` completes all five stages on `ins059`, using one L40S with compute capability 8.9, 46,068 MiB and NVIDIA driver 550.54.14. Every speculative method matches its ordinary-generation control on the four prompts' native token IDs, returned text, stop reasons and output caps. Ordinary Model Runners V1 and V2 also match on all four prompts. The [saved comparison summary](results/hspec-output-check-13930616/summary.json) records the observed results.

| Stage | Observed checkpoint input keys | Four-prompt result |
|---|---|---|
| Ordinary Model Runner V1 | All 291 target keys. | Native stopping and output caps pass for all four prompts. |
| P-EAGLE, Model Runner V1 | All 291 target keys and all 52 draft keys. | Native outputs match ordinary V1 for all four prompts. |
| Ordinary Model Runner V2 | All 291 target keys. | Native outputs match ordinary V1 for all four prompts. |
| H-Spec, Model Runner V2 | All 291 target keys and all 85 draft keys. | Native outputs match ordinary V2 for all four prompts. |
| DSpark, Model Runner V2 | All 291 target keys and all 66 draft keys. | Native outputs match ordinary V2 for all four prompts. |

All native loaders complete. The target loader reports 195 destination parameters after combining compatible projections. The three draft loaders expose complete input-key traces and return no complete destination-name set. Exact loaded tensor-value identity remains a separate check. The [compact runtime summary](results/hspec-output-check-13930616/runtime-summary.json) identifies actual runner, proposer, model and attention classes and the full loading-evidence locations on Insomnia. FlashAttention 2 supplies target and drafter attention in the recorded configuration.

The input-key counts above include tensors presented to the loaders. DSpark's native loader and H-Spec's patched loader explicitly skip confidence-head parameters during inference. Their executed paths use the parallel backbone, active Markov head and seven speculative positions. For DSpark, label the result **H-Spec authors' Llama reproduction, fixed proposal length**. Reproducing the original DSpark paper's confidence-based, load-aware prefix selection requires a separately qualified serving implementation. [DSpark loader, lines 172–184](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/model_executor/models/qwen3_dspark.py#L172-L184); [H-Spec serving patch, lines 1190–1204](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/vllm.patch#L1190-L1204).

CPU setup job `13930257` verifies the two author patches, both official binary-wheel digests, dependency consistency across all 213 installed packages and patched-module imports. GPU job `13930196` passes BF16 matrix multiplication, vLLM RMS normalization, Torch compilation and H-Spec's hybrid state-space scan on an L40S with driver 550.54.14. These checks use the node's existing driver libraries. Broader quality tests, HTTP streaming, optimized serving settings, latency, throughput, GPU-resource measurements and actual-charge comparisons remain to be performed under the shared baseline protocol.

The reusable scripts include three setup corrections. The official CPU TorchCodec wheel resolves the original media import's `libnvrtc.so.13` requirement. The environment's executable directory supplies Ninja, and the matching official FlashInfer cache wheel supplies precompiled CUDA 12.9 kernels for sampling warmup. The prompt helper explicitly sets `return_dict=False` because Transformers 5.13 returns a dictionary by default. The earlier attempts stop at these setup boundaries; job `13930616` verifies the completed configuration.

## Reproduce the software selection

The [released H-Spec installation script](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/apply.sh) specifies the vLLM and Speculators source revisions, both patches and Python 3.13. The local [setup script](setup_hspec.sh) applies those patches to separate source copies, selects the official CUDA 12.9 wheel explicitly and records the installed package versions. The `external/` checkouts remain available as the unchanged upstream sources.

The pinned vLLM installation text describes CUDA 12.9 as its default, while the same revision's [`VLLM_MAIN_CUDA_VERSION` setting](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/envs.py#L564) defaults to `13.0`. The explicit `cu129` artifact URL, digest and Torch constraints determine the runtime installed here.

| Component | Selected artifact |
|---|---|
| H-Spec source | `de911876905440bfd8edfa78b482050f7ac0279c` from [weifanjiang/H-Spec](https://github.com/weifanjiang/H-Spec/tree/de911876905440bfd8edfa78b482050f7ac0279c). |
| vLLM source | `752a3a504485790a2e8491cacbb35c137339ad34`, the [v0.25.1 release](https://github.com/vllm-project/vllm/tree/752a3a504485790a2e8491cacbb35c137339ad34), plus the released H-Spec patch. |
| Speculators source | `3251ed020cfc954717c1d075d83dca8721713f0c` from [vllm-project/speculators](https://github.com/vllm-project/speculators/tree/3251ed020cfc954717c1d075d83dca8721713f0c), plus the released H-Spec patch. |
| Python | Managed CPython `3.13.18`. |
| PyTorch packages | `torch==2.11.0+cu129`, `torchaudio==2.11.0+cu129`, `torchvision==0.26.0+cu129`, from the official PyTorch CUDA 12.9 wheel index. |
| Transformers | `5.13.0`, within the checked requirements of vLLM and Speculators. |
| TorchCodec | `0.17.0+cpu` from the [official PyTorch CPU wheel index](https://download.pytorch.org/whl/cpu/torchcodec/). Text inference uses this CPU media dependency with the selected CUDA 12.9 Torch runtime. |
| FlashInfer packages | `flashinfer-python==0.6.13`, `flashinfer-cubin==0.6.13` and `flashinfer-jit-cache==0.6.13+cu129`. The [official CUDA 12.9 cache wheel](https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.13/flashinfer_jit_cache-0.6.13%2Bcu129-cp39-abi3-manylinux_2_28_x86_64.whl) supplies precompiled sampling kernels. |
| vLLM compiled libraries | The official [v0.25.1 CUDA 12.9 wheel](https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl), reused by the patched Python installation. |
| Local package version | `0.25.1+cu129.hspec`, supplied through vLLM's supported `VLLM_VERSION_OVERRIDE`. The source and patch hashes identify the exact modified code. |
| Speculators package version | `0.6.0.dev0`, built from the pinned source revision and author patch recorded above. |

The installer verifies these SHA-256 digests before applying a patch or using the binary wheel:

```text
vllm.patch        2c59f719bcd1d5747971575f1e65f2770c05891487b353f38d72225f0ec52bdd
speculators.patch 8fd05dd3fbed949b47adc81aef9aeb1726011347015ffaaeecedc839fe472860
vLLM CUDA12.9 wheel 9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431
FlashInfer CUDA12.9 cache 0af546a6a0db3470b0a52b43c8158fc85358852ab1b0a8a42d83a411f0aaa369
```

The authors identify `causal-conv1d` and `mamba_ssm` as optional training kernels in `apply.sh`. Serving uses vLLM's implementation. The setup omits these optional packages and executes inference-package checks.

## Install in a CPU allocation

The commands use Ayo's verified scratch location. Each teammate sets `HPML_ROOT` to their own approved private scratch directory and prepares the same source revisions there. The script checks directory ownership before creating an environment.

1. After connecting to Insomnia, initialize the login shell and select your private project directory.

   ```bash
   bash -l
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   ```

2. For a fresh environment with the verified public source checkouts and `uv 0.13.8` bootstrap available, submit the CPU installation.

   ```bash
   HPML_SETUP_JOB_ID=$(sbatch --parsable --account=edu --partition=short \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G \
     --time=02:00:00 --job-name=hpml-hspec-setup \
     --output="$HPML_ROOT/runs/hspec-setup-%j.log" \
     --wrap="bash -l $HPML_ROOT/source/hpml-project/baselines/setup_hspec.sh")
   printf '%s\n' "$HPML_SETUP_JOB_ID"
   ```

   The installer creates `envs/hspec-cu129-py313` and `source/hspec-cu129-py313`. Eight CPU cores support the requested 48G of host RAM within the partition's 6,400 MB per-CPU limit. The script requires a compute allocation, checks source revisions and preserves existing setup directories. An interrupted installation can resume after you inspect its log by adding `--resume` to the script command.

3. After the job finishes, inspect its Slurm status and saved package record.

   ```bash
   sacct -j "$HPML_SETUP_JOB_ID" --format=JobID,State,ExitCode,NodeList,Elapsed
   cat "$HPML_ROOT/envs/hspec-cu129-py313/hpml-setup.json"
   ```

   `hpml-installed.txt` records all resolved package versions. The reusable [dependency constraints](hspec-requirements.txt) preserve dependency versions; the installer builds vLLM and Speculators from the separately recorded patched sources. `hpml-setup.json` records Python, CUDA runtime, package versions, source revisions, patch hashes and patched-module imports. Keep these records with results from this environment.

## Check the allocated GPU runtime

The inspected L40S node uses NVIDIA driver 550.54.14. CUDA 12.9 packages can use CUDA 12.x minor-version compatibility on this driver, subject to the package's kernel and compiler requirements. The basic GPU check exercises CUDA initialization, BF16 matrix multiplication, vLLM's compiled RMS-normalization operator, a compiled Torch operation and H-Spec's hybrid state-space scan with a known expected result. Full model loading additionally exercises attention, speculative decoding and the pretrained model's dimensions. [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

1. After successful CPU installation, submit the GPU runtime check.

   ```bash
   HPML_GPU_CHECK_JOB_ID=$(sbatch --parsable --account=edu --partition=short \
     --nodes=1 --ntasks=1 --gres=gpu:l40s:1 \
     --cpus-per-task=8 --mem=48G --time=00:20:00 \
     --job-name=hpml-hspec-gpu-check \
     --output="$HPML_ROOT/runs/hspec-gpu-check-%j.log" \
     --wrap="bash -l $HPML_ROOT/source/hpml-project/baselines/setup_hspec.sh --check-gpu")
   printf '%s\n' "$HPML_GPU_CHECK_JOB_ID"
   ```

2. After the job finishes, inspect its Slurm status and GPU record.

   ```bash
   sacct -j "$HPML_GPU_CHECK_JOB_ID" --format=JobID,State,ExitCode,NodeList,Elapsed
   cat "$HPML_ROOT/envs/hspec-cu129-py313/hpml-gpu-check.json"
   ```

   A completed basic check establishes the tested operations on the recorded GPU. The later model-loading check establishes the target/drafter combination and the selected runner.

## Qualify the three pretrained methods

The target is `meta-llama/Llama-3.1-8B-Instruct` at `0e9e39f249a16976918f6564b8830bc894c89659`. Preserve the canonical weights, tokenizer, generation configuration and stop-token IDs. The following draft revisions belong to the H-Spec authors' release:

| Method | Released draft | Required runner and method settings |
|---|---|---|
| P-EAGLE reproduction | [`weifanjiang/llama3.1-8b-it.speculators.peagle`](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/tree/0db2ae706f7a314286516835006183efdcb5bb39), revision `0db2ae706f7a314286516835006183efdcb5bb39`. | `VLLM_USE_V2_MODEL_RUNNER=0`; `method:eagle3`; `parallel_drafting:true`; seven speculative tokens. |
| H-Spec | [`weifanjiang/llama3.1-8b-it.speculators.hspec`](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.hspec/tree/a43cf8a437816ddf894270ef03b23520dedcde88), revision `a43cf8a437816ddf894270ef03b23520dedcde88`. | `VLLM_USE_V2_MODEL_RUNNER=1`; `method:mamba_attn_hybrid`; `enable_kv_sharing:true`; seven speculative tokens. |
| DSpark reproduction | [`weifanjiang/llama3.1-8b-it.speculators.dspark`](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.dspark/tree/110d99b89501e3007a893da937d9560e087243c0), revision `110d99b89501e3007a893da937d9560e087243c0`. | `VLLM_USE_V2_MODEL_RUNNER=1`; `method:dspark`; seven speculative tokens. |

P-EAGLE's draft has four layers and FFN intermediate dimension 14,336. Attribute this checkpoint to the H-Spec authors' Llama reproduction. The current comparison measures P-EAGLE, H-Spec and DSpark with their weights and architectures fixed. Each method has a matched ordinary-generation control in its own model runner. [Released method table](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/README.md#L85).

The P-EAGLE reproduction's target-feature indices are `[1,8,15,22,30]`. The original P-EAGLE paper describes three layers, `2`, `L/2` and `L−1`, using its own indexing convention. Keep the prepared checkpoint's five indices in the configuration record and identify the H-Spec-author reproduction in every result. [Pinned checkpoint configuration](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/blob/0db2ae706f7a314286516835006183efdcb5bb39/config.json); [P-EAGLE paper, Figure 2](https://arxiv.org/abs/2602.01469).

Keep vLLM 0.25.1 and the recorded patches for these checks. The later [Model Runner V1 removal](https://github.com/vllm-project/vllm/pull/63373) removes the runner required by this P-EAGLE route. The older V0 engine has its own history; the checked P-EAGLE path uses Model Runner V1 within the V1 engine.

After the complete Meta-sourced target snapshot and draft snapshots are verified, load each runner in its own process, first with ordinary generation and then with its specified drafter. Inside the allocated GPU job, activate `"$HPML_ROOT/envs/hspec-cu129-py313/bin/activate"` and set `HF_HOME="$HPML_ROOT/cache/huggingface"`. Use the same target, dtype, prompts, generation limits and stopping rules within each runner. Compare returned greedy token IDs and stop reasons before collecting performance data. Record any load failure, unsupported kernel or output mismatch with the exact configuration. The [core-method guide](core-methods.md#run-p-eagle-and-h-spec-in-the-patched-vllm-environment) contains the released server launch templates.

The later benchmark measures both research questions: faster returned output at preserved quality, and fewer GPU resources and lower actual GPU charges per million returned output tokens while preserving quality and response speed. A successful installation supplies the runtime needed for those measurements. Its installation duration and small-kernel checks provide setup evidence.

## Run the shared four-prompt qualification

The [qualification script](qualify_hspec.py) uses the same named prompts, chat template, canonical stop IDs and output caps as [the EAGLE check](qualify_eagle.py). The prompt names are `short_stop`, `length_cap`, `arithmetic` and `code`; their caps are 32, 16, 32 and 32 generated tokens. Each ordinary-generation run and speculative run starts in its own process. P-EAGLE uses its ordinary Model Runner V1 result. H-Spec and DSpark use the ordinary Model Runner V2 result. The script also compares the two ordinary runners when both complete.

The setup check selects BF16, FlashAttention 2, tensor parallelism of one, a 512-token context limit, one active sequence, a 512-token batched-token limit and GPU memory utilization of 0.65. Chunked prefill uses the release's supported enabled setting; every setup prompt fits within one 512-token prefill batch. Prefix caching, Torch model compilation and CUDA graphs are disabled for this initial loading check. These explicit settings keep the setup requests small. The later optimized serving configuration receives its own correctness checks before performance measurement.

1. After the basic GPU check passes and the complete direct-Meta target provenance exists, submit the model qualification job.

   ```bash
   HPML_MODEL_JOB_ID=$(sbatch --parsable --account=edu --partition=short \
     --nodes=1 --ntasks=1 --gres=gpu:l40s:1 \
     --cpus-per-task=16 --mem=64G --time=01:00:00 \
     --job-name=hpml-hspec-model-check \
     --output="$HPML_ROOT/runs/hspec-model-check-%j.log" \
     --wrap="bash -l $HPML_ROOT/source/hpml-project/baselines/setup_hspec.sh --check-models")
   printf '%s\n' "$HPML_MODEL_JOB_ID"
   ```

2. After the job finishes, inspect its status and method results.

   ```bash
   sacct -j "$HPML_MODEL_JOB_ID" --format=JobID,State,ExitCode,NodeList,Elapsed
   cat "$HPML_ROOT/runs/hspec-models-$HPML_MODEL_JOB_ID/summary.json"
   ```

The raw producer token list is the unmodified `vLLM CompletionOutput.token_ids` after the engine's native scheduler stopping. The list preserves a terminal stop token when the engine emits one. The project output policy then preserves the first canonical stop boundary or strict output cap and excludes a terminal stop token from returned tokens. The record keeps both lists and the engine's original finish/stop reason. Native readiness also requires matching engine-returned text. Internal speculative proposals and verifier tokens discarded by the engine require additional cycle instrumentation for the shared baseline measurements. [Native output construction](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/output_processor.py#L375), [stop-token handling](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/engine/detokenizer.py#L107).

The [audit worker](hspec_audit_worker.py) delegates every weight load to the released code. Its records include every observed checkpoint key, shape and dtype; automatic-loader inputs before their optional name mapper; loaded destination names; effective skip/ignore rules; and the final model's parameter/buffer inventory. A worker inspection records actual runner classes, attention implementations and parameters shared with the target. Checkpoint-input coverage, native loader completion and exact tensor-value identity have separate meanings. The helper records the first two; exact loaded-value comparisons require additional checks. Some released drafter loaders return no complete destination set, so review their recorded skip rules and shared target parameters before describing a full parameter-coverage result.

`summary.json` reports native output readiness and adapted output readiness separately. Four matching setup prompts establish the observed behavior for these requests. The shared baseline procedure adds broader correctness checks, explicitly recorded serving settings and streaming measurements. The team selects and records the evaluation prompts before timing.

## Keep the reproducible setup and results

The installed environment remains at `$HPML_ROOT/envs/hspec-cu129-py313`. Its `hpml-setup.json`, `hpml-gpu-check.json` and `hpml-installed.txt` retain package, source, patch, wheel and GPU-operation evidence. The patched source copies remain at `$HPML_ROOT/source/hspec-cu129-py313`; the Speculators installation uses its source copy directly. The unchanged upstream checkouts and verified model snapshots also remain available.

The full successful model records remain at `$HPML_ROOT/runs/hspec-models-13930616`, including checkpoint-key inventories, loader traces and complete worker state. The [local concise evidence directory](results/hspec-output-check-13930616) contains prompt outputs, comparisons, requested settings, allocation metadata, a compact runtime summary and environment records. References inside the saved outputs identify the full cluster records. The three successful setup, GPU-check and model-check logs remain in `runs/`.

Cleanup removes the downloaded wheel files, generated vLLM build directory, superseded setup-attempt logs and the early sampling-warmup cache. The installed packages, shared reusable package caches, final evidence and required sources remain available. Every owned Slurm job has exited. The final correctness result measures these four prompts; performance and cost results require the shared baseline measurement procedure.
