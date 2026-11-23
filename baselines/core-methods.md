# Run the core pretrained baselines

The selected central benchmark methods are **P-EAGLE, EAGLE-3, DSpark and DFlash**. EAGLE-3 uses the prepared SGLang path for the central serving comparison; its original research implementation remains a secondary reference. H-Spec and PARD-2 remain documented as preparation history and related work. The central comparison requires ordinary SGLang, ordinary vLLM Model Runner V1 and ordinary vLLM Model Runner V2.

Checked on November 16, 2026. This guide identifies the released checkpoints, source revisions and software environments for P-EAGLE, H-Spec, DSpark, DFlash, EAGLE-3, PARD and PARD-2. The linked method guides record completed setup checks and their remaining research measurements. Follow the [Insomnia setup guide](insomnia.md) for access, storage and GPU allocation.

The **target model** defines the output that the study preserves. A **drafter** proposes future tokens that the target verifies. **Ordinary generation** loads the target and generates one token at a time. **Multi-token prediction (MTP)** predicts several future token positions from an available context. A **checkpoint** contains a model's saved weights and configuration. The current work benchmarks released methods using existing pretrained weights. The team keeps those weights fixed while establishing comparable measurements.

A **key/value (KV) cache** stores attention state from previous token positions. A **feed-forward network (FFN)** transforms token features through an intermediate set of channels. **Time to first token (TTFT)** measures the interval from request submission to the first returned token; **inter-token latency (ITL)** concerns the intervals during subsequent output. The **dtype** specifies the numerical representation used by the model, such as float16 or bfloat16.

The benchmarks answer two questions: which qualified system returns the same-quality output faster, and which uses fewer GPU resources and lower actual GPU charges while preserving output quality and response speed. Record allocated GPU-seconds and attributable GPU charges per million returned output tokens. The [Insomnia resource instructions](insomnia.md#capture-resource-use-and-charges) define the allocation interval and charge evidence.

The reviewed artifacts support pretrained inference comparisons with `meta-llama/Llama-3.1-8B-Instruct` as the common target. PARD-2's released evaluation configuration names an Unsloth copy. The completed local qualification uses the verified direct-Meta snapshot; preserve that exact target choice in subsequent runs.

## Choose a released implementation

| Candidate | Concrete pretrained route | Qualification needed |
|---|---|---|
| P-EAGLE, H-Spec authors' Llama reproduction | H-Spec authors' Llama reproduction, patched vLLM 0.25.1, Model Runner V1, `eagle3` plus `parallel_drafting:true`. Job `13930616` passes loading and four-prompt native output checks on an L40S. | Preserve the [qualified configuration](hspec-setup.md), ordinary V1 control and checkpoint provenance. Broader quality, streaming and optimized performance checks belong to the shared baseline measurements. |
| H-Spec | Authors' Llama checkpoint, the same pinned patches, Model Runner V2, `mamba_attn_hybrid`, target KV sharing. Job `13930616` passes loading and four-prompt native output checks. | Preserve the ordinary V2 control and qualified configuration. Measure KV storage and GPU resources during the later profiling runs. |
| DSpark, fixed-length Llama reproduction | H-Spec authors' checkpoint, the same patched environment, Model Runner V2 and `method:dspark`, with seven speculative tokens. Job `13930616` passes loading and four-prompt native output checks. | Preserve its ordinary V2 control and fixed-length label. The pinned serving loader skips confidence-head weights; reproducing the original paper's load-aware verification scheduler requires a separate implementation check. |
| DFlash | Official UltraChat Llama checkpoint; the pinned SGLang 0.5.19 CUDA 12.9 image passed loading and four-prompt native output checks on an L40S. The current DFlash repository also has a Transformers inference path. | Use the [qualified SGLang setup](sglang-setup.md) and preserve its exact environment and ordinary control. Broader quality, streaming and performance checks follow the study protocol. |
| EAGLE-3 | Original SafeAILab code with the original checkpoint; the SGLang architecture-converted checkpoint passed four-prompt native output checks in the pinned 0.5.19 CUDA 12.9 image. | Keep the original-code and SGLang results identifiable. Use the [qualified SGLang setup](sglang-setup.md) and target-only ordinary processes for each environment. |
| PARD-2 | AMD's custom Transformers inference script, released drafter plus `warp_model.bin`, and the verified direct-Meta target. Job `13929467` passes loading and all four adapted-output comparisons. | Preserve native overshoot and the shared output adapter separately. Qualify compiled full-model execution and a streaming/concurrent serving path before those measurements. |
| PARD | AMD's custom script; historical vLLM parallel draft-model path. | Pin a runner that retains PARD support and choose a compatible target/drafter pair. Treat PARD separately from PARD-2. |

The present goal is to benchmark each compatible released method under shared target, prompt, hardware and measurement conditions. Ordinary generation in each engine supplies its control. The results will describe the unchanged methods and support a subsequent discussion of research questions and architecture choices.

## Preserve the exact source and model revisions

### Source revisions

| Source | Revision |
|---|---|
| [H-Spec](https://github.com/weifanjiang/H-Spec/tree/de911876905440bfd8edfa78b482050f7ac0279c) | `de911876905440bfd8edfa78b482050f7ac0279c` |
| [Speculators, H-Spec's base](https://github.com/vllm-project/speculators/tree/3251ed020cfc954717c1d075d83dca8721713f0c) | `3251ed020cfc954717c1d075d83dca8721713f0c` plus H-Spec's patch |
| [vLLM 0.25.1, H-Spec's base](https://github.com/vllm-project/vllm/tree/752a3a504485790a2e8491cacbb35c137339ad34) | `752a3a504485790a2e8491cacbb35c137339ad34` plus H-Spec's patch |
| [DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | `07ebd93db9f472af339b644bb70221ad8428328a` |
| [EAGLE](https://github.com/SafeAILab/EAGLE/tree/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b) | `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b` |
| [PARD/PARD-2](https://github.com/AMD-AGI/PARD/tree/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038) | `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038` |
| [SGLang v0.5.20](https://github.com/sgl-project/sglang/tree/94602c9c2b7cbdb8efd5c52802dac6a1c180089e) | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` |

SGLang's latest release checked on this date is [v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23). The existing study's v0.5.20 choice remains a reproducible version to qualify. Changing it creates a new software configuration that needs its own ordinary-generation control.

### Model revisions and stored weight sizes

The Hugging Face model API exposed the following revisions and weight-file metadata. The API endpoints use `https://huggingface.co/api/models/OWNER/REPOSITORY?blobs=true`. Each model link fixes the inspected revision.

| Model artifact | Exact revision | Listed weight-file size |
|---|---|---|
| [Meta Llama-3.1-8B-Instruct target](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659) | `0e9e39f249a16976918f6564b8830bc894c89659` | Four safetensors shards total 16,060,556,376 bytes. |
| [Official DFlash UltraChat drafter](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/tree/d3af30def9601abdd10810aba220d692f0e803f0) | `d3af30def9601abdd10810aba220d692f0e803f0` | `model.safetensors`: 2,097,259,104 bytes. |
| [Original EAGLE-3 drafter](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/tree/ada412b672e293d682423de84a095447bf38a637) | `ada412b672e293d682423de84a095447bf38a637` | `pytorch_model.bin`: 849,768,795 bytes. |
| [SGLang EAGLE-3 conversion](https://huggingface.co/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/tree/e5ed08d66f528a95ce89f5d4fd136a28f6def714) | `e5ed08d66f528a95ce89f5d4fd136a28f6def714` | `pytorch_model.bin`: 849,768,795 bytes. |
| [PARD-2 Llama drafter](https://huggingface.co/amd/PARD2-Llama-3.1-8B/tree/a509e315bfe79a0be8b832004bb3ee14acfdf86a) | `a509e315bfe79a0be8b832004bb3ee14acfdf86a` | `model.safetensors`: 2,996,982,488 bytes; `warp_model.bin`: 67,110,526 bytes. |
| [PARD Llama drafter](https://huggingface.co/amd/PARD-Llama-3.2-1B/tree/f7fbf130e2569268b054ffa2234303070ae82f9b) | `f7fbf130e2569268b054ffa2234303070ae82f9b` | `model.safetensors`: 2,996,982,344 bytes. |
| [H-Spec authors' P-EAGLE Llama reproduction](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/tree/0db2ae706f7a314286516835006183efdcb5bb39) | `0db2ae706f7a314286516835006183efdcb5bb39` | `model.safetensors`: 3,276,266,144 bytes. |
| [H-Spec Llama drafter](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.hspec/tree/a43cf8a437816ddf894270ef03b23520dedcde88) | `a43cf8a437816ddf894270ef03b23520dedcde88` | `model.safetensors`: 3,696,914,682 bytes. |

These sizes describe stored tensors. GPU peak memory also includes target and drafter key/value caches, activations, CUDA graph pools, attention workspaces, allocator reserves and serving capacity. A weight-size sum supplies an initial storage and resident-weight estimate; the loading pilot establishes measured GPU memory.

The target API reports `gated:manual`. The unauthenticated target config request returned HTTP 401. The inspected drafter repositories are public and ungated. Each teammate needs approved Meta-model access and read access on the compute node that downloads the target. Model access remains separate from permission to use Insomnia. The target tokenizer, chat template, generation configuration and stop-token IDs belong to the pinned target snapshot.

### Configuration details that matter

- **P-EAGLE reproduction:** `PEagleDraftModel`, four layers, hidden dimension 4,096, FFN intermediate dimension 14,336, draft vocabulary 32,000, target feature layers `[1,8,15,22,30]`, sliding window 2,048. Its configuration explicitly names `meta-llama/Llama-3.1-8B-Instruct`. The saved `speculators 0.7.0.dev98` and `transformers 5.15.0` strings describe artifact metadata; the runnable environment is the patched stack above. [Pinned config](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/blob/0db2ae706f7a314286516835006183efdcb5bb39/config.json).
- **H-Spec:** `MambaAttnHybridDraftModel`, block size eight with seven speculative positions, target-feature layers `[1,8,15,22,30]`, target KV layers `[15,22,30]`, rank-256 Markov component and confidence head. The configuration names the same Meta target. [Pinned config](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.hspec/blob/a43cf8a437816ddf894270ef03b23520dedcde88/config.json).
- **DFlash:** five layers, hidden dimension 4,096, FFN intermediate dimension 12,288, block size ten and target feature layers `[1,8,15,22,29]`. The block contains an already available anchor plus future proposals. Use the target's stopping rules: the drafter config contains model-family metadata whose EOS fields differ from Llama's target stop rules. [Pinned config](https://huggingface.co/z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat/blob/d3af30def9601abdd10810aba220d692f0e803f0/config.json).
- **EAGLE-3:** one draft layer, hidden dimension 4,096, FFN intermediate dimension 14,336 and draft vocabulary 32,000. The original and SGLang configurations identify different architecture class names. Preserve the artifact ID used by each engine. [Original config](https://huggingface.co/yuhuili/EAGLE3-LLaMA3.1-Instruct-8B/blob/ada412b672e293d682423de84a095447bf38a637/config.json), [SGLang config](https://huggingface.co/jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B/blob/e5ed08d66f528a95ce89f5d4fd136a28f6def714/config.json).
- **PARD-2:** 16 layers, hidden dimension 2,048, FFN intermediate dimension 8,192, target-feature dimension 16,384 and target layers `[-1,-8,-16,-24]`. Retain the adapter file with the backbone. The inference code explicitly casts loaded models to its selected dtype. [Pinned config](https://huggingface.co/amd/PARD2-Llama-3.1-8B/blob/a509e315bfe79a0be8b832004bb3ee14acfdf86a/config.json).

The original P-EAGLE paper's Figure 2 describes three target-feature layers, labeled `2`, `L/2` and `L−1` in the paper's convention. The H-Spec authors' prepared Llama checkpoint specifies five indices, `[1,8,15,22,30]`. Preserve the checkpoint's exact indices and model identity in results. The reproduction has its own architecture configuration and training provenance. [P-EAGLE, Figure 2](https://arxiv.org/abs/2602.01469); [pinned reproduction config](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.peagle/blob/0db2ae706f7a314286516835006183efdcb5bb39/config.json).

For DSpark and H-Spec, distinguish checkpoint contents from components executed during serving. Both released checkpoints include confidence-head parameters. The pinned DSpark loader explicitly skips `confidence_head`, and the H-Spec serving patch applies the same skip rule. Both checked configurations use seven speculative positions and their active Markov heads. The DSpark configuration therefore measures a fixed-length reproduction; the original paper's confidence-based, load-aware prefix selection requires a separately qualified implementation. [DSpark loader, lines 172–184](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/model_executor/models/qwen3_dspark.py#L172-L184); [H-Spec serving patch, lines 1190–1204](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/vllm.patch#L1190-L1204); [prepared method settings](qualify_hspec.py).

## Run P-EAGLE and H-Spec in the patched vLLM environment

The H-Spec authors explicitly release their own P-EAGLE, DFlash and DSpark reproductions alongside H-Spec. Their checkpoint table names Llama-3.1-8B-Instruct, Qwen3-4B and Qwen3-8B. Attribute the Llama P-EAGLE artifact to this reproduction. Its training provenance describes the available artifact; the team's work starts by loading the completed weights. [H-Spec README, lines 85–106](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/README.md#L85).

Use the [Insomnia H-Spec setup](hspec-setup.md) to reproduce the tested environment. Its reusable script applies the author patches to private source copies in a CPU allocation and selects the official CUDA 12.9 vLLM wheel explicitly. The recorded stack uses Python 3.13.18, Torch 2.11.0+cu129, Transformers 5.13.0, patched vLLM 0.25.1+cu129.hspec, patched Speculators 0.6.0.dev0, CPU TorchCodec 0.17.0+cpu and FlashInfer 0.6.13 with its CUDA 12.9 precompiled kernel cache. A separate one-GPU allocation performs loading and output checks.

The authors' `apply.sh` supplies the source revisions and patches used by this setup. Its `SKIP_MAMBA=1` option omits `causal-conv1d` and `mamba_ssm`, which the authors label optional training kernels. The checked serving path uses vLLM's implementations. [Installation script, lines 10–25 and 70–116](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/apply.sh#L10).

After you obtain model access and download the recorded revisions, replace each quoted path placeholder with its local checkpoint directory. The following templates identify the released HTTP-server entry point. The completed four-prompt checks use vLLM's Python `LLM` entry point; the HTTP server, streaming and concurrent-arrival tests receive their own qualification before the corresponding baseline measurements.

```bash
# H-Spec, Model Runner V2.
VLLM_USE_V2_MODEL_RUNNER=1 python -m vllm.entrypoints.openai.api_server \
  --model "<llamaTargetPath>" \
  --speculative-config '{"model":"<hspecDraftPath>","method":"mamba_attn_hybrid","num_speculative_tokens":7,"enable_kv_sharing":true}'

# P-EAGLE reproduction, Model Runner V1.
VLLM_USE_V2_MODEL_RUNNER=0 python -m vllm.entrypoints.openai.api_server \
  --model "<llamaTargetPath>" \
  --speculative-config '{"model":"<peagleDraftPath>","method":"eagle3","parallel_drafting":true,"num_speculative_tokens":7}'
```

The H-Spec README supplies the method settings. The vLLM 0.25.1 source explicitly lists parallel EAGLE drafting as unsupported by Model Runner V2. [README](https://github.com/weifanjiang/H-Spec/blob/de911876905440bfd8edfa78b482050f7ac0279c/README.md#L97), [runner compatibility check, lines 2081–2088](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/config/vllm.py#L2081).

Launch ordinary generation separately under each runner, with the same pinned target and matching dtype, backend, memory settings and request protocol. Omit the speculative configuration in those ordinary processes. A P-EAGLE-versus-H-Spec result then compares complete deployments containing different runner implementations. The corresponding ordinary runs establish each deployment's improvement within its runner. Record the runner as part of each complete configuration.

The paper evaluates one NVIDIA A100 80GB, uses seven draft positions by default and reports single-request acceptance and ITL speedup plus throughput/KV utilization under fixed concurrency. Its sampled main settings differ from the proposed greedy output check. The paper's three-run serving summaries describe fixed-concurrency tests, while the team also needs identical predetermined request arrivals. [H-Spec §4 and appendix C](https://arxiv.org/html/2609.24197v1).

A compatible Llama DSpark reproduction also exists: [`weifanjiang/llama3.1-8b-it.speculators.dspark`](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.dspark/tree/110d99b89501e3007a893da937d9560e087243c0), revision `110d99b89501e3007a893da937d9560e087243c0`. The H-Spec README names `method:dspark` for that column. Job `13930616` loads this checkpoint and matches ordinary Model Runner V2 on all four native token-ID, text, stopping and cap checks. The [comparison summary](results/hspec-output-check-13930616/summary.json) records all three paired results and the matching ordinary V1/V2 outputs.

Label that result **DSpark, H-Spec authors' Llama reproduction, fixed proposal length**. The executed path runs a parallel backbone followed by sequential Markov sampling for the configured number of positions. A count of checkpoint keys presented to a loader records its inputs; the loader's skip rules and executed model components establish which parameters contribute to inference. [DSpark sampling loop](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/v1/worker/gpu/spec_decode/dspark/speculator.py#L110-L159).

### Preserve the model runner required by P-EAGLE

A **model runner** prepares model inputs, invokes GPU execution and updates request state. The runner determines which speculative-decoding implementations the engine can execute.

[PR 63373](https://github.com/vllm-project/vllm/pull/63373) merged on 7 November 2026; its merge commit is [`aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592`](https://github.com/vllm-project/vllm/commit/aac1b2a4f36e742c1f55bcb34ca9aaa5e6e02592). The change removes Model Runner V1 within the V1 engine. The historical V0 engine is a separate engine generation. The removal drops the old P-EAGLE/PARD parallel-drafting path, makes `VLLM_USE_V2_MODEL_RUNNER=0` fail and retains dedicated DFlash/DSpark paths in Model Runner V2.

[PR 44816](https://github.com/vllm-project/vllm/pull/44816), the P-EAGLE Model Runner V2 port, remains open and unmerged at the checked date. Its inspected head is `d9662e87d1f144624c73bfd24aada608d163c727`. Its reported test concerns Qwen3-8B and a different drafter. It supplies engineering evidence for a port, with Llama correctness and performance still requiring qualification.

The recorded H-Spec stack supplies the P-EAGLE reproduction path that passes the four-prompt Insomnia check. Installing current vLLM main changes that path. Version 0.25.1's GPU documentation specifies compute capability 7.5 or higher and describes CUDA 12.9 binaries as its default. The same revision's `VLLM_MAIN_CUDA_VERSION` setting defaults to `13.0`. The [Insomnia H-Spec setup](hspec-setup.md) selects the published `vllm-0.25.1+cu129` wheel explicitly and verifies its SHA-256 digest. Jobs `13930196` and `13930616` establish the recorded small-kernel and pretrained-output results on driver 550.54.14. [Pinned vLLM GPU requirements](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/docs/getting_started/installation/gpu.cuda.inc.md#L4), [pinned CUDA default setting](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/vllm/envs.py#L564).

## Run DFlash and EAGLE-3 through SGLang

SGLang v0.5.20 documents both exact Llama pairings:

```bash
python3 -m sglang.launch_server \
  --model "<llamaTargetPath>" \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path "<dflashDraftPath>"

python3 -m sglang.launch_server \
  --model "<llamaTargetPath>" \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path "<sglangEagle3DraftPath>" \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 4 \
  --speculative-num-draft-tokens 16
```

The EAGLE-3 draft settings reproduce the documentation's example. They are starting settings for qualification. The documentation additionally sets float16, static memory fraction 0.7 and a CUDA-graph batch cap of eight. Record and match the actual dtype and resource settings in ordinary and speculative deployments. SGLang's DFlash setting `--speculative-num-draft-tokens` denotes verify block size and has alias `--speculative-dflash-block-size`; its default comes from the checkpoint when available. Tree capacity, sequential draft steps and DFlash block size count different structures. [EAGLE-3 example, lines 370–390](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/docs/docs/advanced_features/speculative_decoding.mdx#L370), [DFlash example and parameter definitions, lines 528–580](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/docs/docs/advanced_features/speculative_decoding.mdx#L528).

**Cluster compatibility:** the v0.5.20 installation instructions use CUDA 13 and identify v0.5.19 as the final CUDA 12 release line. Its Python package pins Torch 2.13.0 and Transformers 5.12.1 and uses CUDA 13 FlashInfer dependencies. Preserve a dedicated SGLang environment. Capture GPU model, compute capability, driver version, Python version and resolved dependencies before its loading pilot. The documentation's FlashInfer support starts at NVIDIA compute capability 7.5, also called sm75. Check the assigned GPU's compute capability before using that backend. [Pinned install instructions](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/docs/docs/get-started/install.mdx), [Pinned package dependencies](https://github.com/sgl-project/sglang/blob/94602c9c2b7cbdb8efd5c52802dac6a1c180089e/python/pyproject.toml).

The saved DFlash paper's SGLang Llama comparison uses a B200 and FlashInfer; that reported hardware establishes a tested configuration in the paper. The paper's speedup supplies a candidate-selection reason. Measurements on the team's GPU and software determine which candidate supplies the speed and cost reference. [DFlash paper](https://arxiv.org/abs/2602.06036).

## Match the runtime to the confirmed L40S node

The November 16 allocation `13928964` on `ins056` reports an NVIDIA L40S with 46,068 MiB of GPU memory and compute capability 8.9. Its driver is 550.54.14, and `nvidia-smi` reports CUDA 12.4 support. The available `cuda/12.3` toolkit provides `nvcc` 12.3.107. Apptainer is 1.4.5 and system Python is 3.9.18. These measurements describe this allocation; record the same values for every subsequent node. The [Insomnia setup guide](insomnia.md) retains the live access and allocation evidence.

The driver supports CUDA 12.4 directly. NVIDIA lists driver 550.54.14 with CUDA 12.4 GA. CUDA 12.x minor-version compatibility also covers sufficiently new drivers in the 525-and-later family, subject to feature restrictions. Newer PTX intermediate code and features that require a newer driver can still fail. CUDA 13's ordinary driver requirement starts at the 580 family. [CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html), [CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

| Configuration | Concrete preparation path | Qualification on driver 550.54.14 |
|---|---|---|
| Original EAGLE-3 | Python 3.10.22, PyTorch 2.6.0 from the official `cu124` index, Transformers 4.53.1 and the recorded EAGLE source revision. Install the repository with `--no-deps` after its selected dependencies. | Job `13929466` passes model loading and four-prompt equality after the shared output adapter. The released loop produces tokens beyond native stop/cap boundaries; preserve its raw output and the adapter's returned output separately. |
| SGLang 0.5.20 | Keep the original source revision `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` available. Its reviewed dependency set uses CUDA 13. | Driver 550 requires an explicitly qualified forward-compatibility deployment for this CUDA 13 path, or a different node with a compatible driver. |
| SGLang 0.5.19 CUDA 12.9 fallback | Use the separate source revision and exact `cu129` container identified in the following table. The release includes the Llama DFlash pairing and EAGLE-3. | Job `13930204` passed small GPU operations on driver 550.54.14. Job `13930206` loaded all three methods and passed both four-prompt native token/stopping comparisons. Preserve the [exact setup and evidence](sglang-setup.md). |
| Patched vLLM 0.25.1 for P-EAGLE, H-Spec and DSpark | Preserve vLLM `752a3a504485790a2e8491cacbb35c137339ad34`, the H-Spec patches and the [qualified CUDA 12.9 environment](hspec-setup.md). | Job `13930196` passes small GPU operations. Job `13930616` loads both ordinary runners and all three drafters and passes every four-prompt native comparison on an L40S with driver 550.54.14. The checks use the installed node driver libraries. HTTP serving requires its own output check before performance measurements. |

PyTorch's official previous-version instructions provide the EAGLE-compatible CUDA 12.4 build: `torch==2.6.0`, `torchvision==0.21.0` and `torchaudio==2.6.0` use `https://download.pytorch.org/whl/cu124`. Install the packages needed by the chosen EAGLE entry point and preserve the resolved versions. The node's CUDA 12.3 toolkit remains a separate compiler installation. [Official PyTorch versions](https://pytorch.org/get-started/previous-versions/#v260), [vLLM 0.25.1 release assets](https://github.com/vllm-project/vllm/releases/tag/v0.25.1), [vLLM CUDA dependencies](https://github.com/vllm-project/vllm/blob/752a3a504485790a2e8491cacbb35c137339ad34/requirements/cuda.txt).

### Preserve the SGLang 0.5.19 fallback separately

| Artifact | Verified identifier |
|---|---|
| SGLang source | `v0.5.19`, commit `0bcd822377da7b5718e674eaf9c870d349424dd1`. |
| Published container tag | `lmsysorg/sglang:v0.5.19-cu129`. |
| Container manifest digest | `sha256:59e11312666e1c5c155210ea335589b91daa0d70848521b390b93b1b1e8fb0ef`. |
| Linux amd64 image digest | `sha256:1173f6a57ac63209b1bfb054e6a7366896fd3df5539c809bbd6d332694e91d15`. |
| CUDA 12 kernel artifact in the published index | `sglang_kernel-0.4.6.post1+cu129-cp310-abi3-manylinux2014_x86_64.whl`. |
| CUDA 12 DeepGEMM artifact in the published index | `sgl_deep_gemm-0.1.7+cu129-py3-none-manylinux2014_x86_64.whl`. |

The [Docker Hub tag metadata](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.19-cu129) supplies both image digests. The [official SGLang kernel index](https://docs.sglang.ai/whl/cu129/sglang-kernel/) and [DeepGEMM index](https://docs.sglang.ai/whl/cu129/sgl-deep-gemm/) list the CUDA 12.9 artifacts. The image was retrieved and converted successfully in CPU job `13929418`. GPU job `13930204` passed BF16 matrix multiplication and FlashInfer RMSNorm on an L40S with driver 550.54.14. Model job `13930206` passed native greedy token IDs, returned text and stopping equality for both DFlash and EAGLE-3 against ordinary generation on four setup prompts. The [SGLang setup guide](sglang-setup.md) records the exact package versions, node-local extraction route and saved evidence. Broader quality, streaming, speed and resource measurements remain future work.

The v0.5.19 [DFlash documentation, lines 437–484](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/speculative_decoding.mdx#L437) explicitly pairs `meta-llama/Llama-3.1-8B-Instruct` with `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` and launches `--speculative-algorithm DFLASH`. The same source revision contains the [DFlash model](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/models/dflash.py) and [DFlash worker](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/python/sglang/srt/speculative/dflash_worker_v2.py). The released implementation therefore supplies a concrete Llama fallback to qualify.

The default v0.5.19 `pyproject.toml` still selects CUDA 13 dependencies. Its Dockerfile has a dedicated CUDA 12.9.2 branch that selects CUDA 12 kernel wheels, changes `cuda-python` to `>=12,<13`, switches FlashInfer to its `cu12` extra and selects matching PyTorch packages. Selecting `sglang==0.5.19` alone leaves those runtime choices unresolved. Preserve the exact container digest or reproduce and record the CUDA 12 build branch. [Fallback Dockerfile, lines 184–254](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docker/Dockerfile#L184), [v0.5.19 installation instructions](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/get-started/install.mdx).

Apptainer supplies the container's user-space packages while GPU execution uses the node's NVIDIA kernel driver. NVIDIA forward compatibility adds selected user-space driver libraries under documented GPU and driver conditions. A forward-compatibility path requires an explicit package choice, library-path configuration and runtime checks on this node. The container tag alone establishes artifact identity. [NVIDIA forward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html).

For each usable fallback, establish ordinary generation in that exact environment, then qualify its drafter with identical prompts and greedy settings. Keep its results labeled with the fallback source revision, container digest, dependency versions and node driver. The recorded v0.5.20 configuration remains available as a separate software comparison.


## Run the DFlash research implementation and interpret its measurements

The current official DFlash repository exposes `dflash generate` and `dflash benchmark` for Transformers, MLX and OpenAI-compatible servers. Its Linux local package dependencies are Torch 2.13.0 and Transformers 5.15.0, with Python 3.10 or later. Keep this environment separate from SGLang's package set. [Package definition](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/pyproject.toml), [README inference and benchmarks](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md#L43).

After you load the recorded DFlash environment and checkpoint paths, use this inference template:

```bash
dflash generate transformers \
  --model "<llamaTargetPath>" \
  --draft "<dflashDraftPath>" \
  --temperature 0 --max-new-tokens 256 \
  "Write a short function that reverses a list."
```

For a separately launched server, its built-in benchmark accepts `dflash benchmark openai --base-url ... --model ... --dataset gsm8k --num-prompts 128 --concurrency 1 --temperature 0`. This benchmark prints aggregate throughput based on API completion-token counts and its whole shared wall interval, plus acceptance information when the server response provides it. Its thread pool keeps at most the selected number of nonstreaming requests active and immediately replaces completed requests. The resulting workload maintains the selected concurrency while queued requests remain. An externally timed arrival trace requires another harness. [CLI](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/cli.py#L6), [server benchmark](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/benchmark.py#L357).

Current local `dflash_generate` records device-synchronized first-token time separately and starts its decode timer after the first token. The function clips `output_ids` at the maximum length and first detected stop token, retaining that stop token. Its decode-time divisor is the number of output IDs in that clipped tensor. The benchmark then decodes with `skip_special_tokens=True`, so the function's output-ID count can include a stop ID that the client never sees. Its `acceptance_lengths` list records newly produced positions per cycle after clipping. A cycle can include accepted proposals and a correction/bonus position; a stop inside the accepted prefix can exclude the correction/bonus. The first prefill token is counted separately from those cycle increments. Reconcile these counters with the study's accepted-draft and client-returned token definitions. A block-size-one run still holds the drafter object; a separate target-only process establishes ordinary memory. [Generation code, lines 204–324](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py#L204).

The current README distinguishes DFlash 2 and lists Muse-Glimmer-30B and Qwen3.8-27B releases. Its checked DFlash 2 list supplies no Llama-3.1-8B artifact. DFlash UltraChat remains the directly verified official Llama candidate. [Current supported models](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md#L25).

## Use the EAGLE research implementation

The repository contains [SETUP_EAGLE.md](../SETUP_EAGLE.md) and [SETUP_NOTEBOOK.md](../SETUP_NOTEBOOK.md). Apply the following source checks when you prepare the original EAGLE environment.

- **The Python example selects EAGLE-3 through its default.** In the checked official source, `EaModel.from_pretrained(..., use_eagle3=True)` defaults to true. Adding the explicit argument makes the intent durable; the omitted argument is currently valid. [Loader, lines 88–98](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py#L88).
- **The web UI flag has the reversal described by the teammate.** The parser's `--no-eagle3` is `store_true`, and the loader receives `use_eagle3=args.no_eagle3`. Including the flag selects EAGLE-3 at this pin. The evaluation script separately defines `--use_eagle3` as a positive opt-in flag. Record the exact entry point and flags. [Web UI lines 257–283](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/application/webui.py#L257), [evaluation parser](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_ea_answer_llama3chat.py#L408).
- **The package files conflict.** `requirements.txt` pins Torch 2.6.0 and sets Transformers ≥4.53.1; `setup.py` pins Torch 2.0.1 and Transformers 4.46.2. `SETUP_EAGLE.md` installs dependencies first and installs the repository with `--no-deps`, preserving the selected dependency versions. The guide's Transformers 4.53.1 pin gives a concrete environment to qualify. The requirements file's open upper bound otherwise permits future major changes. [Requirements](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/requirements.txt), [setup.py](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/setup.py).
- **The notebook guide changes the EAGLE environment.** `SETUP_NOTEBOOK.md` lines 26–41 install Transformers 4.57.6 into `ea_env` and acknowledge the compatibility risk. Use a dedicated pretrained-baseline environment. The prototype notebook's optimization and random initialization fall outside this inference-only study.
- **The original ordinary script keeps the drafter resident.** `gen_baseline_answer_llama3chat.py` loads `EaModel` with both model paths and calls `naivegenerate`. This provides an ordinary algorithm path inside that program. Its resident drafter affects GPU memory and capacity. A clean ordinary process should load only the target. [Baseline loader, lines 105–115](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_baseline_answer_llama3chat.py#L105).
- **The original generation code requires explicit stopping checks.** Both generation paths reduce `max_length` by the tree allocation plus ten and use `new_token > max_new_tokens` as a stopping condition. The benchmark later processes stop tokens in decoded outputs. Save token IDs and final stop reasons to validate the team's requested bound and matching behavior. [Generation methods](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/model/ea_model.py#L199).
- **The one-GPU instruction defines the intended comparison.** Official `device_map="auto"` can distribute modules across visible devices and memory tiers. Record actual placement and the assigned GPU. The setup command's `--mem=64G` requests host RAM; GPU memory comes from the assigned GPU type. A generic `--gres=gpu:1` leaves that type open.

The original evaluation saves JSONL answers containing text turns, iteration counts, generated-token counts and wall times. It performs three warmups and synchronizes CUDA around generation. Its generation interval includes prefill; request transport, admission and streaming-chunk gaps need the common client harness. [Baseline evaluation](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_baseline_answer_llama3chat.py#L130), [EAGLE evaluation](https://github.com/SafeAILab/EAGLE/blob/cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b/eagle/evaluation/gen_ea_answer_llama3chat.py).

## Run the PARD and PARD-2 research implementations

AMD's README provides these released inference commands:

```bash
# Original PARD.
python3 -m pard.infer -c config/eval/llama3_eval.yaml

# PARD-2.
python3 -m pard.infer -f pard/pard2_infer.py -c config/eval/pard2_l3_eval.yaml
```

The PARD-2 YAML sets draft length 15, 256 output tokens, HumanEval, parallel mode, compilation enabled, maximum cache length 1,024 and target `unsloth/Llama-3.1-8B-Instruct`. Its drafter is `amd/PARD2-Llama-3.1-8B`. The PARD YAML uses the canonical Meta target and includes draft lengths twelve and zero. [PARD-2 YAML](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/config/eval/pard2_l3_eval.yaml), [PARD YAML](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/config/eval/llama3_eval.yaml).

The documented NVIDIA base image is `nvcr.io/nvidia/pytorch:25.02-py3`; a ROCm image is also named. Requirements pin Transformers 4.51.3, NumPy 1.26.4, Numba 0.60.0, TRL 0.14.0 and DeepSpeed 0.16.3. The provided inference code loads static batch-one caches, casts to BF16 and compiles forward functions. The [qualified Insomnia setup](pard2-setup.md) records Python 3.10.22, Torch 2.6.0+cu124 and Transformers 4.51.3, with successful eager target/drafter loading and four adapted-output comparisons. Compiled full-model execution retains its own qualification step. [Installation and models](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/README.md#L76), [requirements](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/requirement.txt), [model loading and compilation](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/pard/pard2_infer.py#L218).

The PARD-2 paper reports vLLM inference and A100 40GB evaluation, with tensor parallelism one for its 8B targets. Its main comparison uses different proposal lengths for EAGLE-3 and the parallel methods. Those results establish a strong candidate; the public custom script supplies a different executable path whose measurements need their own label. [PARD-2 §4.1–4.2](https://arxiv.org/html/2605.08632v1#S4).

**Serving implementation:** AMD's README table includes PARD-2 target-dependent vLLM results. The older rows explicitly say `vllm-v0.16.0`; the added PARD-2 rows say `vLLM` without an exact version. The checked source still leaves the PARD-2 server version, adapter integration and exact concurrent launch path unresolved. [README lines 58–74](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/README.md#L58).

Historical vLLM documentation supplies a PARD parallel `draft_model` example with `amd/PARD-Qwen3-0.6B` and twelve speculative tokens. That documents PARD support. It provides no verified target-dependent PARD-2 adapter recipe. Current runner removal also affects this historical path. [vLLM 0.32 PARD documentation](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/features/speculative_decoding/parallel_draft_model.md).

### PARD-2 measurement boundaries

- The script's zero-proposal setting still loads both target and drafter and, for PARD-2, requests target hidden features. Keep it as an implementation control and measure clean ordinary resource usage in a fresh target-only process.
- The main timer starts after the first target prefill and ends after token decoding. Its reported output count includes committed tokens. This timing requires separate TTFT and complete-request measurements.
- `accept_length` includes the correction/bonus token and is appended before stopping checks. The loop commits a whole accepted block, then checks whether EOS occurred. The team's returned count needs explicit EOS and length clipping.
- The `acc` list records comparisons up to the first rejection; later proposed positions contribute no entries. Its mean measures the examined comparisons, with a different denominator from accepted drafts divided by all proposed drafts.
- The entry point writes result dictionaries to `datas/result/<timestamp>.jsonl`. These summaries provide useful paper-style outputs. The run also needs per-request raw token IDs, stop reasons, timestamps, resource readings and repeat IDs.

Sources: [verification and timing, lines 365–435](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/pard/pard2_infer.py#L365), [result writer, lines 65–72](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/pard/infer.py#L65).

## Qualify each configuration on Insomnia

Use the [Insomnia setup guide](insomnia.md) to establish access and select a fixed GPU type. Then qualify each model and software configuration in this order:

1. Obtain the permitted account, partition or GPU constraint, storage location and compute-node access from the current Insomnia instructions.
2. Record the actual GPU model, GPU count, VRAM, compute capability, driver, CPU/RAM allocation and container support from an allocated compute node.
3. Match that inventory to each pinned software environment. SGLang 0.5.20's CUDA 13 dependencies, vLLM 0.25.1's CUDA 12.9 wheel and PARD's supplied container are separate compatibility checks.
4. Establish approved target-model access. Download fixed model revisions into permitted storage and keep their tokenizers/configurations/adapters with them.
5. Start a small ordinary-generation loading and output check in each engine. Then load its matched drafter and compare token IDs and stopping under the same greedy settings.
6. Measure GPU peak memory at the chosen context length, output limit and concurrency. Increase the workload only after the process fits and the token and stopping checks pass.
7. Run isolated requests, then a common client replay of predetermined arrivals. Keep closed-loop source benchmarks as separately labeled checks.
8. Record each result with the source SHA, patch SHA, model revisions, resolved package versions, hardware, generation settings and exact command.

The official papers document one A100 80GB for H-Spec, A100 40GB for PARD-2's 8B evaluations and a B200 for DFlash's saved SGLang comparison. None of the checked core sources establishes a universal minimum VRAM figure across all these configurations. A V100's sm70 falls below the documented sm75 floor for the checked vLLM/FlashInfer paths; the GPU-type selection therefore matters. The [Insomnia setup guide](insomnia.md#use-a-fixed-gpu-type-for-comparisons) records the university's A6000, H100, L40 and L40S offerings and the team's saved L40S pilot. Each actual allocation still needs its driver and free-memory check.

One source inconsistency needs recording during installation: SGLang v0.5.20's installation prose explains its CUDA 13 transition by mentioning PyTorch 2.14, while its checked `python/pyproject.toml` pins Torch 2.13.0. Use the resolved packages from the exact source/wheel and retain the installation log. Both inspected sources select CUDA 13 dependencies.

The common harness must collect complete-request latency, TTFT, streamed chunk timestamps and token counts, returned output throughput, failures, and exact greedy token/stop matches. Separate profiling runs must attribute time to drafter FFNs, attention, vocabulary projection, target verification and scheduling. GPU time and peak memory need runtime collection. Slurm allocation records establish assigned GPU counts and durations. Actual GPU charges require billing evidence that states the attributable monetary amount and charged interval. Record those charges alongside the returned output tokens for the same interval; the source benchmark scripts require additional collection to supply this charge numerator and returned-output denominator. Preserve provider completion-token counts alongside the team's returned token IDs so their relationship can be checked.

## Record the remaining qualification evidence

| Check | Evidence to save |
|---|---|
| GPU and runtime compatibility | Assigned GPU model, driver, resolved package versions, selected dtype, successful imports and kernel execution. |
| Target-model access | Successful authenticated configuration retrieval and complete fixed-revision checkpoint download into approved storage. |
| H-Spec and P-EAGLE loading | Tensor-load report, target-feature configuration and greedy token/stop comparison for each model runner. |
| PARD-2 target identity | Direct-Meta snapshot provenance, tokenizer and generation configuration, plus output agreement with the target-only control. Preserve the authors' mirror-based YAML as source context. |
| PARD-2 concurrent serving | Exact released server source, adapter-loading path, launch command and successful streamed requests. |
| Output collection | Per-request original token IDs, first-token and chunk timestamps, final stop reason and failures. |
| Resources and charges | Slurm allocation record, measurement interval, every allocated GPU and any attributable monetary charge record. |

Complete these checks through model loading and inference with the released weights. Save each result with the source revision, checkpoint revision and allocated hardware.
