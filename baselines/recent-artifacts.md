# Recent paper artifacts for pretrained inference baselines

Checked on November 16, 2026. The study uses existing pretrained weights throughout. The source review covers the complete substantive text and appendices of eleven saved papers, totaling 238 PDF pages. The artifact audit checks the papers’ methods, reproducibility passages, and links against current repositories, model metadata, configuration files, and inference source. Each paper entry identifies the saved version and any current online version that receives an additional artifact check.

“Available” means that an official repository or identified author release exposes the relevant code and pretrained files. Verification consists of reading file listings, configuration files, and source code. Model downloads, dependency installation, checkpoint loading, and GPU benchmarks remain future work. Runtime qualification checks compatibility, greedy output token identity, stopping behavior, and performance. A serving engine is the software that schedules requests and executes generation. Each engine needs its own ordinary-generation control on the same target checkpoint.

## Readiness overview

| Paper | Pretrained inference readiness | Target and role |
|---|---|---|
| DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | Official DeepSpec checkpoints and an offline inference evaluator are available. H-Spec supplies a separate Llama reproduction with a pinned serving path. | Same-target candidate for Llama-3.1-8B-Instruct through the H-Spec reproduction. Official Qwen3 and Gemma releases support separate target comparisons. |
| D3D: Decoupled Diffusion Drafting for Asynchronous Speculative Decoding | Six pretrained drafters, native inference, and a vendored SGLang path are available. | Candidate for Qwen3-8B or Qwen3-32B. The released asynchronous paths allocate separate target and draft GPUs. |
| FLASHMAP: Flow Map Speculative Decoding | Three pretrained drafters, standalone inference, and serving plugins are available. | Candidate for Qwen3-4B, Qwen3-8B, or Qwen3.8-27B-FP8. |
| SpecOPD: Branch-Parallel On-Policy Distillation for Speculative Decoding | The official repository explicitly releases its overview only; weights and evaluation code remain absent from that release. | Source context for the behavior of trained drafters. |
| Hybrid Spark: Adaptive Speculative Decoding That Lets the Content Choose the Drafter | The paper identifies accessible DSpark checkpoints. A complete official Hybrid Spark implementation was not located in the inspected primary sources. | Source context for combining copying and model drafting. Its identified Llama DSpark checkpoint is a distinct possible DSpark comparator. |
| Acceptance-Aware Draft Model Training for Speculative Decoding | Author checkpoints and inference code were not located in the inspected current primary sources. | Source context for acceptance and objective design. |
| Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding | Code and saved counts are released. The official FAQ explicitly states that trained drafters are unreleased. | Source context and analysis of published counts. |
| Adaptive Multi-Token Prediction: How far should we go? | Several underlying model checkpoints exist. An official AMTP controller, provider, and exact inference implementation were not located. | Source context for horizon selection. The underlying checkpoints have their own inference semantics. |
| Multi-Token Prediction Done Right (Soft-MTP) | Eight pretrained hard/soft MTP checkpoints and a custom self-speculative evaluator are released. | Additional MTP study on the authors’ 167M–1.2B targets. Each checkpoint supplies its own ordinary-generation control. |
| Match the Distribution, Not the Compute: Post-Training Multi-Token Prediction Heads | The saved paper explicitly states that code and weights were unreleased at submission. Current inspected sources expose no subsequent release. | Source context for post-trained MTP heads and verification. |
| T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding | Official adapted checkpoints and inference code were not located in the inspected current primary sources. | Source context for architecture, latency, and acceptance tradeoffs. |

Repository searches and absent paper links establish what this audit could locate. They cannot establish that an unindexed or private artifact does not exist. The explicit release statements for SpecOPD, Target Regeneration, and the saved post-training MTP paper provide stronger evidence about those particular releases.

## 1. DSpark

Saved source: [DSpark, arXiv:2607.05147v1](https://arxiv.org/abs/2607.05147v1), 33 PDF pages. The official [DeepSpec repository](https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2) is pinned at `005e03b81cec38b7da6399833d609ee89a2587f2`.

The current DeepSpec README releases the following drafters from DSpark Table 1. Hugging Face metadata lists public, ungated configuration and safetensors files for each. The checkpoint configurations identify their respective target families and a seven-token block.

| Drafter | Target | Verified model revision |
|---|---|---|
| [deepseek-ai/dspark_qwen3_4b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_4b_block7) | Qwen/Qwen3-4B | `3457dff1417cb84927f6098a5fcb7cee85c934b7` |
| [deepseek-ai/dspark_qwen3_8b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_8b_block7) | Qwen/Qwen3-8B | `03326e5043815da1f81b109078b2889737c26017` |
| [deepseek-ai/dspark_qwen3_14b_block7](https://huggingface.co/deepseek-ai/dspark_qwen3_14b_block7) | Qwen/Qwen3-14B | `83207b416acf99f41c2184648923632fccea6dd0` |
| [deepseek-ai/dspark_gemma4_12b_block7](https://huggingface.co/deepseek-ai/dspark_gemma4_12b_block7) | google/gemma-4-12B-it | `2fa72e765eec2965fc4d86a8663ce6769eba6218` |

The released [eval.py](https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/eval.py) accepts a target and released drafter path directly. The evaluator generates responses and acceptance statistics. Source inspection of the base evaluator and DSpark evaluator did not establish a complete service timing or streaming measurement harness. The evaluator creates workers across visible GPUs; that behavior alone does not establish tensor-parallel execution of a request. The repository pins PyTorch 2.9.1, Transformers 5.10.2, Triton 3.5.1, NumPy 2.4.4, and safetensors 0.7.0. The README associates these drafters with target-regenerated non-thinking data. Thinking-mode behavior therefore needs its own qualification.

### Two distinct Llama DSpark releases

| Release and precedence | Verified revision | Interpretation |
|---|---|---|
| [weifanjiang/llama3.1-8b-it.speculators.dspark](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.dspark) | `110d99b89501e3007a893da937d9560e087243c0` | The H-Spec authors’ DSpark reproduction for their Llama comparison. This release is the directly documented checkpoint for the H-Spec serving recipe. |
| [weifanjiang/llama3.1-8b-it.speculators.dspark-ce01tv09-bs8-conf-swa](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.dspark-ce01tv09-bs8-conf-swa) | `733b3b18b3c34a2659d77ec63c9cb54994c1f0b6` | The exact Llama drafter named in Hybrid Spark Table 3, page 22. The checkpoint name, revision, and source comparison differ from the H-Spec release above. |

Both Llama repositories expose public, ungated safetensors files. Neither is a DeepSpec-authored Llama release. Preserve both identities in the catalog; report results under the exact checkpoint used. Select the first for the documented H-Spec reproduction path. Select the second when investigating Hybrid Spark’s particular DSpark control, with an explicit compatibility check for the chosen loader.

The [H-Spec repository](https://github.com/weifanjiang/H-Spec/tree/de911876905440bfd8edfa78b482050f7ac0279c) pins vLLM 0.25.1 at `752a3a504485790a2e8491cacbb35c137339ad34` and Speculators at `3251ed020cfc954717c1d075d83dca8721713f0c`, with repository patches. Its inference setup supports `SKIP_MAMBA=1`; the DSpark recipe uses `method=dspark` and model runner version 2. The original DSpark paper’s production deployment has additional scheduling behavior, so a result from this reproduction describes this complete patched implementation.

The official [DeepSeek-V4-Pro-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark) (`7c09739fd136abfb70a49ec334157f65f45b52cd`) and [DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) (`62af8fffb2f7030cac4de2f0169f5b8d1101b646`) expose full target-plus-DSpark checkpoints and minimal inference code. Their listings contain 66 and 48 weight shards respectively. The conversion and generation example does not establish equivalence to the paper’s production scheduler. These target-specific systems expand the target and hardware scope substantially.

**Readiness:** usable pretrained candidate. The Llama path needs the identified author reproduction and patched serving environment. The official Qwen/Gemma paths supply further candidates if those targets enter the study.

## 2. D3D

Saved source: [D3D, arXiv:2611.06425v1](https://arxiv.org/abs/2611.06425v1), 19 pages. Official [D3D repository](https://github.com/fudan-generative-vision/D3D/tree/cd7762436217fb51d7e7a310b310b72766bb816f), revision `cd7762436217fb51d7e7a310b310b72766bb816f`.

The repository’s six listed models all expose public, ungated safetensors files. The model zoo identifies these target pairings. The inspected D3D-1.7B-8B and D3D-31D-p8B-8B configuration files provide additional architecture details. The author loader supplies the D3D behavior.

| Drafter under fudan-generative-ai | Target | Verified revision |
|---|---|---|
| [D3D-1.7B-8B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-8B) | Qwen/Qwen3-8B | `36cd7031e7adb68d287e62e87fbde6340dbf65ca` |
| [D3D-8B-8B](https://huggingface.co/fudan-generative-ai/D3D-8B-8B) | Qwen/Qwen3-8B | `a566598258fc7d103759383e52914c7c821fa11a` |
| [D3D-31D-p8B-8B](https://huggingface.co/fudan-generative-ai/D3D-31D-p8B-8B) | Qwen/Qwen3-8B | `6b16222588ec41ce242c9c350e2bbfa8315f53e4` |
| [D3D-1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-1.7B-32B) | Qwen/Qwen3-32B | `8a8b8fe4561588cac90dc2737f366d42b509d694` |
| [D3D-8B-32B](https://huggingface.co/fudan-generative-ai/D3D-8B-32B) | Qwen/Qwen3-32B | `10a3aa13fe1878189610de9f181139c7f0344704` |
| [D3D-24D-p1.7B-32B](https://huggingface.co/fudan-generative-ai/D3D-24D-p1.7B-32B) | Qwen/Qwen3-32B | `a11696b21b4c69d3b0dcfd3106d4471f1ce65b1c` |

The native path has `inference/native/single_request/eval_async_packed.py` and `inference/native/multi_request/bench.py`. The scripts allocate the target to GPU 0 and the drafter to GPU 1. Native requirements include PyTorch 2.8.0 with CUDA 12.8 and Transformers 4.57.3; the documented FlashAttention installation uses 2.8.3. Published pruning and recovery produce some of the released checkpoints. The study can evaluate these existing weights directly.

The current [SGLang instructions](https://github.com/fudan-generative-vision/D3D/blob/cd7762436217fb51d7e7a310b310b72766bb816f/docs/sglang-backend.md) describe a vendored fork originating at `wintem0531/sglang-speedup-nvidia`, revision `db686d124840ebfdd61e787b3a7fbf543244b6f0`, based on SGLang 0.5.18. The runtime lock specifies Python 3.12, PyTorch 2.13.0 with CUDA 12.9, Transformers 5.12.1, FlashInfer 0.6.17, and an SGLang kernel build for CUDA 12.9. The pinned repository includes the vendored source and actual launcher paths.

`scripts/sglang/launch.sh` exposes `hybrid`, `draft`, `remote`, and `ar` modes. The strict remote path uses one target GPU and one draft GPU; its stated scope is tensor parallelism 1, data parallelism 1, concurrency 1, and greedy decoding. The hybrid path combines a target-side local drafter with optional remote drafting. These paths require separate measurement labels. The ordinary `ar` path provides the same-engine control. Count the two-GPU allocation when evaluating remote cost, including draft GPU idle time within the billed allocation.

**Readiness:** usable pretrained candidate for the released Qwen targets. Check the selected runtime against the available GPU architecture. The current README’s smaller-drafter serving results and the saved paper’s pruned large-drafter experiments describe distinct configurations.

## 3. FlashMap

Saved source: [FLASHMAP, arXiv:2610.24365v1](https://arxiv.org/abs/2610.24365v1), 14 pages. Official [FlashMap repository](https://github.com/al-jshen/flashmap/tree/ff1476f182dbc97873e58edeca9dd7c5ce1c618f), revision `ff1476f182dbc97873e58edeca9dd7c5ce1c618f`, package version 0.2.1.

| Released drafter | Exact target | Verified revision |
|---|---|---|
| [j-shen/FlashMap-Qwen3-4B](https://huggingface.co/j-shen/FlashMap-Qwen3-4B) | Qwen/Qwen3-4B | `4e8d6a39a84ad108c5b60cb59072eb3f968c98cd` |
| [j-shen/FlashMap-Qwen3-8B](https://huggingface.co/j-shen/FlashMap-Qwen3-8B) | Qwen/Qwen3-8B | `14da37165f4afec2fabdf8ea84e152df8599200d` |
| [j-shen/FlashMap-Qwen3.8-27B-FP8](https://huggingface.co/j-shen/FlashMap-Qwen3.8-27B-FP8) | Qwen/Qwen3.8-27B-FP8 | `8c68116360a373b5de7619bd164b9f6b79b3d403` |

All three model listings expose public, ungated weights and custom code. Configuration fields identify each verifier target. The Qwen3-8B drafter has six layers, hidden width 2,048, FFN intermediate width 8,192, a 16-token block, and a 65,536-token draft vocabulary mapped to the target vocabulary. Those details establish why an arbitrary target substitution requires implementation and vocabulary checks.

The standalone path loads the target and custom drafter, then calls `draft.spec_generate(target, ...)`. The code supports a one-step or two-step generation setting. The current [README](https://github.com/al-jshen/flashmap/blob/ff1476f182dbc97873e58edeca9dd7c5ce1c618f/README.md) and [pyproject.toml](https://github.com/al-jshen/flashmap/blob/ff1476f182dbc97873e58edeca9dd7c5ce1c618f/pyproject.toml) also provide serving plugins:

- The SGLang extra pins SGLang 0.5.20. The serving recipe selects the target model, `--speculative-algorithm DSPARK`, and the FlashMap draft path.
- The vLLM extra requires vLLM `>=0.31,<0.33`; the README uses 0.32.x. The speculative configuration identifies the draft checkpoint and 16 speculative tokens. The sampled recipe additionally selects probabilistic draft sampling.
- The Qwen3.8 hybrid target requires a supported serving plugin. The standalone cache path explicitly rejects the unsupported hybrid-cache operation. The FP8 path has its own GPU and CUDA kernel requirements; README validation on H100/H200 does not establish L40S compatibility.

**Readiness:** usable pretrained candidate for its exact Qwen targets. Record the chosen plugin and engine version. A generic Hugging Face “serve this model” widget does not supply the target-plus-drafter recipe; use the author instructions.

## 4. SpecOPD

Saved source: [SpecOPD, arXiv:2611.02846v1](https://arxiv.org/abs/2611.02846v1), 28 PDF pages including the final blank page. Official [repository README](https://github.com/QwenApp/SpecOPD/blob/7fe9f9f35b027691633ec424821df5985fe34576/README.md), revision `7fe9f9f35b027691633ec424821df5985fe34576`; [project page](https://qwenapp.github.io/SpecOPD/).

The current README release-status section says that its initial release contains the project overview and excludes training code, evaluation scripts, and model weights. The paper reports trained DFlash/DSpark variants on Qwen3-8B and DFlashV2 variants on larger Qwen targets. The paper’s SGLang evaluation settings describe its results; they do not supply a released SpecOPD checkpoint or a pinned implementation that produces those results.

**Readiness:** source context. An existing original DFlash or DSpark checkpoint remains that checkpoint’s own comparator. Labeling that model “SpecOPD” would misidentify its weights.

## 5. Hybrid Spark

Saved source: [Hybrid Spark, arXiv:2611.05611v1](https://arxiv.org/abs/2611.05611v1), 33 pages. The paper’s current arXiv record, all saved PDF links, title/author web searches, and GitHub repository searches did not reveal an official complete Hybrid Spark implementation.

Section 5.1, page 10, specifies vLLM’s V1 engine with model runner version 2, asynchronous scheduling, and prefix caching disabled. Experiments use H100 80 GB GPUs and fixed-concurrency requests. Table 3, page 22, identifies released DSpark drafters for Qwen3-4B, Qwen3-8B, Gemma-4-12B-it, North-Mini-Code-1.0-w4a16, and Llama-3.1-8B-Instruct.

The DSpark checkpoint table in section 1 lists the Qwen and Gemma drafters. Two additional public, ungated checkpoint listings were verified:

- [sdougbrown/North-Mini-Code-1.0-dspark](https://huggingface.co/sdougbrown/North-Mini-Code-1.0-dspark), revision `fefbb245a48999de60962ce528053b08b19b75f7`, pairs with the paper’s quantized North target.
- [weifanjiang/llama3.1-8b-it.speculators.dspark-ce01tv09-bs8-conf-swa](https://huggingface.co/weifanjiang/llama3.1-8b-it.speculators.dspark-ce01tv09-bs8-conf-swa), revision `733b3b18b3c34a2659d77ec63c9cb54994c1f0b6`, is Hybrid Spark’s Llama DSpark checkpoint. Preserve this identity separately from H-Spec’s `110d99b…` checkpoint.

The full Hybrid Spark system additionally requires its GPU copying selector, drafter selection, adaptive verification, and model-specific cost/survival configuration. The paper explains those components, including calibration and held-out evaluation. The checked sources did not expose an exact versioned implementation and complete ready-to-use parameters. Implementing that system would be new implementation work; fitting new learned parameters falls outside the study’s pretrained inference scope.

**Readiness:** source context for the complete system. The released component checkpoints can support explicitly labeled DSpark comparisons after loader qualification.

## 6. Acceptance-Aware Draft Model Training

Saved source: [Acceptance-Aware Draft Model Training for Speculative Decoding, arXiv:2609.24150v1](https://arxiv.org/abs/2609.24150v1), 13 pages. The [current record](https://arxiv.org/abs/2609.24150) has a November 10 v2. The full reread covers saved v1; the current v2 received an artifact-link check.

The saved paper evaluates learned acceptance-aware variants using Qwen targets and smaller draft models, with seven-token chain drafting and tree settings in its experiments. Those variants have weights produced by the authors’ objectives. Public generic Qwen base checkpoints identify architectures, while reproducing the evaluated variants requires their resulting drafter checkpoints.

The saved PDF annotations, current arXiv/HTML artifact checks, exact-title searches, and repository searches did not identify an author repository, trained model release, or pinned inference implementation. The saved experiments’ hardware and decoder settings therefore remain source context.

**Readiness:** source context. Keep v1 result citations explicit until a full v2 comparison is performed. No team training action follows from this paper in the current study.

## 7. Target Regeneration

Saved source: [Target Regeneration Shrinks the Gains of Draft-Training Objectives in Speculative Decoding, arXiv:2610.29013v1](https://arxiv.org/abs/2610.29013v1), 19 pages. Official [TargetRegeneration repository](https://github.com/js-lee-AI/TargetRegeneration/tree/d2112625288553b625165627795ddf269cfd9351), revision `d2112625288553b625165627795ddf269cfd9351`.

The README FAQ explicitly states that trained drafters are unreleased. The repository releases analysis code, prompt-level vision evaluation counts, and the pipeline. Its evaluation entry point expects a resulting trained checkpoint such as `checkpoints/full_kd_s0/...`. Its original public DFlash initialization does not supply the paper’s final objective-specific variants.

The main study uses Qwen3-VL-8B-Instruct. The text-target extension is Appendix G, Table 14, page 18, with Qwen3-4B DFlash/EAGLE3 variants. The repository’s released analysis can recompute statistics from saved counts. Those counts do not measure performance on the team’s hardware. The pipeline documents PyTorch 2.9.1 and Transformers 4.57.6; its regeneration environment separately uses vLLM 0.15.1.

**Readiness:** source context and optional analysis of saved published counts. The objective-specific drafters cannot yet enter the pretrained benchmark from this release.

## 8. Adaptive Multi-Token Prediction

Saved source: [Adaptive Multi-Token Prediction: How far should we go?, arXiv:2610.28619v1](https://arxiv.org/abs/2610.28619v1), 25 pages. Appendix D, pages 20–21, identifies base/model families and describes a custom Inspect AI provider. The inspected paper links, current arXiv record, author/title searches, and repository searches did not identify the AMTP provider, controller implementation, lockfile, or a released controller state.

Two named underlying model releases were verified:

| Underlying model | Verified revision and access | What the release establishes |
|---|---|---|
| [jwkirchenbauer/Qwen3-4B-Inst-2507-MTP](https://huggingface.co/jwkirchenbauer/Qwen3-4B-Inst-2507-MTP) | `5273e79569b30ca334fe4093d32a225b327fde0a`; public, ungated; custom code and three weight shards listed | A separate MTP model release with its own generation interface. Its model card cites a different source paper. |
| [facebook/multi-token-prediction](https://huggingface.co/facebook/multi-token-prediction) | `6580f46982ea783120bfe10796717e5c139144b3`; manual access gate | The original MTP checkpoints, including `7B_1T_4`, and custom model/generation code. Access must be granted before weight retrieval. |

The Qwen model card labels its static MTP path as potentially lossy and its confidence-adaptive path as nearly lossless. The card also says standard sampling arguments are ignored in its MTP interface and limits that interface to a single example. Those semantics require an independent verifier audit before the checkpoint can satisfy the study’s exact greedy-output control. Calling that interface does not establish AMTP reproduction.

The paper also uses a Llama target with separate Medusa heads, whose exact head repository was not identified in the saved reproducibility passage. AMTP’s online hazard and cost updates form part of its controller behavior. Fixed drafter weights alone do not establish that the entire controller is fixed. The current study can use the paper to motivate fixed horizon measurements; a new fitted or learned controller falls outside its scope.

**Readiness:** source context for AMTP. The underlying MTP releases are separate possible inference subjects, with their own target, access, and verification requirements.

## 9. Soft-MTP

Saved source: [Multi-Token Prediction Done Right, arXiv:2610.38331v1](https://arxiv.org/abs/2610.38331v1), 23 pages. Official [Soft-MTP repository](https://github.com/JaydenTeoh/Soft-MTP/tree/44c48b9693da70c47b27dad632d2ac2984ebb0eb), revision `44c48b9693da70c47b27dad632d2ac2984ebb0eb`; [released collection](https://huggingface.co/collections/JaydenTeoh/soft-multi-token-prediction).

All eight checkpoints in the following table expose public, ungated safetensors files. Each model ID is under `JaydenTeoh/`.

| Checkpoint | Verified revision |
|---|---|
| [hard-mtp-167m-model](https://huggingface.co/JaydenTeoh/hard-mtp-167m-model) | `8aeb4721eda8a2cb7beb2cadb76c225ab54f38ff` |
| [soft-mtp-167m-model](https://huggingface.co/JaydenTeoh/soft-mtp-167m-model) | `b06441f3fd5c2e005584f2857c9e7cadca168018` |
| [hard-mtp-332m-model](https://huggingface.co/JaydenTeoh/hard-mtp-332m-model) | `9047b383c1b57002bdc84a6187f523f02a508e2b` |
| [soft-mtp-332m-model](https://huggingface.co/JaydenTeoh/soft-mtp-332m-model) | `0334ed95345a89e907413fd4e6ea076c28aa030d` |
| [hard-mtp-689m-model](https://huggingface.co/JaydenTeoh/hard-mtp-689m-model) | `dc56c695291871d98c0af5b3b1133b0a82234d6f` |
| [soft-mtp-689m-model](https://huggingface.co/JaydenTeoh/soft-mtp-689m-model) | `2e14a00e181422c71756fa28bc95f28b246c8dd9` |
| [hard-mtp-1p2b-model](https://huggingface.co/JaydenTeoh/hard-mtp-1p2b-model) | `4a3f3ea30abe08ca81c37910360cbd42da7f8526` |
| [soft-mtp-1p2b-model](https://huggingface.co/JaydenTeoh/soft-mtp-1p2b-model) | `a391d451af93334f4b171546978d7aa1a3563776` |

The inspected 689M configuration identifies a custom `HFMTPForCausalLM`, GPT-2 tokenizer, three MTP heads, and `mtp_arch=deepseek`. The README states that ordinary `.generate()` ignores the MTP heads. Its speculative entry point is `python -m src.eval_spec --ckpts ...`; source inspection shows the custom loader calling `HFMTPForCausalLM.from_pretrained(c)`. This establishes the intended loading path by static inspection. Actual loading remains untested.

The [evaluator source](https://github.com/JaydenTeoh/Soft-MTP/blob/44c48b9693da70c47b27dad632d2ac2984ebb0eb/src/eval_spec.py) requires attention before common-control benchmarking: the speculative path samples at its default temperature 1, while the ordinary comparison function uses greedy sequential generation. The team’s comparison must align generation policy and check stopping/count boundaries before interpreting a speed ratio. The evaluator also measures a fixed generation length, so returned-token and EOS behavior need explicit observation.

The custom PyTorch path supplies KV-cached self-speculative generation. No official vLLM or SGLang integration was verified. Requirements use broad ranges including PyTorch `>=2.9`, Transformers `>=4.52`, and Hugging Face Hub `>=0.23,<1.0`; preserve the resolved environment when qualifying the release.

**Readiness:** pretrained inference is available for the authors’ own 167M–1.2B targets. Each target checkpoint’s next-token path supplies its ordinary control. These targets establish a distinct MTP comparison from the canonical Llama-3.1-8B-Instruct study. Hard-MTP and Soft-MTP checkpoints have different learned target weights; their relative performance does not isolate an inference-only intervention on one target.

## 10. Post-training MTP heads

Saved source: [Match the Distribution, Not the Compute: Post-Training Multi-Token Prediction Heads, arXiv:2610.00888v1](https://arxiv.org/abs/2610.00888v1), 12 pages. Appendix F, page 10, explicitly states that code and weights were unreleased with the submission. The inspected current arXiv record, exact-title searches, and repository searches did not identify a subsequent author release.

The paper attaches three or six learned heads to a frozen Qwen3-8B backbone. Its public base checkpoint alone does not contain those learned heads. The serving description uses vLLM V1 with custom verification extensions, without an identified versioned release of those extensions. MiMo-7B-RL is a separate public comparator in the paper; it does not supply the paper’s post-trained Qwen heads. The paper’s relaxed verification variants also require their own output-quality interpretation.

**Readiness:** source context. A future artifact check would need both the resulting heads and their exact verification implementation before this method could join the current pretrained study.

## 11. T2D

Saved source: [T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding, arXiv:2610.34103v2](https://arxiv.org/abs/2610.34103v2), 19 pages. The saved paper’s complete URI annotations, current arXiv record, exact-title/author searches, and repository searches did not identify an official model release or inference repository.

The method produces adapted drafters through structured pruning and further optimization. T2D Table 7 and Appendices C–G describe resulting architectures and evaluation settings for targets including Llama-2-7B-Chat, Vicuna-13B-v1.5, Qwen2.5-7B-Instruct, Qwen3-14B, and Nemotron-32B. The checked sources do not identify exact downloadable resulting checkpoints, a pinned inference environment, or a complete loader for those variants.

The public target checkpoints provide the source architectures. Applying fresh pruning to them would create a separate intervention with different weights and acceptance behavior from the reported adapted T2D drafters. The reported tradeoffs remain relevant evidence about architecture and timing, while the current study requires existing released weights.

**Readiness:** source context pending an identified release of the adapted drafters and inference path.

## Consequences for the baseline catalog

For the current Llama-3.1-8B-Instruct target, the documented H-Spec DSpark reproduction is the immediate additional released candidate in this set. Preserve Hybrid Spark’s different Llama DSpark checkpoint as a separately named entry. Its checkpoint availability does not establish availability of the full Hybrid Spark system.

Qwen3-8B provides the broadest shared target among official DSpark, D3D, and FlashMap releases in this audit. Adding that target would permit a common-target comparison among those candidates. Every selected engine receives an ordinary-generation run with the same target weights, prompts, generation policy, dtype, and GPU type. A complete-system comparison records the GPU count and allocation duration required by each deployment. D3D’s remote deployment uses separate target and draft GPUs, while its ordinary route uses one target GPU. Include both D3D GPUs in allocated GPU time and attributable charge accounting.

The available artifacts establish candidate eligibility. The team’s measurements determine the speed and cost references for each workload and arrival rate. Qualification requires exact checkpoint loading, supported kernels on the selected GPUs, matching greedy output token IDs and stopping behavior, and measurement under the team’s isolated and concurrent arrival traces. Papers with currently unlocated or explicitly unreleased variants remain attributed research context throughout the inference-only study.
