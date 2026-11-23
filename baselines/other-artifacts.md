# Additional paper artifacts for pretrained baselines

Checked on 16 November 2026. This review covers 13 saved PDFs representing 12 works. The two serving-adoption PDFs are versions of arXiv:2608.03036. The entries connect each paper’s artifact references and appendices to its public author, repository, model-hub or conference sources. Each entry identifies the available pretrained material, supported target, inference software and remaining qualification checks. The study uses existing pretrained weights throughout. Runtime correctness and performance measurements remain future work.

## Artifact availability and project role

| Work | Available material | Project use and next gate |
|---|---|---|
| Better & Faster Large Language Models via Multi-token Prediction | Official four-model 7B code release and model implementation | Separate native-MTP comparison after access and self-speculative decoder qualification; target differs from proposed Llama-3.1-8B-Instruct. |
| L-MTP | Official code and released stage-one checkpoints; Llama release explicitly targets Llama-3.1-8B base | Separate base-model candidate. Reconcile checkpoint/source metadata, missing imports/tree choices, and output handling before benchmarking. |
| Decoding Speculative Decoding | Official code and four Wide-Sheared-LLaMA checkpoints | Useful pretrained shallow-draft comparison with older Llama targets; official runner requires correctness repair and qualification. |
| Cascade Speculative Drafting | Official implementation, pretrained model paths and bigram artifacts | Optional inference algorithm comparator. Qualify the real inference path, strict verifier and compatible target family. |
| NextLat | Official code; author offers some TinyStories weights by request | Background and potential separate native-MTP case. Public FineWeb checkpoint availability remains unresolved. |
| ParallelSpec | Paper and author publication page | Paper reference pending an official implementation and compatible released parallel draft. |
| SD² | Paper, author-lab page and conference materials | Paper reference pending released sparse draft weights and runnable sparse-engine path. |
| CAS-Spec | Paper and proceedings page; paper states code submitted | Paper reference pending publicly accessible implementation. Selected DSIA components need their exact existing weights. |
| Chen speculative sampling | Original algorithm and correctness proof | Verifier reference. The original Chinchilla target/draft pair has no verified public artifact in this audit. |
| Speculative-decoding survey | Survey source | Taxonomy and bibliography. |
| Serving-evaluation paper | Methodology/checklist; related author-released Etalon harness | Metrics and arrival replay reference. Etalon is a separately named companion work. |
| Serving adoption, v1/v2 | Empirical study and cited replication URL | Serving-system context. Cited repository returned 404 during this check. |

An available artifact becomes a benchmark candidate after you qualify its target checkpoint, tokenizer, precision and inference implementation. Establish ordinary generation in the same engine, then verify identical greedy output token IDs and stopping behavior. Measure live execution with identical prompts and arrival traces. Each target family supports its own comparison. A comparison across engines measures the complete deployments; attribution to one algorithm requires additional matched controls.

## 1. Better & Faster Large Language Models via Multi-token Prediction

Saved source: 29 pages, arXiv:2404.19737v1, 30 April 2024. [Paper](https://arxiv.org/abs/2404.19737).

**Official artifact:** [facebook/multi-token-prediction](https://huggingface.co/facebook/multi-token-prediction). The [model card](https://huggingface.co/facebook/multi-token-prediction/blob/main/README.md) lists `7B_200B_1`, `7B_200B_4`, `7B_1T_1`, and `7B_1T_4`: four 7B code models with one or four prediction heads. It provides the Llama-2 SentencePiece tokenizer, a Llama-format state dictionary, `llama/model.py`, and `example_completion.py`. The documented dependencies are `torch`, `fairscale`, `fire`, and `sentencepiece`; package versions remain to be pinned. The model forward function exposes all prediction heads through `return_all_heads`. Access requires the hub’s gated process and research license.

**Use:** The existing four-head weights support a native-MTP study once the self-speculative generation path is qualified. Measure ordinary generation using the same four-head checkpoint’s first head. The separately released one-head models answer a different model comparison. The target is this code model; compatibility with Meta-Llama-3.1-8B-Instruct does not follow from Llama-format storage.

**Blocker:** The public card verifies model artifacts and forward support. The gated generation source returned 401 during the review. A complete self-speculative driver and streaming server remain unverified. After you obtain authorized model access, confirm the generation path.

## 2. L-MTP: Leap Multi-Token Prediction Beyond Adjacent Context for Large Language Models

Saved source: 24 pages, arXiv:2505.17505v2, 22 September 2025. [Paper](https://arxiv.org/abs/2505.17505).

**Official artifacts and pinned identities:** [Xiaohao-Liu/L-MTP](https://github.com/Xiaohao-Liu/L-MTP), inspected source commit `ffe8cf5a715cd7cc25c8459fe2b418034282f1d1`; [xiao-hao/L-MTP](https://huggingface.co/xiao-hao/L-MTP), inspected hub revision `323488eba01a410e8d0528c7cbd87e698a7ad255`. The README announces the S1 checkpoint release on 13 October 2025. The hub contains Gemma 4B/12B, Llama-3.1-8B, Llama-3.2-3B and Qwen2.5 3B/7B directories.

**Exact Llama target:** Both the [S1 source configuration](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/configs/s1/llama31_8B_sft_mttp_stage1.yaml) and the [released checkpoint configuration](https://huggingface.co/xiao-hao/L-MTP/blob/323488eba01a410e8d0528c7cbd87e698a7ad255/llama3.1-8b/config.json) identify **`meta-llama/Llama-3.1-8B`**, the base model. The study’s proposed `Meta-Llama-3.1-8B-Instruct` target has a distinct checkpoint identity. This L-MTP release therefore belongs to a separate base-model comparison unless an compatible Instruct release with an exact target identity is established. The source S1 construction freezes the backbone and language-model head. The audit has checked configuration and index metadata; tensor equality with the upstream backbone remains untested.

**Released checkpoint contents:** The [public folder metadata](https://huggingface.co/api/models/xiao-hao/L-MTP/tree/323488eba01a410e8d0528c7cbd87e698a7ad255/llama3.1-8b) lists four safetensors shards, a shard index, config, tokenizer files and saved preparation statistics. The [index](https://huggingface.co/xiao-hao/L-MTP/blob/323488eba01a410e8d0528c7cbd87e698a7ad255/llama3.1-8b/model.safetensors.index.json) contains 300 tensor names and 19,313,229,824 tensor bytes. It includes the backbone, LM head and **three** extra head groups (`heads.heads.0` through `.2`), each with a residual projection and vocabulary projection. The configuration records BF16, vocabulary size 128256 and Transformers 4.51.2. The checkpoint inspection uses file metadata and the tensor index.

**Source/checkpoint reconciliation required:** The hub configuration records `architectures=[MTTPModel]`, `head_type=mttp`, `n_head=5` and `skip_token=2`. Current source defines `LMTPModel`, the S1 YAML selects three heads, and [the model constructor](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/models/lmtp/model.py#L128) sets `skip_token=3`. The public [inference helper](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/models/__init__.py#L42) accepts an explicit `num_head`, whose default is five. A correct loader must reconcile the released three-head tensors, leap offset and configuration semantics without introducing new weights. The helper’s module also imports `models.medusa.model` and `models.medusa.model_official`; both referenced files are absent from this source checkout. `get_medusa_choice()` reads `self.choices`; a repository-wide source search finds no assignment or released choices JSON. The evaluation script `eval/get_all.py` can write `heads_accuracy.json`; the checked release provides neither that generated file nor code assigning it to `self.choices`. Tree construction therefore needs an explicit supplied artifact or validated deterministic setup.

**Inference software and preparation:** L-MTP uses a custom PyTorch/Transformers cache-and-tree decoder with PEFT and safetensors. The bundled [requirements](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/LLaMA-Factory/requirements.txt) specify Transformers 4.41.2–4.51.2 with exclusions, PEFT 0.14.0–0.15.1, Accelerate 0.34.0–1.6.0, datasets 2.16.0–3.5.0, tokenizers 0.19.0–0.21.0 and NumPy below 2. The [package setup](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/LLaMA-Factory/setup.py) requires Python ≥3.9 and declares an optional PyTorch dependency ≥1.13.1. These broad package ranges provide dependency constraints; an exact compatible PyTorch/CUDA environment still needs a pinned installation record. A preparation command derived from the package metadata is `python -m pip install -e ./LLaMA-Factory` from the source root, with the selected PyTorch/CUDA distribution recorded separately. The command remains unexecuted. The README provides no complete inference-only launch command. Optional vLLM/SGLang extras in the bundled package establish generic LlamaFactory support; L-MTP serving integration remains unverified.

**License evidence:** The project README and hub metadata declare MIT. The source checkout’s explicit `LICENSE` file belongs to bundled LLaMA-Factory and contains Apache-2.0. Record the upstream Llama model’s license and access terms alongside these package declarations. Keep the upstream model terms attached to the checkpoint record.

**Greedy verification and ordinary control:** [The posterior routine](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/models/utils.py#L79) contains a greedy argmax match and cumulative-prefix acceptance. The source also provides [Vanilla ordinary generation](https://github.com/Xiaohao-Liu/L-MTP/blob/ffe8cf5a715cd7cc25c8459fe2b418034282f1d1/models/vanilla/model.py). The inspected repository contains no paired ordinary-versus-L-MTP token-ID and stopping assertion harness. The streaming methods expose decoded text and a count, so a test must capture the internal IDs. Both paths use `new_token > max_new_tokens`; the speculative path yields a whole accepted block before checking stopping. `Vanilla.generate()` appends the generated suffix again in its returned `sequences` (line182), duplicating output IDs. These static findings require correction and exact-output tests before timing. `accepted_tokens = accept_length + 1` includes the extra committed token. Ordinary generation must use the same qualified base-model weights, prompts, precision and token settings.

**Project role:** Released pretrained MTP candidate for a separate base-model experiment, with substantial loader/runner qualification outstanding. A complete live benchmark and concurrency adapter remain future work. The inspection covers source code and small metadata files.

## 3. ParallelSpec: Parallel Drafter for Efficient Speculative Decoding

Saved source: 16 pages, arXiv:2410.05589v1, 8 October 2024. [Paper](https://arxiv.org/abs/2410.05589).

**Artifact status:** The saved paper supplies no author repository or checkpoint URL. The [first author’s publication archive](https://zilin.me/archive/) lists ParallelSpec with its paper link. The inspected primary sources and title/author repository searches did not identify an official implementation or released parallel-draft weights.

**Compatibility/software:** The paper’s principal comparisons use Vicuna-v1.3 7B/13B and Llama-2-Chat 7B/13B with SpecBench on A100 PCIe 40GB. Appendix A.2 and §5 describe the authors’ parallel-drafter integration with Medusa/EAGLE. The paper’s links to `double7/vicuna-68m` and `JackFram/llama-68m` supply ordinary speculative-sampling baselines. They do not establish availability of the trained ParallelSpec modules.

**Use and blocker:** Keep the paper as evidence about parallel drafting and the latency/accepted-length tradeoff. A pretrained baseline needs the authors’ parallel-draft checkpoint, its exact target and implementation. A new implementation without those weights cannot reproduce the released-method comparison within the current scope.

## 4. SD²: Self-Distilled Sparse Drafters

Saved source: 24 pages, arXiv:2504.08838v2, 31 May 2025. [Paper](https://arxiv.org/abs/2504.08838).

**Artifact status:** The saved paper, [author-lab listing](https://www.calgaryml.com/publications/) and [official conference slides](https://icml.cc/media/icml-2025/Slides/51779.pdf) describe the experiments. This audit found no author-linked public SD² checkpoint repository or complete inference artifact. The OpenReview forum required browser verification; its public paper attachment was available. Availability remains unresolved.

**Compatibility/software:** Saved §3 and Appendix C pair Llama-3.2 1B/3B drafts with Llama-3.1-70B-Instruct and Qwen-2.5 0.5B/1.5B/3B-Instruct drafts with Qwen-2.5-72B-Instruct; cross-family experiments use Universal Assisted Generation. Appendix C.5 specifies **nm-vLLM 0.5.0**, 4×NVIDIA A6000, and single requests of 300 input/256 output tokens for latency measurements. The paper combines acceptance and separately measured model timings in its improvement estimate. Ampere-oriented sparse kernels and exact sparse weight format are relevant prerequisites.

**Use and blocker:** The source motivates measuring real draft latency alongside acceptance and physically supported sparsity. Existing dense model weights provide dense comparators. SD²’s sparse checkpoint identity and runtime serialization are required for an SD² baseline. A generic sparsity-engine repository establishes software context, while the paper-specific weights remain a separate dependency.

## 5. Decoding Speculative Decoding

Saved source: 14 pages, NAACL 2025, paper 328. [ACL paper](https://aclanthology.org/2025.naacl-long.328/).

**Official artifacts:** [uw-mad-dash/decoding-speculative-decoding](https://github.com/uw-mad-dash/decoding-speculative-decoding). Released drafts: [Wide-Sheared-LLaMA-1.3B](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-1.3B), [796M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-796M), [543M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-543M), and [290M](https://huggingface.co/minghaoyan/Wide-Sheared-LLaMA-290M). The README identifies 796M as its default. These shallow, wide drafts derive from LLaMA-7B/Sheared-LLaMA.

**Inference software:** The README specifies Python 3.9, CUDA 11.8, PyTorch 2.0.1+cu118, Hugging Face Transformers/datasets and DeepSpeed. Transformers and DeepSpeed version pins remain unresolved. `spec_decoding_demo.py` uses precomputed target outputs. `spec_decoding_deployment.py` attempts live target/draft inference; its example target is Llama-2-70b-hf with four-way target tensor parallelism. The README requires tokenizer/padding and checkpoint-format changes.

**Use and blocker:** The published weights make an older-Llama shallow-draft experiment feasible in principle. Llama-3.1 uses a different tokenizer/vocabulary and needs an explicit compatibility mechanism. Static inspection of the [deployment script](https://raw.githubusercontent.com/uw-mad-dash/decoding-speculative-decoding/main/spec_decoding_deployment.py) finds unresolved paths and apparent defects: `past_key_values` reaches the first verifier call without a preceding assignment; `first_false_positions` is initialized to the full width and never computed from the mismatch tensor. The release therefore needs correctness repair or a validated generic speculative runner before timing. A trace-driven demo supports algorithm exploration; GPU charges require live model execution.

## 6. Cascade Speculative Drafting for Even Faster LLM Inference

Saved source: 15 pages, arXiv:2312.11462v5, 13 July 2025. [Paper](https://arxiv.org/abs/2312.11462).

**Official artifact:** [lfsszd/CS-Drafting](https://github.com/lfsszd/CS-Drafting). The repository contains `csd.py`, KV-cache model wrappers, a MaG bigram component, stored bigram models, datasets and an inference example. It documents Python 3.9. Its [requirements](https://github.com/lfsszd/CS-Drafting/blob/master/requirements.txt) pin PyTorch 2.0.1, Transformers 4.34.1 and datasets 2.4.0; FastChat is listed without a version.

**Compatibility/software:** Documented pairs include `google/flan-t5-xxl` with FLAN-T5-base/small, and a local Llama-7B/chat target with `JackFram/llama-160m` and a Llama bigram model. The real inference example uses `CountedCSDraftingDecoderModelKVCache`. The experiment path also contains cached-target and counting modes. Its default configurations expose lenience values greater than one.

**Use and blocker:** An optional pretrained cascade comparator can use an explicitly compatible target/tokenizer and existing bigram artifact. Qualify strict greedy verification, the live target forward path, stopping and output counts before latency measurement. The Llama2-era vocabulary-specific bigram artifact needs replacement or compatibility proof for Llama-3.1. Record the strict verification setting after checking the implementation’s lenience semantics. This audit establishes source availability; it has not executed the examples.

## 7. CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs

Saved source: 23 pages, arXiv:2510.26843v1, 30 October 2025. [Paper](https://arxiv.org/abs/2510.26843); [NeurIPS proceedings](https://papers.neurips.cc/paper_files/paper/2025/hash/5e27f5d8ecd30ac6235892ef97676e78-Abstract-Conference.html).

**Artifact status:** The saved checklist, p16, says the code will be submitted under conference guidelines. The proceedings page exposes the paper and bibliography. The inspected sources and title/author searches did not expose an official public repository. Submitted supplemental material and a public release have separate availability conditions.

**Compatibility/software:** §5.1 evaluates Llama-2-7B and Vicuna-v1.3 on H100 80GB, using SpecBench. The paper combines dynamic routing, draft-length selection, layer skipping and prompt lookup. Its early-exit variant uses existing Kangaroo adapters available for 7B/13B; the precise chosen components therefore determine checkpoint requirements. Appendix C discusses QSpec/TEAL implementations; Appendix E describes configuration. An exact executable dependency environment remains unresolved.

**Use and blocker:** The dynamic horizon/controller design is relevant to an inference-only experiment. Its current status is paper reference until a runnable release is obtained and its components pass compatibility/correctness checks. The project can benchmark an existing compatible adapter if available; creating new weights falls outside the study.

## 8. Accelerating Large Language Model Decoding with Speculative Sampling

Saved source: 11 pages, arXiv:2302.01318v1, 2 February 2023. [Original DeepMind paper](https://arxiv.org/abs/2302.01318).

**Artifact status:** The saved paper provides the algorithm and proof, with Chinchilla-70B as target and an eight-layer 4B draft (supplementary Table 2, p10). It contains no public checkpoint or author-code link. The checked current paper record and title/author search did not establish a public artifact for this original model pair. Third-party implementations require separate attribution and correctness qualification.

**Use:** Use this source for the acceptance/rejection and residual-correction definition, plus the reasoning that the unchanged verifier preserves the target distribution within hardware numerics. A modern serving engine’s two-model speculative path supplies a practical baseline once its existing draft and target share the required token semantics. That baseline should carry the engine/method/checkpoint names actually used. Exact reproduction of the original Chinchilla result remains blocked by its model and implementation availability.

## 9. Next-Latent Prediction Transformers Learn Compact World Models

Saved source: 32 pages, arXiv:2511.05963v4, 15 June 2026. [Paper](https://arxiv.org/abs/2511.05963).

**Official code:** [JaydenTeoh/NextLat](https://github.com/JaydenTeoh/NextLat). The README names `eval/eval_speculative_checkpoints.py` as the post-hoc speculative evaluation path and `utils/speculative_sampling.py` as its decoder, with draft horizon controlled by `gamma`. The code evaluates saved NextLat checkpoints. The native models have jointly learned trunks and latent-prediction components; their weights belong to their own targets.

**Pretrained availability:** The inspected README links no public pretrained language-model checkpoint, and the [release page](https://github.com/JaydenTeoh/NextLat/releases) has no packaged releases. In official [issue2, Pre-trained weights](https://github.com/JaydenTeoh/NextLat/issues/2), the repository owner states on 27 July 2026 that some TinyStories weights can be shared by email. This establishes an on-request pretrained path. Public FineWeb checkpoint availability remains unverified. Author contact remains a separate action. [Issue1](https://github.com/JaydenTeoh/NextLat/issues/1) also requests artifacts; its later correction identifies CoverageQA as unrelated, so that dataset is excluded from this audit’s NextLat artifacts.

**Inference software:** The [requirements](https://raw.githubusercontent.com/JaydenTeoh/NextLat/main/requirements.txt) specify `torch>=2.6.0`, `datasets==4.6.1`, `omegaconf>=2.3.0`, Transformers, Lightning and other evaluation dependencies. The README explains use of PyTorch’s distributed/FSDP facilities. The exact inference-only subset, PyTorch/CUDA pin, checkpoint architecture and decoder defaults require qualification when compatible weights become available.

**Use and blocker:** This supplies a reference for native MTP, finite horizon and accepted-token accounting. Its accepted draft count excludes the first ordinary prediction position. Existing TinyStories weights could support their own controlled case if obtained; their target and workload differ from the proposed Llama-3.1-8B-Instruct study. A runnable main-study baseline requires a compatible pretrained target/checkpoint path, which this audit has not established.

## 10. Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding

Saved source: 13 pages, arXiv:2411.13157v2, 27 November 2024. [Survey](https://arxiv.org/abs/2411.13157).

The survey classifies earlier methods and supplies bibliography. The saved PDF and current arXiv record provide no author code/checkpoint artifact. It introduces no new pretrained draft required for this study. Use its taxonomy to explain method families and follow each original paper for implementations. The independent verifier correctness reference remains the original algorithm paper. There is no baseline execution gate attached to the survey itself.

## 11. On Evaluating Performance of LLM Inference Serving Systems

Saved source: 17 pages, arXiv:2507.09019v1, 11 July 2025. [Paper](https://arxiv.org/abs/2507.09019).

**Artifact status and related software:** The paper provides an evaluation checklist and a speculative-decoding case study. Its saved source and current [author-lab publication page](https://sail.cc.gatech.edu/research/) do not link a distinct case-study reproduction repository. The same author group links the separately named **Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems** to [project-etalon/etalon](https://github.com/project-etalon/etalon). The older project-metron URL redirects there.

**Use:** Etalon is a candidate client-side measurement harness. Its README specifies Python 3.10, OpenAI-compatible endpoints or a custom client, trace-based request lengths, configurable Poisson request arrivals, timing deadlines, a prefill profiler and capacity search. It requires a separately served pretrained model; it is a measurement tool. Pin its revision and inspect whether streamed chunks, returned token IDs, failures, raw event timestamps and exact arrival replay satisfy this project’s protocol. Add allocation/actual-charge evidence separately. The saved serving-evaluation source motivates full latency distributions, burst-aware streaming gaps and independent TTFT reporting.

**Blocker:** The checklist can be applied immediately. The precise speculative case-study code was not identified. The harness requires adapter/event-semantic qualification before becoming the common measurement implementation.

## 12. LLM Serving in the Wild and Understanding Efficient LLM Serving

Saved sources: two 55-page versions of the same work, arXiv:2608.03036v1 (4 August 2026) and v2 (8 November 2026). [Current paper](https://arxiv.org/abs/2608.03036v2).

**Artifact:** Both versions cite a replication package at [ForoughMajidi1995/LLM-Serving-Paper](https://github.com/ForoughMajidi1995/LLM-Serving-Paper), reference 22. The saved v2 explicitly states availability on p5 and in its methodology. The public repository URL returned **404** during this check. The author’s public GitHub profile was reachable. A 404 establishes the observed access failure; the repository’s private/deleted/renamed status remains unknown.

**Use:** This is a repository-adoption study, with datasets and analysis as the relevant artifact. It supplies context for vLLM, SGLang, TensorRT-LLM, LMDeploy and FlashInfer. It supplies no new pretrained inference method and no direct hardware benchmark winner. Check framework support against the actual serving release and model implementation. The replication-package access issue affects independent reconstruction of the adoption analysis; it does not block the study’s ordinary-generation or released-method benchmarks.

## Coverage and next qualification checks

The 12 work entries cover all 13 PDFs in this review, including both serving-adoption versions. The repository’s companion baseline records cover the other saved method papers and current candidate releases.

1. Use the verified L-MTP base target and pinned revisions; reconcile its source/checkpoint discrepancies and build an exact-output qualification path before execution.
2. Check Meta’s gated generation code after existing authorized access, and identify a complete self-speculative driver.
3. Classify Decoding Speculative Decoding as released weights plus a runner requiring repair/validation.
4. Record ParallelSpec, SD² and CAS-Spec artifact availability as unresolved within the inspected sources.
5. Preserve NextLat’s owner-confirmed on-request TinyStories path; public FineWeb weights remain unverified.
6. Keep the serving-adoption repository’s observed 404 and Etalon’s separate-paper attribution visible in the final artifact inventory.
