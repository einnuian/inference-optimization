# PARD-2 pretrained inference setup

The separate PARD-2 environment installs successfully on Insomnia and passes its inference-module and Triton imports. CPU setup job `13929116` completes on `ins095`; repair job `13929170` completes on `ins075` with exit code `0:0` and adds the setuptools dependency used by Triton. GPU qualification job `13929173` also passes on an L40S and loads the released drafter and adapter. Job `13929467` then loads the canonical Meta target and released drafter in separate ordinary/speculative processes. All four setup prompts match after the explicit canonical-stop/output-cap adapter. The released producer appends extra tokens after the required boundary on each prompt, so native-output readiness remains incomplete. Performance measurements have their own evidence requirements.

## Source, environment and completed checks

| Item | Verified value |
|---|---|
| Public source | [AMD-AGI/PARD](https://github.com/AMD-AGI/PARD/tree/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038), revision `6f279bf3f1680e0b5d71c562ca5b91bdeef4c038`. |
| Cluster source | `/insomnia001/depts/edu/users/aa5886/hpml-baselines/source/hpml-project/external/pard`. |
| Separate environment | `/insomnia001/depts/edu/users/aa5886/hpml-baselines/envs/pard2-cu124-py310`. |
| Interpreter | Python 3.10.22. |
| PyTorch and CUDA runtime | PyTorch 2.6.0+cu124; bundled CUDA runtime 12.4. |
| Transformers | 4.51.3, matching the author's requirement. |
| Other direct inference dependencies | NumPy 1.26.4, Click 8.1.8, Loguru 0.7.3, PyYAML 6.0.2 and setuptools 80.9.0. |
| Resolved model-loading dependencies | Hugging Face Hub 0.36.2, safetensors 0.8.0 and tokenizers 0.21.4. |
| Dependency consistency | `uv pip check` reports that all 41 installed packages satisfy their declared requirements. |
| Successful source imports | `PardInfer`, `TargetFeatEmbedWarp`, `pard.infer`, Transformers `StaticCache` and Triton 3.2.0. |
| Installation record | `hpml-setup.json` and `hpml-installed.txt` inside the PARD-2 environment. |

The [setup script](setup_pard2.sh) creates the separate environment, verifies the pinned checkout and imports the inference entry points in offline mode. The source imports require PyTorch, Transformers, NumPy, Click, Loguru, PyYAML and Transformers' model-loading dependencies. The author's [full requirement file](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/requirement.txt) also contains TRL, DeepSpeed, W&B and Numba. The prepared environment follows the inspected inference import path. It executes no training command.

The author [installation instructions](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/README.md#L76) name NVIDIA's `nvcr.io/nvidia/pytorch:25.02-py3` container. The prepared CUDA 12.4 environment is a separate compatibility choice for the observed Insomnia driver 550.54.14. The completed import, small-kernel and drafter-load checks establish the dependency and released-drafter stages. The later eager model check establishes adapted output agreement on the four recorded prompts. Compiled full-model behavior, native boundary enforcement and broader token correctness need their own checks under this chosen environment. Preserve its package record when reporting later results.

## GPU qualification

GPU check job `13929173` requests one L40S and executes the setup script's `--check-gpu` mode. The check imports the inference code, verifies a 32-by-32 BF16 matrix multiplication in eager and compiled execution, and constructs/resets a small `StaticCache`. BF16 is the `bfloat16` numerical representation. The check initializes no language model and performs no text-generation benchmark.

The first GPU check, job `13929137`, identifies a missing `setuptools` dependency in Triton's import path. Repair job `13929170` installs the pinned dependency. Job `13929173` then completes successfully on `ins056` with exit code `0:0` in 3 minutes 27 seconds, including the subsequent drafter-load check. Eager BF16 computation, compiled BF16 computation and `StaticCache` creation/reset pass. The environment contains `hpml-gpu-check.json` with these results. Full model correctness and combined target/drafter memory capacity remain separate from this small compatibility check.

## Released drafter and feature adapter

The selected pretrained drafter is [amd/PARD2-Llama-3.1-8B](https://huggingface.co/amd/PARD2-Llama-3.1-8B/tree/a509e315bfe79a0be8b832004bb3ee14acfdf86a), revision `a509e315bfe79a0be8b832004bb3ee14acfdf86a`. Its required files include `model.safetensors`, `config.json` and `warp_model.bin`. The adapter file contains the learned projection that maps the selected target features into the drafter's hidden representation. The released configuration specifies 16 draft layers, hidden width 2,048, FFN intermediate width 8,192 and concatenated target-feature width 16,384 from target layers `[-1,-8,-16,-24]`.

The [author loader](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/pard/pard2_infer.py#L171) loads the drafter backbone and copies the saved adapter into `TargetFeatEmbedWarp.target_proj`. A ready checkpoint therefore contains both files. The [checkpoint downloader](download_checkpoints.py) verifies the adapter's presence and records the exact snapshot under `HF_HOME/hpml-provenance/pard2/`.

The same GPU job loads the actual released drafter in BF16. All 147 saved backbone tensor names match the loaded model. The loaded 2,048-by-16,384 adapter equals the saved projection after BF16 conversion. The environment contains `hpml-draft-load.json` with the revision, shapes, feature layers and job identity. The check creates a 32-position draft cache and generates no tokens.

Use the existing snapshot directory as the model path during qualification. The loader's fallback `hf_hub_download` call otherwise requests the adapter by repository name without an explicit revision. A fixed local snapshot preserves the paired backbone, configuration and adapter revision.

## Canonical target and the paper's Unsloth copy

The [PARD-2 evaluation YAML](https://github.com/AMD-AGI/PARD/blob/6f279bf3f1680e0b5d71c562ca5b91bdeef4c038/config/eval/pard2_l3_eval.yaml) names `unsloth/Llama-3.1-8B-Instruct`. The team's canonical target is `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`.

The November 16 [canonical model API](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct/revision/0e9e39f249a16976918f6564b8830bc894c89659?blobs=true) and [Unsloth model API](https://huggingface.co/api/models/unsloth/Llama-3.1-8B-Instruct/revision/4699cc75b550f9c6f3173fb80f4703b62d946aa5?blobs=true) list the same four safetensors filenames, byte sizes and LFS SHA-256 values. The inspected Unsloth revision is `4699cc75b550f9c6f3173fb80f4703b62d946aa5`, with public ungated access.

| Safetensors file | Bytes in both repositories | SHA-256 in both API records |
|---|---:|---|
| `model-00001-of-00004.safetensors` | 4,976,698,672 | `2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668` |
| `model-00002-of-00004.safetensors` | 4,999,802,720 | `09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15` |
| `model-00003-of-00004.safetensors` | 4,915,916,176 | `fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa` |
| `model-00004-of-00004.safetensors` | 1,168,138,808 | `92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b` |

Both repositories also list the same weight-index Git blob, `0fd8120f1c6acddc268ebc2583058efaf699a771`. The listed configuration and tokenizer files differ:

| File | Metadata comparison |
|---|---|
| `config.json` | Different Git blob and size. |
| `generation_config.json` | Different Git blob and size. |
| `tokenizer.json` | Different file identities and sizes. |
| `tokenizer_config.json` | Different Git blob and size. |
| `special_tokens_map.json` | Different Git blob and size. |
| `LICENSE` and `USE_POLICY.md` | Present in the canonical snapshot; absent from the inspected Unsloth snapshot. |

These metadata checks document the weight identities advertised by the authors' target choice. The team acquires every target file directly from Meta's canonical Hugging Face repository and verifies the canonical snapshot. The canonical configuration, tokenizer, generation settings, index and license files remain part of that snapshot. Actual target output still depends on the selected precision, attention implementation, chat template and stopping settings, so the ordinary/speculative token comparison remains required.

## Reuse the prepared environment

1. In an initialized cluster shell, set the private project root.

   ```bash
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   ```

2. Inside a compute allocation, activate the separate PARD-2 environment and select the source directory.

   ```bash
   source "$HPML_ROOT/envs/pard2-cu124-py310/bin/activate"
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   cd "$HPML_ROOT/source/hpml-project/external/pard"
   ```

3. Before loading PARD-2, use the fixed local target and drafter snapshot paths and preserve the explicit Llama model-family setting.

   The automatic family detector searches the target path string for names such as `Llama-3`. A custom canonical snapshot path can obscure that name. Passing `model_serie="llama3"` to `PardInfer` makes the intended path explicit. The initial loading check should use `nc=True` to isolate eager loading and adapter correctness; compilation receives its own qualification afterward.

4. After loading succeeds, compare a small set of greedy output token IDs and stop behavior with clean target-only generation under identical settings.

   The source evaluation command executes warmups and a dataset benchmark. The current setup stage stops before that command. The later benchmark plan uses the [common metrics](metrics.md) and records the chosen implementation.

For a fresh separate PARD-2 environment, run `bash baselines/setup_pard2.sh` inside a CPU compute allocation. `--resume` verifies any installed direct package pins before continuing a partial installation. `--check-gpu` requires a GPU allocation and performs the small compatibility check described above. `--check-draft` loads the pinned pretrained drafter and its adapter, compares the backbone tensor names with the saved checkpoint, and checks the loaded projection against `warp_model.bin` after BF16 conversion. The drafter check generates no output tokens. The script preserves the EAGLE environment and the original PARD source.

## Four-prompt target qualification

The [PARD-2 qualification wrapper](qualify_pard2.py) uses the same four setup prompts as the EAGLE check: a short `READY` reply, a 16-token integer-list cap, brief arithmetic and a small Python function. The wrapper uses the verified canonical Meta snapshot from final import job `13929459`. Target-only and PARD-2 execution use separate processes, BF16, eager attention, a 512-position cache and the canonical tokenizer's user-only chat template.

The released PARD-2 `generate` method returns aggregate statistics and passes its full output IDs to `tokenizer.decode`. The wrapper captures those IDs at that call. It preserves the author's verification and cache code, discards the internal timing summaries, and applies the shared returned-output policy afterward. The saved records contain raw producer IDs, strict capped/stopped IDs, returned IDs, stop reasons and the first divergence. The wrapper supplies the agreed user-only input; the author's default helper inserts a system message.

For another GPU/software configuration, run the prepared check inside a single-L40S allocation with sufficient host memory:

```bash
export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
export HF_HOME="$HPML_ROOT/cache/huggingface"
"$HPML_ROOT/envs/pard2-cu124-py310/bin/python" \
  "$HPML_ROOT/source/hpml-project/baselines/qualify_pard2.py" \
  --output-dir "$HPML_ROOT/runs/pard2-output-check-$SLURM_JOB_ID"
```

The check writes each stage and its comparison into a fresh evidence directory. The comparison reports normalized returned-token identity, normalized stopping equality and raw producer boundary compliance separately. `adapted_output_ready` requires the explicit canonical-stop/output-cap adapter to match the complete ordinary reference. `native_output_ready` additionally requires the released producer to end at the exact boundary with matching raw IDs. Exit code 0 indicates native readiness on these prompts; code 3 indicates adapted readiness with a native-boundary failure; code 2 indicates failed output qualification. Runtime errors use their error exit status. Raw IDs and extra-token counts remain available in every completed comparison. Four passing prompts establish those observed setup behaviors; the baseline experiment still requires its broader correctness protocol.

Run `python baselines/qualify_pard2.py --check-policy` to verify the shared boundary rules using small synthetic token lists. These checks cover all three canonical stop IDs, stops exactly at the cap, extra tokens after a stop/cap, premature termination and separate native/adapted outcomes. The checks load no model.

## Observed eager output qualification

Job `13929467` executes on `ins039` with one L40S, six CPUs and 32G of host memory. The ordinary and PARD-2 stages complete in separate processes. The job exits with code `3`, the wrapper's explicit partial-qualification status. Slurm labels that nonzero code `FAILED`; the [saved comparison](results/pard2-output-check-13929467/comparison.json) identifies the completed output checks and the native boundary failures.

| Setup prompt | Ordinary raw tokens | PARD-2 raw tokens | Returned tokens in both paths | Extra PARD-2 tokens after required boundary | Adapter output and stop match |
|---|---:|---:|---:|---:|---|
| `short_stop` | 2 | 3 | 1 | 1 | Pass. |
| `length_cap` | 16 | 29 | 16 | 13 | Pass. |
| `arithmetic` | 10 | 11 | 9 | 1 | Pass. |
| `code` | 32 | 35 | 32 | 3 | Pass. |

The [ordinary stage record](results/pard2-output-check-13929467/target.json) and [PARD-2 stage record](results/pard2-output-check-13929467/pard2.json) preserve every prompt, input ID, raw output ID, normalized output ID, returned ID and stop reason. Both stop-terminated prompts end their normalized sequence at token `128009`. Their raw PARD-2 sequences append token `128006` afterward. The length-limited prompts append 13 and three tokens beyond their caps. These observations follow the released implementation's block commitment and subsequent stopping check. The upstream algorithm and checkpoint weights remain unchanged.

The explicit adapter produces identical generated-token sequences through the canonical boundary, identical returned sequences and matching normalized stop reasons on all four prompts. Native producer boundaries fail on all four. Future serving must apply the named boundary policy before exposing tokens and include every executed draft/verification operation in timing. Broader prompts and streaming integration still need qualification. The four observations establish setup behavior and provide no performance ranking.

The cluster retains the complete run at `HPML_ROOT/runs/pard2-output-check-13929467` and its job log at `HPML_ROOT/runs/pard2-output-13929467.log`. The linked local JSON copies are preserved research data. The [tracking guide](tracking.md) identifies the saved correctness result and its verified private W&B record.

## Qualification evidence and remaining work

| Qualification stage | Status | Evidence or next check |
|---|---|---|
| Canonical target transfer | Complete. | Final import job `13929459` verifies all four direct Meta weight hashes and nine canonical small-file hashes. |
| Drafter and adapter load | Complete. | Job `13929173` matches all 147 backbone tensor names; the loaded adapter equals `warp_model.bin` after BF16 conversion. |
| Target feature extraction | Complete for the four setup prompts. | Eager generation in job `13929467` executes the configured four-layer feature extraction and 16,384-input adapter successfully. |
| Eager greedy correctness | Four adapted outputs pass; native boundaries fail. | All four normalized generated/returned sequences and stop reasons match in job `13929467`. Extend the check to the development prompts. |
| Compiled model correctness | Pending. | Repeat the exact output checks with the selected compilation mode and cache capacity. The small compiled matrix check establishes basic compiler execution. |
| Streaming stopping policy | Pending. | Apply the canonical stop-token set and strict cap before exposing each token. Count all executed work and test final-block clipping during streaming. |
| Concurrent serving | Pending. | Identify a concrete PARD-2 server, adapter-loading path and version before including it in concurrent streaming measurements. The checked custom script uses static batch-one inference. |

The completed environment, loading and eager output checks establish the preparation stages described above. The common baseline comparison requires qualified output-boundary enforcement and checks for the selected execution and serving settings. A compiled full-model configuration and a concurrent serving path each require their own qualification. Performance measurements follow the common baseline protocol, using the released pretrained weights.
