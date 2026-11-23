# Prepare SGLang for DFlash and EAGLE-3

This environment uses the official SGLang 0.5.19 CUDA 12.9 container. It supports ordinary Llama generation, the released DFlash UltraChat drafter and the architecture-converted EAGLE-3 drafter. Each method uses existing pretrained weights. The setup checks container identity, package imports and small GPU operations. Model loading, output correctness and performance each require their own recorded result.

## Pinned environment

| Item | Exact selection |
|---|---|
| SGLang source | `0bcd822377da7b5718e674eaf9c870d349424dd1`, release 0.5.19. |
| Official image tag | `lmsysorg/sglang:v0.5.19-cu129`. |
| Linux amd64 image digest | `sha256:1173f6a57ac63209b1bfb054e6a7366896fd3df5539c809bbd6d332694e91d15`. |
| Multi-platform manifest digest | `sha256:59e11312666e1c5c155210ea335589b91daa0d70848521b390b93b1b1e8fb0ef`. |
| Container CUDA toolkit | 12.9.2, confirmed in the image configuration. |
| Container runtime | Insomnia module `apptainer/1.4.5`. |
| Image location | `$HPML_ROOT/containers/sglang-0.5.19-cu129-amd64.sif`. |
| Setup script and retrieval helper | [setup_sglang.sh](setup_sglang.sh) and [prepare_sglang_oci.py](prepare_sglang_oci.py). |

The official registry metadata identifies the same source commit in the image labels and runtime environment. The image has 70 layers totaling 18,553,874,820 compressed bytes. The retrieval helper verifies the Linux amd64 manifest and image configuration, downloads resumable file segments with the cluster's existing `aria2c`, and checks each layer's SHA256 checksum. Apptainer converts that verified local OCI image into a SIF file. OCI describes a container's manifest, configuration and filesystem layers. SIF is Apptainer's packaged container format. The recorded SIF checksum identifies the locally converted image; the Docker digest identifies its upstream source. [Official tag metadata](https://hub.docker.com/v2/repositories/lmsysorg/sglang/tags/v0.5.19-cu129), [pinned Dockerfile](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docker/Dockerfile).

The recorded L40S node uses driver 550.54.14. CUDA 12 minor-version compatibility supplies a possible execution path for this CUDA 12.9 container. Kernel execution and model loading establish whether the required features work on the assigned node. Apptainer's `--nv` option supplies the host GPU devices and driver libraries. Job `13930204` passed the small kernel checks with that host driver and the pinned CUDA 12.9 image. The setup uses the standard host driver libraries supplied by `--nv`. Model loading has its separate qualification record below. [NVIDIA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html).

## Current qualification record

| Check | Recorded result |
|---|---|
| Official registry metadata | Source label, tag and amd64 digest verified on November 16, 2026. |
| CPU image preparation | Job `13929418` completed successfully on `ins075`. All 70 layer checksums passed. The final SIF is 17,367,351,296 bytes. |
| GPU kernels and server imports | Job `13930204` passed on `ins039`, one L40S, driver 550.54.14. BF16 matrix multiplication, FlashInfer RMSNorm, SGLang Kernel import and server help all passed. |
| Target and drafter loading | Job `13930206` completed successfully on `ins039`. Separate ordinary, DFlash and EAGLE-3 processes loaded their pinned checkpoints. The canonical target provenance records direct acquisition from Meta. |
| Greedy token and stopping equality | DFlash and SGLang EAGLE-3 each passed all four prompts: native token IDs, returned text, stop reasons and strict output caps match ordinary generation. Each comparison and the job exit code are `0`. |
| Performance measurements | Future work under the agreed study protocol. |

The local evidence includes the [CPU package inspection](results/sglang-setup-13929418/package-inspection.json), [GPU check](results/sglang-gpu-check-13930204/gpu-check.json), [runtime identity](results/sglang-gpu-check-13930204/runtime.json) and [four-prompt comparison](results/sglang-output-check-13930206/summary.json). Full native responses and server logs remain in `$HPML_ROOT/runs/sglang-qualification-13930206`. The GPU allocation completed and all three owned server groups stopped.

DFlash's released configuration emits warnings for BOS token 151643 and EOS token 151645, which exceed its declared vocabulary range. The configuration remains unchanged. The paired checks supply the canonical Llama input IDs and stop-token set explicitly; all four native outputs and stopping results match ordinary generation. Preserve this configuration warning when changing input or stopping behavior.

CPU inspection recorded Python 3.12.3, SGLang 0.5.19, PyTorch 2.13.0+cu129, Transformers 5.12.1, SGLang Kernel 0.4.6.post1+cu129, FlashInfer 0.6.18, Triton 3.7.1 and CUDA Python 12.9.7. These versions come from the pinned official image.

The initial Apptainer transfer progressed slowly through partial layers. The reusable helper now retrieves resumable segments and verifies each completed layer against the official manifest. Job `13929255` verified all 70 layers. Apptainer 1.4.5 reads the local single-image layout through `oci:/path/to/layout`; adding a tag suffix caused that release to treat the suffix as part of the directory name. The final script uses the verified directory directly. Its explicit conversion settings also keep `mksquashfs` within the CPU allocation's resources.

Shared GPFS extraction stalled during the first conversion attempt. Job `13929418` completed conversion using the allocation's private node-local `TMPDIR`. The subsequent GPU job `13929419` reached its time limit while Apptainer extracted the SIF into a temporary shared-filesystem directory. Its log records missing `squashfuse`; that attempt ended before the CUDA checks started. The current script explicitly extracts the SIF once into `$TMPDIR/hpml-sglang-runtime-0.5.19-cu129/rootfs`, records the source digest and SIF identity, and reuses that directory for each process in the same allocation. Each new allocation creates its own runtime directory.

The runtime requires an owned `/local/slurm-job-*` directory and at least 100 GiB free before extraction. The SIF, package records, model cache and reusable kernel cache persist under `HPML_ROOT`. Slurm removes its temporary allocation directory. The automated qualification job below also removes its private runtime directory on exit. Container retrieval, extraction, compilation, model loading and readiness checks belong to setup. Future inference measurements begin at the explicit measurement boundary defined by the study protocol.

## Retained files and cleanup

Cleanup job `13930240` completed successfully after qualification. It removed 107,758,820,352 bytes, approximately 100.36 GiB, of disposable assembly files from `$HPML_ROOT/cache/sglang-oci`, `$HPML_ROOT/cache/apptainer-sglang` and `$HPML_ROOT/tmp/sglang`. The superseded failed-job logs and temporary server-help output were removed after their recovery details were recorded here.

The final SIF remains under `$HPML_ROOT/containers`. Exact upstream manifests, image configuration, SIF checksum, package inspections and successful GPU checks remain under `$HPML_ROOT/runs/sglang-setup`. Full model evidence remains under `$HPML_ROOT/runs/sglang-qualification-13930206`. The checkpoint cache and reusable SGLang kernel cache remain available. The helper reuses the final SIF for subsequent allocations; reconstructing the SIF retrieves the pinned image layers again.

## Repeat the environment setup

1. In an initialized Insomnia login shell, select your private project directory.

   ```bash
   bash -l
   export HPML_ROOT="/insomnia001/depts/edu/users/$USER/hpml-baselines"
   cd "$HPML_ROOT/source/hpml-project"
   ```

2. Submit the container preparation job to a CPU allocation.

   ```bash
   sbatch --parsable --account=edu --partition=short \
     --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=40G --time=04:00:00 \
     --job-name=hpml-sglang-setup \
     --output="$HPML_ROOT/runs/sglang-setup-%j.log" \
     --wrap="bash -l $HPML_ROOT/source/hpml-project/baselines/setup_sglang.sh --prepare"
   ```

   The returned number identifies your job. The script keeps verified image layers under your scratch project and creates conversion work inside the allocation's node-local `TMPDIR`. File segments resume through the `aria2c` control files, and every completed layer must match the pinned manifest's checksum. The image conversion uses the allocated CPU count and 8G of working memory. A repeated preparation command reuses an existing image after checking its labels. An interrupted conversion retains its partial image and cache for inspection. The script refuses to replace an existing partial image automatically.

3. After preparation completes, request one L40S for compatibility checks.

   ```bash
   srun --pty --partition=short --account=edu --time=01:00:00 \
     --nodes=1 --ntasks=1 --gres=gpu:l40s:1 \
     --cpus-per-task=8 --mem=40G /bin/bash -l
   ```

4. Inside that GPU allocation, verify the container's GPU operations and server imports.

   ```bash
   export HPML_ROOT="/insomnia001/depts/edu/users/$USER/hpml-baselines"
   cd "$HPML_ROOT/source/hpml-project"
   bash baselines/setup_sglang.sh --check-gpu
   ```

   The check runs a 32-by-32 BF16 matrix multiplication and a small FlashInfer RMSNorm operation. RMSNorm scales a vector using its root-mean-square magnitude. The check also imports SGLang's kernel module and opens the server's command help. These operations check compatibility. The evidence files preserve resolved package versions, GPU identity and individual outcomes under `$HPML_ROOT/runs/sglang-setup`.

## Repeat the four-prompt output check

The reusable [qualify_sglang.py](qualify_sglang.py) starts ordinary generation, DFlash and EAGLE-3 in separate processes. The host EAGLE environment tokenizes the four setup prompts with the canonical target tokenizer and chat template. Each server receives the same token IDs, strict output cap and canonical stop-token list. The native response, producer finish reason, completion-token count and explicit output-adapter result remain separate in the saved evidence.

1. After the GPU compatibility check succeeds, submit the qualification job from the initialized login shell.

   ```bash
   sbatch --account=edu --partition=short --nodes=1 --ntasks=1 \
     --cpus-per-task=16 --mem=64G --gres=gpu:l40s:1 --time=02:00:00 \
     --job-name=hpml-sglang-models \
     --output="$HPML_ROOT/runs/sglang-setup/models-%j.log" <<'SLURM'
   #!/bin/bash -l
   set -Eeuo pipefail
   umask 077
   : "${HPML_ROOT:?Export your private project directory before submission.}"
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   export PYTHONDONTWRITEBYTECODE=1
   cleanup() {
     if [[ ${TMPDIR:-} == /local/slurm-job-* && -d "$TMPDIR" && -O "$TMPDIR" ]]; then
       rm -rf -- "$TMPDIR/hpml-sglang-runtime-0.5.19-cu129"
     fi
   }
   trap cleanup EXIT
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" \
     "$HPML_ROOT/source/hpml-project/baselines/qualify_sglang.py" \
     --output-dir "$HPML_ROOT/runs/sglang-qualification-$SLURM_JOB_ID"
   SLURM
   ```

   Copy the code block without its Markdown indentation. `sbatch` reads the script from the first `#!` line. Each invocation uses a fresh output directory. The helper verifies all three pinned artifact identities and direct-Meta target provenance before starting a server. Each server binds to `127.0.0.1:31000` on the allocated compute node and shuts down its complete process group before the next method starts. A pre-existing listener causes the check to stop and preserve the existing process.

2. After the job finishes, inspect `summary.json` and each method's comparison record in its qualification directory.

   Exit code `0` records matching native output IDs and stopping on all four setup prompts. Exit code `3` records matching outputs after the explicit stop-token and length-cap adapter, with a remaining native-output discrepancy. Exit code `2` records a token or stopping mismatch. Exit code `1` records a loading, startup, request or cleanup failure. The four setup prompts establish a bounded compatibility result. Broader quality and performance measurements follow the common baseline protocol.

## Load the pinned models for qualification

The [checkpoint downloader](download_checkpoints.py) prepares the target and public drafts in `$HPML_ROOT/cache/huggingface`. Select `target`, `dflash` and `eagle3-sglang` and verify their provenance before using these commands. The target requires your approved Meta access and direct acquisition from the canonical Meta repository. The prepared target records the canonical revision, all file hashes and direct-Meta provenance. The SGLang EAGLE-3 conversion has its own artifact identity. [Released DFlash pairing](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/speculative_decoding.mdx#L437), [released EAGLE-3 pairing](https://github.com/sgl-project/sglang/blob/0bcd822377da7b5718e674eaf9c870d349424dd1/docs/docs/advanced_features/speculative_decoding.mdx#L360).

The wrapper mounts `$HPML_ROOT` as `/hpml` inside the container, preserves the Slurm GPU selection and enables offline Hugging Face access. Local snapshot paths below identify exact revisions. The small qualification configuration uses BF16, one active request, a 2,048-token context, FlashInfer attention and a static memory fraction of 0.7. CUDA graphs, prefix caching and scheduler overlap are disabled consistently across these initial checks. The eventual benchmark configuration needs its own recorded settings and controls.

1. Inside the GPU allocation, define the pinned snapshot paths and common launch options.

   ```bash
   target=/hpml/cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
   dflash=/hpml/cache/huggingface/hub/models--z-lab--LLaMA3.1-8B-Instruct-DFlash-UltraChat/snapshots/d3af30def9601abdd10810aba220d692f0e803f0
   eagle3=/hpml/cache/huggingface/hub/models--jamesliu1--sglang-EAGLE3-Llama-3.1-Instruct-8B/snapshots/e5ed08d66f528a95ce89f5d4fd136a28f6def714
   common=(--model-path "$target" --host 127.0.0.1 --port 31000 \
     --dtype bfloat16 --context-length 2048 --mem-fraction-static 0.7 \
     --max-running-requests 1 --attention-backend flashinfer \
     --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
     --disable-radix-cache --disable-overlap-schedule)
   ```

2. Start ordinary generation in a fresh process and inspect its readiness message.

   ```bash
   bash baselines/setup_sglang.sh --exec \
     python3 -m sglang.launch_server "${common[@]}"
   ```

3. After completing the ordinary output check and stopping that process, start the DFlash process with the same common options.

   ```bash
   bash baselines/setup_sglang.sh --exec \
     python3 -m sglang.launch_server "${common[@]}" \
     --speculative-algorithm DFLASH \
     --speculative-draft-model-path "$dflash" \
     --speculative-dflash-block-size 10
   ```

   DFlash's block size ten includes the already available anchor token and nine proposed positions. The setting follows the released checkpoint configuration. Target tokenization and stopping rules govern the returned output.

4. After completing the DFlash output check and stopping that process, start EAGLE-3 with the same common options.

   ```bash
   bash baselines/setup_sglang.sh --exec \
     python3 -m sglang.launch_server "${common[@]}" \
     --speculative-algorithm EAGLE3 \
     --speculative-draft-model-path "$eagle3" \
     --speculative-num-steps 3 --speculative-eagle-topk 4 \
     --speculative-num-draft-tokens 16
   ```

   The EAGLE-3 tree settings follow the released example. A tree capacity of sixteen and a DFlash block size of ten describe different proposal structures. Record each setting with the associated method.

5. For each process, save the complete launch command, startup configuration, local checkpoint revisions, output token IDs and stop reason for the same short greedy requests.

   Qualify exact token IDs and stopping before measuring speed or resource use. Establish a fresh target-only ordinary process whenever an engine, dtype, attention backend or scheduler configuration changes. Keep any reserved evaluation prompts separate from prompts used to choose serving settings.
