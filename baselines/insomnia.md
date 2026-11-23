# Prepare pretrained baselines on Insomnia

Checked on November 16, 2026. The local project checkout is `PROJECT/hpml-project`. The Mac holds the source, configurations and copied results. Insomnia executes CUDA inference inside a Slurm GPU allocation. Slurm is the cluster scheduler that assigns compute resources to jobs.

## Confirmed access and current connection

The September 29 welcome email names the `edu` account, a 50 GB home directory, a 150,000-file quota and `/insomnia001/depts/edu/users/aa5886` as Ayo's scratch location. The authenticated November 16 session confirms UNI `aa5886`, login node `2402-login-002`, default Slurm account `edu` and group `somedu`. Other teammates use their own UNI and approved account.

Cisco Secure Client 5.1.20.333 is installed from Columbia's official download, with the required system permissions enabled. The client confirms **Connected to vpn.columbia.edu**. University sign-in, Duo and SSH authentication are complete. The university requires VPN off campus or on an insecure campus network; secure campus connections can reach Insomnia directly. [Insomnia: Getting Started](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145140/Insomnia+Getting+Started).

The [Columbia VPN instructions](https://www.cuit.columbia.edu/install-vpn) specify `vpn.columbia.edu`, the UNI as username, the UNI password and `push` in the **Second Password / Duo Action** field. The instructions also support a fresh passcode from the Columbia University account in Duo Mobile.

1. When your network requires VPN, connect to the Columbia VPN.
2. In your terminal, connect with your own UNI.

   ```bash
   ssh aa5886@insomnia.rcs.columbia.edu
   ```

3. Complete the SSH password and Duo prompts.
4. After sign-in, start a login Bash shell.

   ```bash
   bash -l
   ```

5. In the initialized Bash shell, inspect your account and current GPU resources.

   ```bash
   id
   printf '%s\n' "$SLURM_CONF"
   sinfo -N -o '%N|%P|%G|%f|%t'
   sacctmgr -nP show assoc user="$USER" format=Account,Partition,QOS
   ```

Login initialization is required for the checked account. Its default `SHELL` is `/bin/sh`, and a noninteractive connection initially has no `SLURM_CONF`. A login Bash shell sets `SLURM_CONF=/etc/slurm/slurm.conf` and initializes the cluster environment. For remote noninteractive commands, use `bash -lc '<command>'` so the same initialization runs.

`sinfo` reports exact GPU type names and current node state. The verified `edu` association establishes the Slurm account used by the inspection job. The account name alone supplies no monetary rate.

## Use a fixed GPU type for comparisons

The saved team pilot reports **NVIDIA L40S**. The university hardware page lists A6000, H100, L40 and L40S nodes. The live scheduler additionally reports the GPU type names `A6000`, `h100`, `l40`, `l40s` and `R6000`. Preserve their capitalization in a typed GPU request. Node availability changes as jobs start and finish. [Insomnia technical information](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145136/Insomnia+-+Technical+Information).

The inspected `short` partition permits all accounts and has a 12-hour limit. Its `MaxMemPerCPU` is 6,400 MB. The following two-hour setup request chooses one L40S, one task, 16 CPU cores and 64G of host RAM. Sixteen CPU cores provide sufficient per-CPU memory allowance for that host-memory request.

1. After confirming that the requested GPU type is available to your account, request the setup allocation.

   ```bash
   HPML_GPU_TYPE='l40s'
   srun --pty --partition=short --time=02:00:00 \
     --nodes=1 --ntasks=1 --gres="gpu:${HPML_GPU_TYPE}:1" \
     --cpus-per-task=16 --mem=64G --account=edu /bin/bash -l
   ```

2. Inside the allocation, record the node, driver and visible GPU.

   ```bash
   hostname
   nvidia-smi
   module avail cuda
   ```

3. On the compute node, load the verified Apptainer module and check its version.

   ```bash
   module load apptainer/1.4.5
   apptainer --version
   ```

`--ntasks=1` requests one process. During the inspection, an implicit two-task allocation executed the inspection command twice; setting the task count makes the execution count explicit. `--mem=64G` requests host RAM. The assigned GPU supplies GPU memory. Benchmark job duration follows the measured run requirements. CPU-intensive installation, compilation and inference run on compute nodes. [Insomnia job submission](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145134/Insomnia+-+Submitting+Jobs), [GPU and Apptainer examples](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145126/Insomnia+-+Job+Examples).

### Verified driver-inspection allocation

Job `13928964` completes on `ins056`. Slurm records `cpu=2`, `gpu=1`, `mem=2G`, `node=1` and one second of elapsed allocation time. The job inspects hardware and software; it supplies no inference measurement.

| Item | Observed value |
|---|---|
| GPU | NVIDIA L40S, 46,068 MiB, compute capability 8.9. |
| NVIDIA driver | 550.54.14. |
| CUDA version displayed by `nvidia-smi` | 12.4. |
| CUDA toolkit module | `cuda/12.3`; `nvcc` 12.3.107. |
| Apptainer module | `apptainer/1.4.5`, loaded on the compute node. |
| System Python | 3.9.18. |

Record these values again on the node used for model execution. The inspection establishes the available runtime configuration on `ins056`. Subsequent jobs qualify EAGLE, PARD-2 and SGLang model loading and four-prompt output comparisons. The [setup handover](setup-report.md#baseline-readiness) records the result and remaining checks for each method.

## Create project storage and isolated environments

The scratch path named in the welcome email initially did not exist. The authenticated session creates the private project directories under `/insomnia001/depts/edu/users/aa5886/hpml-baselines` with `umask 077`: `source`, `envs`, `cache/huggingface`, `cache/uv` and `runs`.

The shared filesystem reports 20T total capacity and 7.4T available at the inspection time. Those figures describe shared filesystem capacity. The personal scratch quota requires its own quota record.

The private team repository is copied to `/insomnia001/depts/edu/users/aa5886/hpml-baselines/source/hpml-project` with the user's approval. Its source commit matches the local checkout: `c8af4638d2b7ba046765a31425be5e7fda80dc6f`. The prepared baseline documents and scripts accompany that checkout as uncommitted changes. The Mac remains the local working copy. All seven public source repositories under the cluster checkout's `external/` directory match the revisions in [core-methods.md](core-methods.md).

1. In the login Bash shell, select your approved scratch location and initialize private project storage.

   ```bash
   umask 077
   export HPML_ROOT="/insomnia001/depts/edu/users/$USER/hpml-baselines"
   mkdir -p "$HPML_ROOT"/{source,envs,cache/huggingface,cache/uv,runs}
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   export UV_CACHE_DIR="$HPML_ROOT/cache/uv"
   df -h "$HPML_ROOT"
   df -i "$HPML_ROOT"
   ```

2. After each authorized project update, verify the files in `"$HPML_ROOT/source/hpml-project"` against the Mac's source revision and accompanying uncommitted changes.
3. Inside a compute allocation, create a separate environment for each software family listed in [core-methods.md](core-methods.md).
4. Save code revisions, model revisions and installed package versions with every qualified configuration.

Scratch storage and home storage require separate copies of important results because the cluster supplies no backup. Filesystem free space and personal quota measure different limits; retain both records when available. [Insomnia storage](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145132/Insomnia+-+Storage).

The existing `SETUP_NOTEBOOK.md` installs Transformers 4.57.6 into the EAGLE environment, while `SETUP_EAGLE.md` uses 4.53.1. Keep the historical notebook environment separate. The pretrained baseline environments retain their method-specific dependencies.

## Check the driver before selecting an engine build

| Software path | Source requirement or reviewed configuration | Live check |
|---|---|---|
| EAGLE research implementation | CPU job `13929007` installs Python 3.10.22, PyTorch 2.6.0+cu124, Transformers 4.53.1 and Accelerate 0.26.0. Pinned EAGLE model-class imports pass. | GPU job `13929049` passes the CUDA/BF16 check. Model job `13929466` verifies all saved drafter tensors and matches four adapted outputs; native length-cap overshoot remains recorded. |
| SGLang for DFlash/EAGLE-3 | The selected version 0.5.19 uses the pinned official CUDA 12.9 container. Exact source and image digests appear in `core-methods.md`. | GPU job `13930204` and model job `13930206` pass on driver 550.54.14. DFlash and EAGLE-3 each match ordinary native output and stopping on four prompts. Version 0.5.20's CUDA 13 path retains separate qualification requirements. |
| H-Spec/P-EAGLE pinned vLLM route | The H-Spec recipe patches vLLM 0.25.1; the selected binary explicitly uses CUDA 12.9. P-EAGLE and H-Spec select different model runners. | Job `13930196` passes basic GPU checks. Job `13930616` completes both ordinary controls plus P-EAGLE, H-Spec and DSpark; all three methods pass four native output/stopping comparisons on driver 550.54.14. |
| PARD-2 custom route | The released PARD-2 inference script uses standard Transformers 4.51.3 in the prepared environment. | Job `13929173` verifies the released backbone and adapter. Model job `13929467` matches all four adapted outputs and records native stop/cap overshoot. |

The university software page lists CUDA 12.3 and Python 3.9 among installed software, and explicitly notes that the list can lag installed versions. A CUDA toolkit listing, the maximum CUDA version displayed by `nvidia-smi`, and the CUDA runtime bundled with PyTorch are separate quantities. Record all available values. A container supplies user-space packages; the node's NVIDIA driver still determines runtime compatibility. [Insomnia software](https://columbiauniversity.atlassian.net/wiki/spaces/rcs/pages/62145130/Insomnia+-+Software). Exact method sources and pins appear in `core-methods.md`.

From the verified project checkout, use `python baselines/preflight.py --mode login` from the repository in an initialized login Bash shell to capture scheduler and storage information. Inside a GPU allocation and the selected method environment, use `python baselines/preflight.py --mode gpu` to capture the GPU and PyTorch environment. The script prints a JSON record and performs no model download or benchmark. Save each record in that run's evidence directory.

## Prepare the EAGLE inference environment

CPU job `13929007` completes the installation in `"$HPML_ROOT/envs/eagle-cu124-py310"`. The verified environment contains Python 3.10.22, PyTorch 2.6.0+cu124, Transformers 4.53.1 and Accelerate 0.26.0. Imports of the pinned EAGLE model classes pass. GPU compatibility job `13929049` completes with exit code 0 on `ins056` after 153 seconds. CUDA is available and the 32-by-32 BF16 matrix multiplication passes. The [subsequent model check](#qualify-the-eagle-target-and-released-drafter) records target/drafter loading and generation separately.

[setup_eagle.sh](setup_eagle.sh) creates a private Python 3.10 environment with PyTorch 2.6.0/CUDA 12.4, Transformers 4.53.1 and Accelerate 0.26.0. The script installs the pinned public EAGLE source with `--no-deps` and checks its model-class imports. The environment's `hpml-installed.txt` records all resolved package versions; `hpml-setup.json` records the source revision and successful import check.

1. In an initialized login Bash shell, define the scratch root.

   ```bash
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   ```

2. For a fresh workspace, run the installer in a CPU allocation.

   ```bash
   srun --partition=short --account=edu --ntasks=1 \
     --cpus-per-task=4 --mem=16G --time=00:30:00 \
     bash -l -c 'cd "$HPML_ROOT/source/hpml-project"; bash baselines/setup_eagle.sh'
   ```

   The installer preserves existing environments. After an interrupted installation, `--resume` validates the existing interpreter, package versions and source before continuing. The installer obtains the CUDA-specific Torch wheel from PyTorch's official wheel page and resolves its NVIDIA dependencies through PyPI. The initial full CUDA-index attempt timed out at `pypi.nvidia.com`; the revised source selection preserves the chosen package versions.

3. On each new GPU configuration, run the small compatibility check in a one-GPU allocation.

   ```bash
   srun --partition=short --account=edu --ntasks=1 --gres=gpu:l40s:1 \
     --cpus-per-task=2 --mem=8G --time=00:03:00 \
     bash -l -c 'cd "$HPML_ROOT/source/hpml-project"; bash baselines/setup_eagle.sh --check-gpu'
   ```

   The check imports EAGLE, confirms CUDA availability and verifies a 32-by-32 BF16 matrix multiplication. BF16 is the `bfloat16` numerical format. This check establishes basic package and GPU operation. Target loading, exact output matching, memory capacity and performance each require their own subsequent tests.

## Obtain the model artifacts

Final import job `13929459` verifies all thirteen canonical Meta target files, including every weight SHA-256, and writes the direct-Meta provenance record. The target is ready for local offline loading.

All seven released drafter snapshots are present in the private scratch cache. Job `13929052` retrieves the original EAGLE-3 checkpoint; job `13929071` retrieves the SGLang EAGLE-3 conversion, DFlash, P-EAGLE, H-Spec, PARD-2 and DSpark snapshots. Each snapshot has its exact revision and required files checked, with a provenance record under `HF_HOME/hpml-provenance/`. PARD-2 includes the required `warp_model.bin` feature adapter.

The common target is `meta-llama/Llama-3.1-8B-Instruct` at revision `0e9e39f249a16976918f6564b8830bc894c89659`. Obtain every target file directly from that Meta repository through your own approved access. The selected target uses thirteen files: four safetensors weight shards, the shard index, model and generation configurations, three tokenizer files, the model card, license and use policy.

### Retrieve snapshots from an authenticated cluster session

1. Under your own Hugging Face account, confirm approval for the Meta target repository.
2. In your cluster shell, select the scratch cache and installed Hugging Face CLI.

   ```bash
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/hf" auth login
   ```

   Enter the credential privately in the CLI prompt. Keep tokens out of source files, command arguments, notebook outputs and chat. Another teammate uses that teammate's own approved scratch location and account.

3. Inside a compute allocation, retrieve the canonical target with the prepared downloader.

   ```bash
   cd "$HPML_ROOT/source/hpml-project"
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" \
     baselines/download_checkpoints.py --download --only target
   ```

4. In a fresh setup, retrieve the released drafters at their pinned revisions.

   ```bash
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" \
     baselines/download_checkpoints.py --download \
     --only eagle3 eagle3-sglang dflash peagle hspec pard2 dspark
   ```

The [checkpoint downloader](download_checkpoints.py) lists the eight pinned artifacts with `--list`. Public drafter selections use anonymous access. The target selection uses the authenticated Meta account. Both the downloader and browser importer verify all thirteen target file sizes, all four weight SHA-256 hashes, all nine small-file Git blob hashes, valid JSON files and exact four-shard index coverage. Each route publishes one complete source-verification record after those checks pass.

When the target cache already has a valid direct-Meta acquisition record, repeated setup reuses the files, verifies their content and preserves the original acquisition details. A target cache without that record requires a fresh download from Meta's canonical repository. Matching file content and the recorded acquisition route establish separate parts of the target provenance.

### Import direct Meta browser downloads

The November 16 setup uses the already approved Meta browser session. The Mac downloads one weight file at a time, transfers that file to the private Insomnia cache and removes the local transfer copy after cluster verification. Each weight file is approximately 1.2–5.0 GB. Sequential transfer keeps the local storage requirement close to the largest file's size.

1. In Chrome, open the [pinned Meta file list](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659).
2. Download the nine small files named by `SMALL_FILE_BLOBS` in [import_meta_target.py](import_meta_target.py), and transfer them to the private import directory.

   ```bash
   export META_IMPORT="$HF_HOME/hpml-import/llama3.1-8b/0e9e39f249a16976918f6564b8830bc894c89659"
   mkdir -p "$META_IMPORT"
   ```

   Preserve each canonical filename. A browser-added suffix such as `README (1).md` needs the canonical destination name `README.md`. The import script verifies each small file's exact byte size and Git blob hash.

3. For each weight file, wait for the browser download to finish, then transfer the complete file to that directory.

   From the Mac, the first weight file's transfer command is:

   ```bash
   scp "$HOME/Downloads/model-00001-of-00004.safetensors" \
     aa5886@insomnia.rcs.columbia.edu:/insomnia001/depts/edu/users/aa5886/hpml-baselines/cache/huggingface/hpml-import/llama3.1-8b/0e9e39f249a16976918f6564b8830bc894c89659/
   ```

4. After the transfer command completes successfully, verify and import that file inside a CPU allocation.

   ```bash
   cd "$HPML_ROOT/source/hpml-project"
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" \
     baselines/import_meta_target.py --meta-files "$META_IMPORT" \
     --only-weights model-00001-of-00004.safetensors
   ```

5. After the import confirms the exact SHA-256 and byte size, remove the completed local transfer copy and repeat the sequence for the next weight file.
6. After all four files are present, retain the final target provenance record and use its canonical snapshot path for model loading.

Keep all four transferred weight files in the cluster import directory until the first complete provenance record succeeds. The importer verifies those source files even when matching weights already exist in the cache. An incomplete source directory leaves the first acquisition record pending. After complete provenance exists, a partial retry preserves the original acquisition details.

The import script preserves existing cache content, creates canonical snapshot links and verifies all four weight hashes before publishing the final provenance record. The target's configuration, tokenizer and stopping configuration remain exactly those from Meta. The browser route uses approved browser authentication; the cluster CLI can load the completed local snapshot in offline mode.

### Qualify the EAGLE target and released drafter

1. In the verified EAGLE GPU environment, load the target alone and record ordinary greedy token IDs and stopping behavior.
2. In a separate process, load the released EAGLE-3 drafter and compare its outputs under the same prompt template, precision and generation settings.

   ```bash
   export HF_HOME="$HPML_ROOT/cache/huggingface"
   "$HPML_ROOT/envs/eagle-cu124-py310/bin/python" \
     "$HPML_ROOT/source/hpml-project/baselines/qualify_eagle.py" \
     --output-dir "$HPML_ROOT/runs/eagle-output-check-$SLURM_JOB_ID"
   ```

   Run that command inside a single-L40S allocation. The wrapper executes both stages in separate processes and saves their raw token IDs. Each invocation requires a fresh output directory. The four setup prompts exercise a short answer, a length cap, arithmetic and code. The comparison records returned-output identity and the released producer's native stopping behavior separately.

3. Before performance measurements, resolve output differences and freeze the complete output policy, then apply the broader development-prompt protocol in [metrics.md](metrics.md).

### Observed EAGLE output qualification

Job `13929466` runs the original EAGLE implementation on one NVIDIA L40S at `ins038`, using Torch 2.6.0+cu124 and BF16. The job loads the canonical Meta target and the released EAGLE-3 drafter in separate processes. All fifteen saved drafter tensors match their loaded names, shapes, dtypes and values after the documented conversions. The drafter's target-sourced embedding also matches.

| Setup prompt | Ordinary raw tokens | EAGLE-3 raw tokens | Returned tokens under the shared policy | Returned IDs and stopping match |
|---|---:|---:|---:|---|
| Short `READY` reply | 2 | 2 | 1 | Yes. |
| Integer list with a 16-token cap | 16 | 18 | 16 | Yes. |
| Brief arithmetic | 10 | 10 | 9 | Yes. |
| Python function with a 32-token cap | 32 | 33 | 32 | Yes. |

The raw counts retain a terminal stop token when one occurs. Returned counts exclude that token. The released EAGLE generator exceeds the strict length cap by two tokens for the integer list and one token for the code prompt. The common output adapter clips those extra tokens and preserves exact ordinary output on all four prompts. Native boundary compliance therefore fails on the two capped prompts, while adapted output qualification passes on all four.

Slurm records exit `3:0` after 129 seconds because the wrapper uses exit `3` for this partial result. The model loads and comparisons complete successfully. The [comparison record](results/eagle-output-check-13929466/comparison.json), [ordinary outputs](results/eagle-output-check-13929466/target.json), [EAGLE-3 outputs](results/eagle-output-check-13929466/eagle3.json) and [drafter-loading record](results/eagle-output-check-13929466/drafter-load.json) preserve the evidence. Future benchmark timing includes all work performed before clipping, and broader development-prompt checks precede the final benchmark.

## Preserve the baseline comparison

1. Establish ordinary generation with the target alone in each engine and runner.
2. Load one released draft and verify greedy output IDs and stopping against that ordinary control.
3. Compare ordinary outputs across engines before applying one exact-output condition to a combined ranking.
4. Run the same isolated development prompts with the same precision, template, output cap and stopping rules.
5. Replay the same predetermined concurrent arrivals for each qualified serving implementation.
6. Profile components in separate runs and connect each cycle's time with its proposed, accepted and returned tokens.
7. Record allocated GPU time from the job allocation and preserve any attributable charge records.

Use a clean target-only process for ordinary memory and resource measurements. A loaded drafter can occupy GPU memory even when generation calls an ordinary decoding routine. Keep each released drafter unchanged while establishing its baseline measurements.

## Capture resource use and charges

After a completed job, collect its scheduler record:

```bash
sacct -j JOB_ID --parsable2 \
  --format=JobID,JobName,State,ExitCode,Start,End,ElapsedRaw,AllocTRES,NodeList
```

Replace `JOB_ID` with the measured allocation. Count allocation time once at the allocation level and account for every assigned GPU. Nested job steps describe execution within the allocation. Save separate warmup, measurement and drain timestamps. Preserve timeouts and failed requests.

Allocated GPU-seconds per million returned output tokens equals the summed GPU allocation time divided by returned output tokens, multiplied by one million. Actual GPU charges use attributable billed charges in the numerator. A zero returned-token denominator gives an undefined rate. Slurm accounting supplies resource evidence; actual charges require billing evidence. For the `edu` allocation, attributable GPU charges remain **unmeasured** until that evidence is available.

## Source checks and unresolved items

| Item | Evidence | Status |
|---|---|---|
| Personal account and scratch location | Welcome email; authenticated November 16 session | UNI `aa5886`, account `edu`, group `somedu` and private project directories confirmed. |
| Cluster access procedure | University getting-started page | Confirmed from source. |
| Team execution | Discord general, November 15–16; saved notebook at project commit `c8af4638d2b7ba046765a31425be5e7fda80dc6f` | L40S pilot recorded. |
| CourseWorks | Course syllabus, Pages and Resources file index inspected in Chrome | Course performance workflow verified; personal access instructions came from the welcome email. |
| Ed | `Insomnia` search returned five threads; Compute Credits #66 and storage/outage search results inspected | Course access context verified. Live account and driver evidence appears in the allocation record. |
| Mac VPN installation | Official Columbia installer; Cisco system-approval wizard | Installed; required approvals enabled. |
| VPN authentication | Cisco connection to `vpn.columbia.edu` | Connected; university browser sign-in and Duo tests also succeed. |
| Live cluster connection | Authenticated November 16 SSH session | Connected to `2402-login-002`; login Bash initialization supplies the Slurm configuration. |
| Driver and scheduler GPU names | Live `sinfo` and completed inspection job `13928964` | GPU types verified; `ins056` L40S and driver 550.54.14 recorded. Current node availability remains transient. |
| Private project transfer | User approval; private scratch destination; remote Git revision | Completed; remote source commit matches `c8af4638d2b7ba046765a31425be5e7fda80dc6f`. |
| Personal scratch quota | Shared filesystem capacity inspected | Personal quota remains unverified; shared free space is recorded separately. |
| Gated target access | Approved Meta browser access and completed import job `13929459` | All thirteen canonical target files are obtained directly from Meta, verified and staged in the private cluster cache. |
| EAGLE environment installation | Completed CPU job `13929007` | Python 3.10.22, Torch 2.6.0+cu124, Transformers 4.53.1 and Accelerate 0.26.0 installed; pinned EAGLE model-class imports pass. |
| EAGLE GPU compatibility | Job `13929049`, `ins056`, exit code 0, 153 seconds | CUDA availability and 32-by-32 BF16 matrix multiplication pass. |
| Public EAGLE-3 checkpoint | Job `13929052`, `ins072`, exit code 0, 79 seconds | All eight files downloaded and verified at revision `ada412b672e293d682423de84a095447bf38a637`. |
| Target loading and output correctness | Complete canonical target and all prepared environments | Checks `13929466` and `13929467` match four adapted outputs and record native overshoot. SGLang check `13930206` and patched vLLM check `13930616` pass all four native output/stopping comparisons for their five drafter configurations. |
| Actual GPU charges | Requires attributable billing evidence | Pending. |

Course source links: [CourseWorks](https://courseworks2.columbia.edu/courses/251549), [Resources](https://courseworks2.columbia.edu/courses/251549/files/folder/Resources), [Ed Compute Credits #66](https://edstem.org/us/courses/106104/discussion/8332622), [team Discord general](https://discord.com/channels/1551782531043426455/1551782531572043818).
