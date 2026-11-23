# Track pretrained inference checks with W&B

The project has a separate W&B 0.30.0 environment at `/insomnia001/depts/edu/users/aa5886/hpml-baselines/envs/tracking-py310`. Installation job `13929225` completes with exit code `0:0`; package consistency and the configured capture settings pass. Offline logging job `13929242` also completes with exit code `0:0` and records the three actual PARD-2 setup checks. The inference environments retain their own dependency versions.

The [tracking tool](tracking.py) validates a selected record and prints its contents. Adding `--record` creates a local offline W&B run under `HPML_ROOT/tracking/wandb/`. Cloud synchronization is a separate action. The prepared tool never calls a login function or changes stored credentials.

The saved offline run IDs are `avn1kzox` for dependency imports, `lh92y7zh` for GPU/cache checks, and `99pjbb1s` for released drafter/adapter loading. Their source records identify jobs `13929170` and `13929173`. Every saved run has an empty performance-metric set. The browser confirms the signed-in account `ayoadetayo` and team `ayoadetayo-columbia-university`. The team's [Weights & Biases workspace](https://forge.coreweave.com/wandb/ayoadetayo-columbia-university) now opens through CoreWeave Forge. The user completes cluster CLI authentication privately with `wandb login --verify`; the confirmation identifies the same account and team at `api.wandb.ai`. The project `hpml-pretrained-baselines` has **Team** visibility, and the separate API read-back confirms the exact entity, project and `access=PRIVATE` on November 16, 2026.

Explicit synchronization uploads the eleven reviewed setup and correctness records with their existing run IDs. The cloud API read-back confirms all eleven runs have state `finished` in the [private tracking project](https://wandb.ai/ayoadetayo-columbia-university/hpml-pretrained-baselines), which retains `access=PRIVATE`. The run state describes the completed logging operation. Each run's separate `status` and named checks preserve the observed setup or correctness result, including the two partial native-output qualifications.

EAGLE setup/import evidence from job `13929007` is saved as offline run `idr01ymh` by logging job `13929384` (completed, exit code `0:0`). Its destination is entity `ayoadetayo-columbia-university`, project `hpml-pretrained-baselines`. The record includes the actual source revision, Python/Torch/Transformers/Accelerate versions and completed import check, with empty performance metrics. The three PARD-2 runs above retain their original identities.

W&B coverage for EAGLE's earlier basic GPU check, job `13929049`, remains pending because the original GPU JSON appears in the session output and has no confirmed saved file. Completed EAGLE target qualification supplies its own saved model/output evidence and the correctness record described below. The completed GPU check remains documented in the [Insomnia guide](insomnia.md).

## Completed correctness records

Logging job `13930032` completes with exit code `0:0` and creates exactly one offline record for each completed output comparison. Both records use entity `ayoadetayo-columbia-university`, project `hpml-pretrained-baselines`.

| Method | Output-check job | W&B run | Recorded result |
|---|---|---|---|
| PARD-2 | `13929467` | `ui20ch1y` | Partial: all four adapted token sequences and stop reasons match; raw boundary checks fail on all four prompts. |
| Original EAGLE-3 | `13929466` | `3hkmkmr3` | Partial: all four adapted token sequences and stop reasons match; raw boundary checks fail on the two length-limited prompts. All 15 saved drafter tensors and the target-sourced embedding pass their separate load checks. |

Each correctness record contains `prompts_checked=4` and `prompts_matched=4`, plus named checks for normalized IDs, normalized stopping, native boundaries and adapted/native readiness. Performance metrics remain unmeasured. The records hash the comparison and both stage files; the EAGLE record also hashes its drafter-load evidence. Raw prompts and output IDs remain in the preserved local [PARD-2 evidence](results/pard2-output-check-13929467/comparison.json) and [EAGLE evidence](results/eagle-output-check-13929466/comparison.json), with their complete stage records beside them.

The selectors preserve the observed partial result. Exit code `3` from the qualification wrapper means the explicit output adapter matches the ordinary reference and the released producer's native boundary check fails. W&B records that distinction as `status=partial`; the offline logger itself completes successfully.

SGLang logging job `13930246` completes with exit code `0:0` and records the two successful comparisons from output-check job `13930206`.

| Method and engine | W&B run | Saved run directory under `HPML_ROOT/tracking/wandb/` | Recorded result |
|---|---|---|---|
| DFlash in SGLang 0.5.19 | `sfujnd5f` | `offline-run-20261116_105141-sfujnd5f` | Passed: all four native output sequences, stop reasons, output caps and returned texts match SGLang ordinary generation. |
| EAGLE-3 in SGLang 0.5.19 | `zwtt7rlq` | `offline-run-20261116_105143-zwtt7rlq` | Passed: all four native output sequences, stop reasons, output caps and returned texts match SGLang ordinary generation. |

The [DFlash comparison](results/sglang-output-check-13930206/dflash-comparison.json) and [SGLang EAGLE-3 comparison](results/sglang-output-check-13930206/eagle3-sglang-comparison.json) each establish `prompts_checked=4` and `prompts_matched=4`. The native boundary is SGLang's `/generate` response after its scheduler applies stopping. The records preserve that boundary definition, the exact target and drafter revisions, the container digest, the qualification source hash, and hashes of the ordinary output, speculative output, requested settings and server configuration. SGLang's `speculative_num_draft_tokens` field retains its engine-defined count: 10 for this DFlash configuration and 16 for this EAGLE-3 configuration. The records include no accepted-token measurements or performance metrics.

Both SGLang records are created offline and then synchronized explicitly. Cloud read-back confirms `sfujnd5f` and `zwtt7rlq` are `finished`, with `status=passed` and four matched prompts each. Their captured files contain package requirements, W&B diagnostic logs, run data and synchronization state. The four-prompt results qualify these settings for the next correctness checks; workload-scale quality, streaming behavior and performance still require the planned experiment.

Patched-vLLM logging job `13930688` completes with exit code `0:0` and saves one offline record for each completed comparison from model-check job `13930616`. Explicit synchronization uploads these three records, and cloud read-back confirms each run is `finished` with `status=passed`, `prompts_checked=4` and `prompts_matched=4`.

| Method | Ordinary-generation reference | W&B run | Saved run directory under `HPML_ROOT/tracking/wandb/` |
|---|---|---|---|
| P-EAGLE | vLLM Model Runner V1 | `3e6a0nw5` | `offline-run-20261116_112530-3e6a0nw5` |
| H-Spec | vLLM Model Runner V2 | `k0s6ddkc` | `offline-run-20261116_112532-k0s6ddkc` |
| DSpark | vLLM Model Runner V2 | `4t1hxfnv` | `offline-run-20261116_112534-4t1hxfnv` |

The [P-EAGLE comparison](results/hspec-output-check-13930616/peagle-comparison.json), [H-Spec comparison](results/hspec-output-check-13930616/hspec-comparison.json) and [DSpark comparison](results/hspec-output-check-13930616/dspark-comparison.json) each pass native output IDs, stopping, caps and returned-text checks on all four prompts. The loading check establishes complete checkpoint input-key coverage and completion of the released loader. Exact loaded tensor-value equality remains unmeasured for these three methods. W&B records `checkpoint_key_coverage` separately and identifies the loading audit's scope in configuration.

Each patched-vLLM record preserves the target and drafter revisions, H-Spec/vLLM/Speculators source revisions, patch and wheel hashes, package versions, actual runner and proposer classes, observed attention backend and selected inference settings. The record also hashes the comparison, both output stages, requested settings, loaded-model inventories, checkpoint inventories and loading audit files. The native producer boundary is vLLM's `CompletionOutput.token_ids` after scheduler stopping. Internal proposals discarded by the verifier need separate instrumentation during the common baseline measurements. The three offline records contain no performance measurements, model files, raw prompts or output IDs.

## Existing assignment and notebook evidence

The inspected HW2 starter package lists `wandb` as a dependency. The saved HW2 notebooks, associated scripts, text/HTML files and starter ZIP contain no W&B entity, project or run URL. The saved P-EAGLE notebook contains links to [einnuian-personal/peagle-ffn](https://wandb.ai/einnuian-personal/peagle-ffn), including its three training runs and evaluation run. Those links identify the existing notebook's tracking destination. The new setup records use the project label `hpml-pretrained-baselines` under the verified team `ayoadetayo-columbia-university`. Create the project with explicit team visibility and verify its access setting before synchronizing any run.

## What a record contains

| Field | Meaning |
|---|---|
| `stage` | `setup`, `correctness` or `benchmark`. Setup records establish installation, loading or small runtime checks. |
| `status` | `passed`, `failed` or `partial`, according to the recorded evidence. |
| `config` | Explicit identifiers and settings, such as method, code/model revisions, software versions, GPU model, precision, workload hash and arrival rate. |
| `checks` | Named outcomes with values `passed`, `failed` or `pending`. |
| `metrics` | Measured nonnegative numeric values from the common baseline protocol. Setup records contain no performance metrics. |
| `evidence_sha256` | A hash of the local source record. The source file stays local. |

The tool rejects unrecognized fields, nonfinite measurements and raw prompt fields. Omit unmeasured quantities. Missing billing evidence leaves actual-charge fields absent. Estimated charges and actual charges have separate field names. The [common metrics and controls](metrics.md) define counts, timing boundaries, GPU allocation and charge normalization.

W&B supplies bookkeeping fields such as `_runtime`, `_timestamp` and `_step`. For these setup and correctness records, `_runtime` measures elapsed time inside the offline logging run, which reads evidence after the GPU check finishes. Inference latency, request throughput and GPU time remain unmeasured. Use the explicit experiment metric fields and their measurement boundaries when the benchmark produces those results.

Automatic code capture, Git capture, console capture, system metadata and background system metrics are disabled through the pinned [W&B settings implementation](https://github.com/wandb/wandb/blob/v0.30.0/wandb/sdk/wandb_settings.py). The tool logs selected scalars and identifiers. Model files, notebooks, source files and prompt/response text stay in the local and cluster project folders. Each offline run includes W&B's own run bookkeeping and SDK diagnostic files; inspect the specific run before synchronization.

## Record completed checks

1. In a cluster shell, select the private project and tracking interpreter.

   ```bash
   export HPML_ROOT=/insomnia001/depts/edu/users/aa5886/hpml-baselines
   TRACK_PYTHON="$HPML_ROOT/envs/tracking-py310/bin/python"
   TRACK_TOOL="$HPML_ROOT/source/hpml-project/baselines/tracking.py"
   ```

2. Before recording a completed check, review the fields selected from its actual saved evidence.

   ```bash
   "$TRACK_PYTHON" "$TRACK_TOOL" --pard2-check gpu --name pard2-gpu --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --eagle-check setup --name eagle3-setup --entity ayoadetayo-columbia-university
   ```

   For the completed correctness comparisons, use these validation commands:

   ```bash
   "$TRACK_PYTHON" "$TRACK_TOOL" --pard2-output "$HPML_ROOT/runs/pard2-output-check-13929467" --name pard2-correctness-13929467 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --eagle-output "$HPML_ROOT/runs/eagle-output-check-13929466" --name eagle3-correctness-13929466 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --sglang-dflash-output "$HPML_ROOT/runs/sglang-qualification-13930206" --name sglang-dflash-correctness-13930206 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --sglang-eagle-output "$HPML_ROOT/runs/sglang-qualification-13930206" --name sglang-eagle3-correctness-13930206 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --peagle-output "$HPML_ROOT/runs/hspec-models-13930616" --name vllm-peagle-correctness-13930616 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --hspec-output "$HPML_ROOT/runs/hspec-models-13930616" --name vllm-hspec-correctness-13930616 --entity ayoadetayo-columbia-university
   "$TRACK_PYTHON" "$TRACK_TOOL" --dspark-output "$HPML_ROOT/runs/hspec-models-13930616" --name vllm-dspark-correctness-13930616 --entity ayoadetayo-columbia-university
   ```

   The `--pard2-check` options `setup`, `gpu` and `draft` read the corresponding PARD-2 environment records. `--eagle-check setup` reads EAGLE's saved package/import record. The two SGLang selectors read the named method's comparison and both its ordinary and speculative stage records. The selectors preserve each original check's job identity and any saved timestamp. The EAGLE setup record has a job identity and no explicit check timestamp. The logger's own execution time identifies when the record enters W&B.

   The patched-vLLM selectors pair P-EAGLE with `target-v1` and H-Spec/DSpark with `target-v2`. They retain the comparison's exit-code meaning: `0` records a passed qualification, `3` records an adapted-output partial qualification, and `1` or `2` records a failed completed comparison. A record with failed checkpoint-key coverage keeps the failed qualification outcome even when output IDs match. A stage that produces no completed comparison requires review of its saved failure evidence before selecting a tracking record.

3. Inside a CPU allocation, save the reviewed check as an offline run.

   ```bash
   "$TRACK_PYTHON" "$TRACK_TOOL" --eagle-check setup --name eagle3-setup --entity ayoadetayo-columbia-university --record
   ```

   The tool prints the exact offline run directory. Each invocation creates a new record. Preserve the first useful record and avoid repeated copies of the same completed check.

4. For another method or a later measurement, prepare a JSON record containing the allowed fields and validate that record before adding `--record`.

   ```bash
   "$TRACK_PYTHON" "$TRACK_TOOL" --input "$HPML_ROOT/runs/selected-result.json" --name released-baseline-check
   ```

   Select aggregate results from the original evidence. Keep prompt-level IDs, output text, traces and complete run records in the private project. The tool does not discover or upload nearby files.

5. In the signed-in W&B team workspace, create `hpml-pretrained-baselines` under `ayoadetayo-columbia-university` and select **Team** in **Project Visibility**.

   **Team** visibility gives members of the parent team access to the project. **Restricted** visibility gives access to explicitly invited team members. For an existing project, open the visibility control beside its name and verify the selected scope. Preserve an existing **Restricted** scope. W&B documents both choices in [Manage access control for projects](https://docs.wandb.ai/platform/hosting/iam/access-management/restricted-projects/).

   W&B 0.30.0's public [`Api.create_project`](https://github.com/wandb/wandb/blob/v0.30.0/wandb/apis/public/api.py#L219-L232) accepts a name and entity and leaves visibility to server defaults. The installed generated [`UpsertModelInput`](https://github.com/wandb/wandb/blob/v0.30.0/wandb/apis/_generated/input_types.py#L42-L66) includes an `access` string. W&B uses `PRIVATE` and `RESTRICTED` for project-backed registry access in its [visibility conversion](https://github.com/wandb/wandb/blob/v0.30.0/wandb/apis/public/registries/_utils.py#L153-L190) and [registry creation](https://github.com/wandb/wandb/blob/v0.30.0/wandb/apis/public/registries/registry.py#L323-L335). The supported project visibility control supplies the explicit regular-project setting for this procedure.

6. After private CLI authentication, verify the project identity and saved access setting with the pinned tracking interpreter.

   ```bash
   "$TRACK_PYTHON" - <<'PY'
   import json
   import wandb

   if wandb.__version__ != "0.30.0":
       raise SystemExit("Use the inspected W&B 0.30.0 tracking environment.")
   entity = "ayoadetayo-columbia-university"
   name = "hpml-pretrained-baselines"
   query = """
   query HpmlProjectAccess($entity: String!, $name: String!) {
     project(entityName: $entity, name: $name) {
       id
       name
       entityName
       access
     }
   }
   """
   api = wandb.Api(overrides={"base_url": "https://api.wandb.ai"})
   result = api._service_api.execute_graphql(
       query, {"entity": entity, "name": name}
   )
   project = result.get("project")
   if not isinstance(project, dict):
       raise SystemExit("Project lookup failed; keep the runs offline.")
   if project.get("name") != name or project.get("entityName") != entity:
       raise SystemExit("Project identity differs; keep the runs offline.")
   if project.get("access") not in {"PRIVATE", "RESTRICTED"}:
       raise SystemExit("Private project access is unconfirmed; keep the runs offline.")
   print(json.dumps({key: project[key] for key in
                     ("id", "name", "entityName", "access")}, indent=2))
   PY
   ```

   The command reads four project fields through the installed SDK's [GraphQL transport](https://github.com/wandb/wandb/blob/v0.30.0/wandb/apis/public/service_api.py#L135-L196). The transport is an internal interface pinned to version 0.30.0. The SDK uses the existing authenticated session. The command preserves credentials and uploads no run data. A lookup, identity or access failure stops the command. Complete both the visible **Team**/**Restricted** check and the successful API read-back before proceeding.

7. After both visibility checks pass, preview and synchronize one reviewed offline run with an explicit path.

   ```bash
   "$HPML_ROOT/envs/tracking-py310/bin/wandb" sync --dry-run --entity ayoadetayo-columbia-university --project hpml-pretrained-baselines /absolute/path/to/one/offline-run
   "$HPML_ROOT/envs/tracking-py310/bin/wandb" sync --entity ayoadetayo-columbia-university --project hpml-pretrained-baselines /absolute/path/to/one/offline-run
   ```

   Replace the run path with the directory printed by the logger. Use the verified team and project for every prepared record. W&B's interactive `wandb login --verify` command uses its credential interface. Complete API-key creation and credential entry privately when the service requests those steps. Keep credentials out of scripts, command arguments and project records. The official [login reference](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-login.mdx) describes authentication, and the [sync reference](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-sync.mdx) documents explicit run selection and the dry-run preview.

## Future benchmark records

Use one record for each method, configuration, workload, arrival trace and repeat. Record each released method and its ordinary-generation control with their exact identities. Preserve a separate ordinary control for each engine and model runner.

Metric field names carry units: `ttft_seconds`, `streaming_pause_seconds`, `tpot_seconds`, `completion_latency_seconds`, `returned_tokens_per_second`, `allocated_gpu_seconds`, `actual_gpu_charges` and memory fields ending in `_bytes`. Latency and rate fields accept `_p50`, `_p95`, `_p99`, `_max` and `_mean` suffixes. Specify the aggregation and measurement interval in the run evidence. Record token counts and request outcomes beside normalized rates. A pooled result needs its pooled count and time denominator; averaging already-normalized rates can produce a different result.

The current setup and correctness records establish the completed preparation and four-prompt observations described above. Performance measurements, concurrent arrivals, streaming timing and actual GPU charges enter W&B when the corresponding baseline run produces evidence. The project uses pretrained models throughout.
