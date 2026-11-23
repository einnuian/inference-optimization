# Team FFN-width pilot evaluation

The experiment evaluates the team's custom P-EAGLE-style drafter at 100%, 75%, and 50% feed-forward width. Each drafter has four layers and proposes eight tokens per verification round. The ordinary Transformers decoder provides the speedup reference. The current procedure imports the teammate's existing pilot checkpoints and evaluates all four configurations on one L40S GPU.

The notebook's pilot configuration specifies 200 training examples, 50 held-out examples, one epoch, and 50 optimizer updates per width. The import verifies the saved updates, parameter shapes, weights, and supplied data. Its manifest records the unknown original training revisions and seed explicitly. The available data determines how many held-out prompts can be checked.

The evaluation notebook comes from commit `64b24c42f1859e25b0ffb69293e7e9f793a697d3`. Its SHA-256 is `401a612174145e4eb55e324223045804bc7840514944152d2f0ecdb5cb642490`. The evaluation target revision is `0e9e39f249a16976918f6564b8830bc894c89659`. The original training target revision remains unverified. The optional training runner pins UltraChat to revision `8049631c405ae6576f93f445c6b8166f76f5505a`.

## Prepare the runtime and source

1. On the assigned Insomnia compute node, enter the project checkout and run the environment setup script.

```bash
bash baselines/setup_team_peagle.sh
```

The script uses the existing `HPML_ROOT` and creates `envs/team-peagle-cu124-py310`. Hardlinks share the cached package files' disk blocks. The environment records its package list, installation mode, setup source hash, GPU, driver, and Slurm job.

2. Set the paths for the reproduction and export the pinned notebook.

```bash
export HPML_TEAM_PYTHON="$HPML_ROOT/envs/team-peagle-cu124-py310/bin/python"
export HPML_TEAM_RUN="$HPML_ROOT/runs/team-peagle-pilot-20261122"
export HPML_PREPARED="<absolute-path-to-the-shared-prepared.json>"
export HPML_GPU_UUID="<assigned-L40S-GPU-UUID>"
mkdir -p "$HPML_TEAM_RUN"
"$HPML_TEAM_PYTHON" baselines/team_peagle_trial.py export-source \
  --output "$HPML_TEAM_RUN/source.ipynb"
```

Replace both angle-bracket values with the existing trial paths and assigned GPU identity. `HF_HOME` must point to the verified target cache. The export preserves the notebook in the working checkout.

## Run an optional local reproduction

The current evaluation imports the existing checkpoints through the next section. The following command performs new training when a separate reproduction is requested.

1. From the prepared package, run setup, training, qualification, measurement, diagnostics, and comparison in sequence.

```bash
"$HPML_ROOT/envs/eagle-cu124-py310/bin/python" baselines/team_peagle_sequence.py \
  --output "$HPML_TEAM_RUN" \
  --prepared "$HPML_PREPARED" \
  --hf-home "$HF_HOME" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --notebook "$HPML_TEAM_RUN/source.ipynb" \
  --setup \
  --train \
  --diagnostics \
  --run
```

The controller preserves server logs and phase timestamps. It evaluates the ordinary Transformers decoder and all three widths on the same physical GPU. Each measured configuration uses the same 30 prompts and three repetitions. The training overlap check reports any normalized-text match with the evaluation prompts and stops before optimizer training.

## Import and evaluate original checkpoints

1. Set the path to the shared directory that contains `drafter_100.pt`, `drafter_75.pt`, `drafter_50.pt`, and `train_data.jsonl`.

```bash
export HPML_TEAM_SHARED="/insomnia001/depts/edu/users/ahn2130/peagle_ffn_pilot"
```

2. Run the sequential controller to import, qualify, measure, and compare all four configurations.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_sequence.py \
  --output "$HPML_TEAM_RUN" \
  --prepared "$HPML_PREPARED" \
  --hf-home "$HF_HOME" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --notebook "$HPML_TEAM_RUN/source.ipynb" \
  --import-from "$HPML_TEAM_SHARED" \
  --diagnostics \
  --run
```

The controller runs the paired ordinary decoder first. It loads one configuration at a time and saves every server and recorder log. Each sequence attempt contains `phases.jsonl` with process, GPU, Slurm job, and start/end timestamps. Primary results go into `results`, and diagnostic results go into `diagnostic-results`. A new allocation should use a new result directory and its own ordinary-decoder reference.

The example command creates references to the shared checkpoints. The import records their source paths and hashes. Before every load, the adapter verifies the complete checkpoint hash. The shared files must remain readable and retain the recorded contents. To create independent copies, add `--copy-checkpoints` when the destination has sufficient space.

If the teammate supplies the exact original vocabulary or settings, add `--original-vocabulary <vocabulary.json>` or `--reported-settings <settings.json>`. The importer compares the original vocabulary with its reconstruction and retains supplied settings with attribution.

The importer recovers prompt text from the saved token IDs and checks an exact token-ID roundtrip. It compares normalized text with all 30 evaluation prompts. Normalization uses Unicode NFC, case folding, and collapsed whitespace. Any match stops the sequence and appears in `evaluation-overlap.json`. The prompt selection remains intact.

The command bundle requires the project's `baselines` Python and shell sources, the pinned notebook copy, the shared prepared prompt JSON, and access to the verified target cache. A teammate can use the same bundle on an allocated L40S. The runtime and target checks must pass in that account before model execution.

## Separate primary measurements and diagnostics

The current allocation runs the four primary configurations. A later allocation can collect diagnostic observations. Both phases use all 30 prompts and three repetitions.

1. Before the controller starts, save `sequence-policy.json` in the result directory to select the phase.

```json
{
  "phases": ["primary"],
  "checkpoint_storage": "verified_references",
  "reason": "Measure all primary configurations in this allocation and collect diagnostics separately."
}
```

The controller records the policy and its file hash. The phase policy takes precedence over the `--diagnostics` option. The storage policy takes precedence over `--copy-checkpoints`.

2. For a separate diagnostic allocation, use a new result directory and set the policy's `phases` value to `["diagnostic"]`.
3. Pass the primary allocation's checkpoint directory through `--checkpoint-dir` and its source copy through `--notebook`.

The diagnostic controller qualifies and evaluates the ordinary Transformers decoder and all three widths on that allocation's GPU. Each phase preserves its own GPU, job, source, and timing records.

## Run or resume individual training steps

The training runner supports an individual training step. The controller executes that step when its command includes `--train`.

1. Generate the examples and train all three widths.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_train.py \
  --notebook "$HPML_TEAM_RUN/source.ipynb" \
  --hf-home "$HF_HOME" \
  --prepared "$HPML_PREPARED" \
  --output "$HPML_TEAM_RUN/checkpoints"
```

The runner saves `train_data.jsonl`, `draft_vocab.json`, `training-manifest.json`, and `drafter_100.pt`, `drafter_75.pt`, and `drafter_50.pt`. Each checkpoint retains the notebook's weights, optimizer state, and update count. The manifest records the source revisions, seeds, vocabulary, file hashes, and completion status. Training uses the notebook's batch size of 32 for target answer generation.

2. When an interrupted run has a saved manifest, resume that directory by adding `--resume` to the same command.

The resume check requires the same source, target, dataset, runtime, and completed training data. The notebook saves the final checkpoint at update 50. The evaluation loader requires a completed training manifest and checks the checkpoint hashes.

## Qualify and measure each configuration

The configuration order is `team-ar`, `team-ffn-100`, `team-ffn-75`, and `team-ffn-50`. Complete the following steps for each configuration. Set `HPML_TEAM_CONFIGURATION` to the corresponding identifier.

1. Compare the streaming adapter's output with the original notebook functions.

```bash
export HPML_TEAM_CONFIGURATION="team-ar"
"$HPML_TEAM_PYTHON" baselines/team_peagle_trial.py qualify \
  --configuration "$HPML_TEAM_CONFIGURATION" \
  --notebook "$HPML_TEAM_RUN/source.ipynb" \
  --hf-home "$HF_HOME" \
  --checkpoint-dir "$HPML_TEAM_RUN/checkpoints" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --prepared "$HPML_PREPARED" \
  --output "$HPML_TEAM_RUN/qualification-$HPML_TEAM_CONFIGURATION.json"
```

Qualification compares all three warmup prompts and the longest evaluation input at output limits of 1, 9, and 33 tokens. Every case must preserve the original token IDs and stopping condition.

2. Start the selected configuration's server.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_trial.py serve \
  --configuration "$HPML_TEAM_CONFIGURATION" \
  --notebook "$HPML_TEAM_RUN/source.ipynb" \
  --hf-home "$HF_HOME" \
  --checkpoint-dir "$HPML_TEAM_RUN/checkpoints" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --server-record "$HPML_TEAM_RUN/server-$HPML_TEAM_CONFIGURATION.json"
```

The server listens on compute-node loopback port 31002. It emits the first target token after prefill and then emits each committed group. The shared stopping rule ends the output at its first EOS token or at 2,048 generated tokens. The selected configuration occupies the GPU until the server exits.

3. From another shell on the same compute node, run the shared measurement procedure.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_trial.py record \
  --configuration "$HPML_TEAM_CONFIGURATION" \
  --prepared "$HPML_PREPARED" \
  --server-record "$HPML_TEAM_RUN/server-$HPML_TEAM_CONFIGURATION.json" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --output "$HPML_TEAM_RUN/results/$HPML_TEAM_CONFIGURATION"
```

The recorder uses the same 30 prompts, three warmups, three repetitions, token IDs, clocks, and output limit as the initial trial. TTFT measures the time until the compute-node client receives its first nonempty decoded text. The output includes the shared block files, summaries, raw stream events, and provenance.

4. After the recorder finishes, stop the server before starting the next configuration.

## Calculate the paired results

1. For each drafter width, compare its result directory with the ordinary Transformers result directory.

```bash
for HPML_TEAM_CONFIGURATION in team-ffn-100 team-ffn-75 team-ffn-50; do
  "$HPML_TEAM_PYTHON" baselines/team_peagle_trial.py compare \
    "$HPML_TEAM_RUN/results/$HPML_TEAM_CONFIGURATION" \
    "$HPML_TEAM_RUN/results/team-ar" \
    --output "$HPML_TEAM_RUN/comparison-$HPML_TEAM_CONFIGURATION.json"
done
```

The comparison checks delivered token IDs, output lengths, and stopping conditions for every paired prompt. The team's saved HTML shows output differences on two of four prompts. The new measurements determine the agreement rate on the fixed 30 evaluation prompts.

## Collect separate diagnostic observations

1. Launch each configuration again with `--diagnostic-dir "$HPML_TEAM_RUN/diagnostics/$HPML_TEAM_CONFIGURATION"` and a new server-record path.
2. Run the recorder with that new server record and a new result directory under `diagnostic-results`.

Diagnostic records include CUDA phase intervals and process memory peaks. Primary records supply throughput and response timing. Both runs preserve integer verification-round counts. Each drafter round records eight proposed tokens, the accepted prefix, one retained target token, and the delivered count after stopping. The initial target token has a separate count. These records support acceptance, acceptance length, and all eight position measurements.

The result provenance identifies original teammate checkpoints or a local reproduction. The released P-EAGLE configuration keeps its separate row and matching vLLM reference. Resource records identify setup, import, qualification, primary measurement, and diagnostic measurement as separate phases. A local training run also records its data preparation and training phase.
