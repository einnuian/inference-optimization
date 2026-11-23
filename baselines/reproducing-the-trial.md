# Reproduce the first 30-prompt trial

Run the following commands from the repository root. The project README explains the research questions and current findings.

The commands below describe a new primary measurement sequence. The commands require an allocated Insomnia L40S, the pinned environments, and accessible model weights. The completed observations remain preserved in their publication archive.

### Prepare inputs and runtimes

1. Clone this repository with its history and enter the checkout.

```bash
git clone https://github.com/einnuian/hpml-project.git
cd hpml-project
```

2. Follow the [Insomnia setup guide](insomnia.md) and prepare the [SGLang runtime](sglang-setup.md) and [patched vLLM runtime](hspec-setup.md).

SGLang uses release 0.5.19 at source revision `0bcd822377da7b5718e674eaf9c870d349424dd1` in the pinned CUDA 12.9 container. P-EAGLE uses patched vLLM 0.25.1 with ModelRunnerV1. The pilot runtime uses Python 3.10, PyTorch 2.6.0 with CUDA 12.4, and Transformers 4.57.6. The setup scripts record actual packages, driver, host, job, and GPU identity. Model access and checkpoint acquisition require the corresponding account permissions.

3. Inside the assigned compute node, set the project paths and identify its allocated GPU.

```bash
export HPML_ROOT="/insomnia001/depts/edu/users/$USER/hpml-baselines"
export HF_HOME="$HPML_ROOT/cache/huggingface"
export HPML_PREPARED="$PWD/benchmarks/inputs/trial-tokenized.json"
export HPML_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader)"
export HPML_REPLAY_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export HPML_TRIAL="$HPML_ROOT/results/replay-$HPML_REPLAY_STAMP/trial-first30-$(date -u +%Y%m%d)"
export HPML_JOB_ID="$SLURM_JOB_ID"
mkdir -p "$HPML_TRIAL"
python3 baselines/trial_recorder.py validate benchmarks/inputs/trial-first-three.jsonl
```

The GPU variable must identify the single L40S assigned to the allocation. The sequence controller checks the compute allocation and runtime before inference. The prepared bundle preserves exact input token IDs and stopping rules. Its SHA-256 is `8d3302b67672b22bda85c929b464c56c8935d823609776fa96da3d091cfe6e1e`.

### Measure the released methods

1. In a second shell on the same allocated node, set the same trial path and GPU UUID, then start the passive memory recorder.

```bash
python3 baselines/trial_resources.py gpu \
  --device "$HPML_GPU_UUID" \
  --output "$HPML_TRIAL/gpu-memory.jsonl"
```

2. In the main shell, run the five configurations sequentially.

```bash
python3 baselines/trial_sequence.py \
  --prepared "$HPML_PREPARED" \
  --output "$HPML_TRIAL" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --configurations ar-sglang eagle3 dflash ar-vllm-v1 peagle \
  --run
```

The controller starts one server, checks one-token and natural-stop streaming, records warmups and 90 timed responses, then stops that server. Saved records preserve token IDs, text, event times, engine settings, and native acceptance counts. A repeated invocation validates completed configurations and preserves incomplete attempts before retrying them.

3. After inference finishes, stop the memory recorder with **Control+C** and preserve its completed log.
4. After the allocation ends, restore `HPML_JOB_ID` and `HPML_TRIAL` in the login shell to their recorded values, then save the final Slurm record.

```bash
python3 baselines/trial_resources.py allocation "$HPML_JOB_ID" \
  --output "$HPML_TRIAL/allocation.json"
```

Set `HPML_JOB_ID` to the allocation's recorded job ID. The [measurement runbook](measurement-runbook.md) explains the clocks, manual prompt interface, and project-wide resource accounting.

### Measure the original pilot checkpoints

1. In an allocated compute node with the prepared target cache, create a fresh pilot run and prepare its runtime.

```bash
export HPML_TEAM_RUN="$HPML_ROOT/runs/pilot-replay-$HPML_REPLAY_STAMP"
export HPML_TEAM_SHARED="/insomnia001/depts/edu/users/ahn2130/peagle_ffn_pilot"
bash baselines/setup_team_peagle.sh
export HPML_TEAM_PYTHON="$HPML_ROOT/envs/team-peagle-cu124-py310/bin/python"
```

2. Run the primary pilot sequence with its matching AR reference.

```bash
"$HPML_TEAM_PYTHON" baselines/team_peagle_sequence.py \
  --output "$HPML_TEAM_RUN" \
  --prepared "$HPML_PREPARED" \
  --hf-home "$HF_HOME" \
  --gpu-uuid "$HPML_GPU_UUID" \
  --import-from "$HPML_TEAM_SHARED" \
  --configurations team-ar team-ffn-100 team-ffn-75 team-ffn-50 \
  --run
```

The shared directory must contain the three original checkpoints and `train_data.jsonl`. The controller exports the notebook from pinned commit `64b24c42f1859e25b0ffb69293e7e9f793a697d3`. It checks checkpoint hashes, vocabulary, model dimensions, input overlap, and source-equivalent generation before primary measurement. The default phase is primary. The command preserves the imported weights and evaluates the four configurations sequentially on the assigned GPU. The [pilot runbook](team-peagle-runbook.md) documents individual adapter commands and optional future training.

### Rebuild tables from saved observations

1. Download and extract the final recording ZIP from the [final W&B artifact](https://wandb.ai/ayoadetayo-columbia-university/hpml-first30-trial-2026-11-22/runs/48b0a0ffa383382fa011).
2. Set `HPML_SAVED_TRIAL` to the extracted `trial-first30-20261122` directory and choose a fresh export directory.

```bash
export HPML_EXPORT="$(mktemp -d)/trial-export"
python3 baselines/trial_export.py \
  --trial "$HPML_SAVED_TRIAL" \
  --catalogue baselines/metric-catalogue.ts \
  --output "$HPML_EXPORT" \
  --memory-samples gpu-memory-snapshot-20261123T013545Z.jsonl \
  --additional-memory-samples gpu-memory-snapshot-20261123T015408Z.jsonl \
  --report-output measurements/report/generated
```

The catalogue snapshot makes this command independent of the Site checkout. The exporter validates evidence, preserves unavailable-cell reasons, and regenerates the five report fragments. The [report procedure](../measurements/report/README.md) covers LaTeX compilation, Site packaging, and W&B publication. The historical partial AR attempt remains available as a separate archive.

