# Update the evaluation report

The report uses one column throughout. The main section contains OTPS, paired speedup, allocated GPU time per million output tokens, mean acceptance length, and interpretation. The appendix contains the remaining measured metrics. Each table uses methods as rows and benchmarks as columns. Bold means identify the best observed value within a benchmark and metric definition. Ties at the displayed precision share bold formatting. The caption preserves the distinction between an observed ranking and a statistical claim.

The five initial configurations have completed 150 benchmark blocks and 450 timed requests. EAGLE-3 completed 90 separate diagnostic requests. DFlash saved four diagnostic blocks before the sequence paused. The four pilot configurations add 120 benchmark blocks and 360 responses. The complete report contains 810 timed responses and 2,056 measured cells. Further diagnostics remain paused. DSpark and its vLLM ModelRunnerV2 reference also remain deferred.

`report.tex` contains the editable paper. `generated` contains evidence-derived tables and findings. `report.pdf` is the reviewed compiled report. `benchmark_report.tex` contains the complete standalone source for Overleaf. `HPML_Midpoint_Report_All_Categories.pptx` contains the eleven-slide presentation, with an explanation of at most two sentences for each result excerpt. The presentation includes mathematics, code, conversation, instruction following, and summarization. The original proposal remains in `PROJECT/proposal`.

## Assemble completed evidence

1. Create a fresh publication directory with a `trial-first30-20261122` subdirectory, and copy the completed configuration records into that subdirectory.

The final publication root is `measurements/publications/final-pilot-20261123T015828Z/trial-first30-20261122`. Its `export` and `report` directories preserve the delivered tables and report. Preserve that version when preparing future updates.

Each complete configuration contains `run.json`, `warmup.json`, 30 benchmark block files, `summary.json`, and raw stream events. Each block contains three requests from one benchmark and repetition. Preserve native counter records and qualification evidence. Keep each speculative method with its matching AR reference.

2. Copy diagnostic result directories into the trial's `diagnostics` directory, and preserve the relative trace paths under `diagnostics/attempts`.

The recorded `output_results_relative` identifies each trace directory. The exporter checks hardware, inputs, settings, clock boundaries, native counters, and round identities. Primary records continue to supply throughput and response latency. Diagnostic records supply eligible component, tree, and framework-memory observations. A partial diagnostic run contributes only its completed validated blocks.

3. Copy each selected pilot configuration into the same publication root, and preserve its source attempt.

The team directory names are `team-ar`, `team-ffn-100`, `team-ffn-75`, and `team-ffn-50`. Preserve their checkpoint and import manifests. The current experiment imports original teammate checkpoints. The supplied data contains 200 training examples and zero held-out examples. Original training revisions and seed fields remain unknown where evidence is unavailable.

The active 75%, 50%, and Transformers AR records share the continuation GPU. The original 100% record uses another physical L40S. Its active paired metrics remain empty. Pairing checks require compatible input, target, settings, repetition, and physical GPU evidence. A diagnostic run requires the same physical GPU as its corresponding primary run.

4. Preserve the selected memory samples in an immutable file within the trial directory.

The final export uses `gpu-memory-snapshot-20261123T013545Z.jsonl` and `gpu-memory-snapshot-20261123T015408Z.jsonl`. The continuation sampler starts during AR evaluation. The 75% and 50% device-memory cells therefore remain empty. A later export uses a new snapshot filename. The active sampler writes its own file. Freeze diagnostic trace files before export so their source hashes remain reproducible.

## Record actual allocations

1. After an allocation ends, query its final Slurm record from the cluster checkout.

```bash
python3 baselines/trial_resources.py allocation 14056875 \
  --output "$HPML_ROOT/results/trial-first30-20261122/allocation.json"
```

The command writes a fresh file. The job identifier above is the shared initial allocation. Each additional job needs its own allocation record. Include elapsed time from cancelled or unsuccessful jobs in the project total.

2. After copying that record to the Mac, create the shared allocation report.

```bash
python3 baselines/trial_allocation_report.py \
  --trial measurements/trial-first30-20261122 \
  --allocation measurements/trial-first30-20261122/allocation.json \
  --output measurements/trial-first30-20261122/allocation-report.json \
  --configurations ar-sglang eagle3 dflash ar-vllm-v1 peagle
```

The report validates the Slurm job ID, counts each request ID once, and separates primary requests, warmups, qualification, diagnostics, and score probes. GPU-hours and CPU-core-hours cover the entire allocation once. Missing token evidence remains explicit. Method-by-benchmark GPU-time ratios use recorded block intervals and the matching final allocation count.

3. For each additional job, run the same commands with its actual job ID, separate allocation files, and the configurations that used that job.
4. Refresh `compute-usage.json` through the [measurement runbook](../../baselines/measurement-runbook.md).

The project-wide snapshot supplies all parent-job totals. The Site packager recognizes the root resource records and per-job files at `allocations/<numeric-job-id>.json`. The final publication contains allocations `14056875` and `14058516`. Keep each parent job's identity and total separate.

## Export the measurement tables

1. From `hpml-project`, set a fresh export path and the existing immutable memory snapshot.

```bash
export HPML_LOCAL_TRIAL="measurements/publications/final-pilot-20261123T015828Z/trial-first30-20261122"
export HPML_EXPORT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export HPML_EXPORT="/private/tmp/hpml-export-$HPML_EXPORT_STAMP"
export HPML_MEMORY_SNAPSHOT="gpu-memory-snapshot-20261123T013545Z.jsonl"
export HPML_CONTINUATION_MEMORY="gpu-memory-snapshot-20261123T015408Z.jsonl"
```

2. Export the actual evidence and refresh the report fragments.

```bash
python3 baselines/trial_export.py \
  --trial "$HPML_LOCAL_TRIAL" \
  --catalogue baselines/metric-catalogue.ts \
  --output "$HPML_EXPORT" \
  --memory-samples "$HPML_MEMORY_SNAPSHOT" \
  --additional-memory-samples "$HPML_CONTINUATION_MEMORY" \
  --report-output measurements/report/generated
```

The output directory must be new. The command preserves the authored paper and refreshes five generated fragments. The initial cells are in `trial-results.json`, and the team cells are in `team-trial-results.json`. Their corresponding evidence files preserve source hashes, observations, and empty-cell reasons.

A zero represents an observed zero. An empty cell represents unavailable evidence. The bracketed count gives eligible repetitions. Standard deviation appears when at least two repetitions exist. The exporter keeps DSpark and its reference deferred unless a later export includes `--include-deferred`.

3. Inspect completion counts, agreement results, source hashes, and the reason for each unavailable metric.
4. Run the exporter tests before using changed analysis code.

```bash
python3 -m unittest discover -s tests -p 'test_trial_export.py' -v
```

The agreement summary records exact matches and output lengths separately. EAGLE-3 tree-candidate acceptance and linear-proposal acceptance have different denominators. The tree-prefix diagnostic retains its own variant. Component timing requires complete round boundaries. Benchmark task accuracy remains unmeasured.

## Prepare the Site and W&B records

1. Package saved responses and the selected export for the Site.

```bash
python3 baselines/trial_site_assets.py \
  --results "$HPML_LOCAL_TRIAL" \
  --prompts benchmarks/inputs/trial-tokenized.json \
  --site ../site \
  --catalogue-export "$HPML_EXPORT"
```

The packager writes the recording index and required public assets. It derives response counts from complete configurations. The current five configurations contribute 450 recordings. Four completed team configurations add 360 recordings.

2. Merge the initial and team result cells into the Site's result file.

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["HPML_EXPORT"])
primary = json.loads((source / "trial-results.json").read_text())
team = json.loads((source / "team-trial-results.json").read_text())
assert primary["trialId"] == team["trialId"]
cells = primary["cells"] + team["cells"]
keys = [tuple(row[field] for field in
              ("metricId", "variant", "method", "benchmark", "position"))
        for row in cells]
assert len(keys) == len(set(keys))
destination = Path("../site/lib/trial-results.json")
temporary = destination.with_suffix(".tmp")
temporary.write_text(json.dumps({**primary, "cells": cells},
                                indent=2, ensure_ascii=False) + "\n")
temporary.replace(destination)
PY
```

3. Validate the W&B export with the explicit list of completed configurations.

```bash
python3 baselines/trial_wandb.py \
  --results "$HPML_LOCAL_TRIAL" \
  --configurations ar-sglang eagle3 dflash ar-vllm-v1 peagle \
    team-ffn-100 team-ffn-75 team-ffn-50 team-ar \
  --allow-unpaired-team \
  --catalogue-export "$HPML_EXPORT" \
  --evidence-version "$HPML_EXPORT_STAMP" \
  --catalogue-only
```

The explicit unpaired option preserves the original 100% observations while requiring compatible references for every paired cell. The command above validates the selected configurations. Add `--upload` for the publication step. Each changed evidence set uses a new evidence version. Existing raw configuration runs retain their original evidence.

4. Before publishing the Site, run its tests and build, then inspect the measured tables, prompt recordings, and report links.

Site publication and W&B upload are separate actions. The local preparation commands make their exact contents reviewable.

## Compile and share the report

1. Create an internal temporary directory and compile the authored paper.

```bash
export HPML_REPORT_BUILD="$(mktemp -d /private/tmp/hpml-report.XXXXXX)"
tectonic --only-cached --untrusted \
  --outdir "$HPML_REPORT_BUILD" measurements/report/report.tex
```

2. Inspect the PDF's results pages, main tables, changed appendix tables, and page boundaries.
3. After checking the rendering, copy the PDF into the report directory.

```bash
cp "$HPML_REPORT_BUILD/report.pdf" measurements/report/report.pdf
```

4. Expand the five generated inputs into the standalone Overleaf source.

```bash
python3 - <<'PY'
import re
from pathlib import Path

report = Path("measurements/report")
allowed = {"tables.tex", "appendix-tables.tex", "metric-definitions.tex",
           "evidence-availability.tex", "results-summary.tex"}
def expand(match):
    name = match.group(1)
    if name not in allowed:
        raise ValueError("Unexpected generated input: " + name)
    return (report / "generated" / name).read_text().rstrip("\n")
text = re.sub(r"\\input\{generated/([^}]+)\}",
              expand, (report / "report.tex").read_text())
target = Path("/private/tmp/hpml-overleaf-upload/benchmark_report.tex")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(text)
PY
```

5. Compile that standalone file and confirm that its text, tables, and page layout match the reviewed report.

```bash
tectonic --only-cached --untrusted --outdir "$HPML_REPORT_BUILD" \
  /private/tmp/hpml-overleaf-upload/benchmark_report.tex
```

6. Copy the reviewed PDF and standalone source into `../site/public/reports`, and update the download links' checksum query values.
7. Upload `benchmark_report.tex` to the existing Overleaf project, select it as the main file, and confirm the compiled result.
8. After publication checks pass, remove the internal compilation directory and disposable preview images.

Keep the report source, generated fragments, reviewed PDF, published standalone source, original proposal, and evidence used by the published results. Preserve earlier immutable exports while published versions depend on them.
