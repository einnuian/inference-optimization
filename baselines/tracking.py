#!/usr/bin/env python3
"""Record selected pretrained-inference evidence in an offline W&B run.

Default execution validates and prints the selected record. --record writes an
actual offline run. Authentication and synchronization use W&B's separate CLI.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

CONFIG_FIELDS = set('method engine engine_version runner target_id target_revision draft_id draft_revision source_revision python torch transformers accelerate cuda_runtime gpu_model gpu_count precision attention_backend dataset dataset_revision prompt_set_sha256 arrival_trace_sha256 arrival_rate concurrency input_length output_cap draft_length cache_capacity context_length repeat slurm_job_id checked_at_utc stop_policy warmup_policy compilation_mode measurement_interval currency billing_source output_adapter target_stage_sha256 speculative_stage_sha256 drafter_load_sha256 container_origin container_sha256 qualification_source_sha256 requested_settings_sha256 server_info_sha256 raw_producer_boundary speculative_num_draft_tokens speculative_steps speculative_topk radix_cache_enabled overlap_schedule_enabled'.split())
CONFIG_FIELDS.update('vllm_source_revision speculators_source_revision vllm_patch_sha256 speculators_patch_sha256 vllm_wheel_sha256 flashinfer_cache_wheel_sha256 speculators requested_attention_backend actual_runner_class producer_class weight_audit_scope ordinary_loaded_model_state_sha256 loaded_model_state_sha256 ordinary_checkpoint_keys_sha256 checkpoint_keys_sha256 load_audit_set_sha256'.split())
CHECK_FIELDS = set('dependency_imports eager_bf16 compiled_bf16 static_cache backbone_keys adapter_equal target_loaded greedy_token_ids stop_behavior returned_length normalized_token_ids normalized_stopping raw_boundary adapted_output native_output drafter_weights native_finish_reasons native_returned_text checkpoint_key_coverage'.split())
METRIC_FIELDS = set('requests_submitted requests_completed requests_failed requests_timed_out requests_cancelled prompts_checked prompts_matched input_tokens cached_input_tokens proposed_draft_tokens accepted_draft_tokens correction_bonus_tokens committed_tokens returned_tokens successful_returned_tokens cycles wall_seconds allocated_gpu_seconds allocated_gpu_seconds_per_million_returned_tokens actual_gpu_charges actual_gpu_charges_per_million_returned_tokens estimated_gpu_charges peak_allocated_gpu_bytes peak_reserved_gpu_bytes peak_observed_gpu_bytes returned_tokens_per_second ttft_seconds streaming_pause_seconds tpot_seconds completion_latency_seconds drafter_seconds verification_seconds other_seconds'.split())
QUANTILE_FIELDS = {'ttft_seconds', 'streaming_pause_seconds', 'tpot_seconds', 'completion_latency_seconds', 'returned_tokens_per_second'}


def fail(message):
    raise SystemExit(message)


def scalar(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, str) and len(value) <= 256 and re.fullmatch(r'[A-Za-z0-9_./:@ +,=()-]*', value):
        return value
    fail('Use finite numbers, booleans or short identifiers in the selected record.')


def validate(data):
    if not isinstance(data, dict) or set(data) - {'stage', 'status', 'config', 'checks', 'metrics'}:
        fail('The record permits stage, status, config, checks and metrics only.')
    if data.get('stage') not in {'setup', 'correctness', 'benchmark'}:
        fail('Set stage to setup, correctness or benchmark.')
    if data.get('status') not in {'passed', 'failed', 'partial'}:
        fail('Set status to passed, failed or partial.')
    result = {'stage': data['stage'], 'status': data['status']}
    for section, allowed in [('config', CONFIG_FIELDS), ('checks', CHECK_FIELDS), ('metrics', METRIC_FIELDS)]:
        values = data.get(section, {})
        if not isinstance(values, dict):
            fail(f'{section} must be an object.')
        result[section] = {}
        for key, value in values.items():
            quantile = re.fullmatch(r'(.+)_(p50|p95|p99|max|mean)', key)
            if key not in allowed and not (section == 'metrics' and quantile and quantile[1] in QUANTILE_FIELDS):
                fail(f'Unsupported {section} field: {key}. Review the schema before adding a field.')
            if section == 'checks' and (not isinstance(value, str) or value not in {'passed', 'failed', 'pending'}):
                fail('Checks use passed, failed or pending.')
            if section == 'metrics' and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                fail('Metrics require finite, nonnegative numbers. Omit unmeasured values.')
            result[section][key] = scalar(value)
    if data['status'] == 'passed' and any(v != 'passed' for v in result['checks'].values()):
        fail('A passed record requires every recorded check to pass.')
    if 'method' not in result['config']:
        fail('Identify the method in config.method.')
    if data['stage'] == 'setup' and result['metrics']:
        fail('Setup records use checks. Performance metrics belong to measured benchmark records.')
    return result


def pard_record(root, kind):
    names = {'setup': 'hpml-setup.json', 'gpu': 'hpml-gpu-check.json', 'draft': 'hpml-draft-load.json'}
    evidence = root / 'envs/pard2-cu124-py310' / names[kind]
    data = json.loads(evidence.read_text())
    cfg = {'method': 'pard2', 'source_revision': data['source_revision'],
           'slurm_job_id': data['slurm_job_id'], 'checked_at_utc': data['checked_at_utc'],
           'cuda_runtime': data['cuda_runtime'], 'python': data['python'].split()[0]}
    for package in ('torch', 'transformers'):
        cfg[package] = data['packages'][package]
    checks = {'dependency_imports': 'passed'}
    if kind == 'gpu':
        cfg['gpu_model'] = data['gpu']
        for src, dst in [('bf16_matmul_32x32', 'eager_bf16'), ('compiled_matmul_32x32', 'compiled_bf16'), ('static_cache', 'static_cache')]:
            checks[dst] = data[src]
    if kind == 'draft':
        cfg.update({'gpu_model': data['gpu'], 'draft_id': 'amd/PARD2-Llama-3.1-8B', 'draft_revision': data['draft_revision'], 'cache_capacity': data['draft_cache_capacity']})
        checks.update({'backbone_keys': 'passed' if data['backbone_tensor_keys'] > 0 else 'failed', 'adapter_equal': 'passed' if data['adapter_equal_after_bf16_cast'] else 'failed'})
    return validate({'stage': 'setup', 'status': 'passed' if all(v == 'passed' for v in checks.values()) else 'failed', 'config': cfg, 'checks': checks}), evidence


def eagle_record(root, kind):
    if kind != 'setup':
        fail('The EAGLE selector currently records its saved setup/import evidence.')
    evidence = root / 'envs/eagle-cu124-py310/hpml-setup.json'
    data = json.loads(evidence.read_text())
    if not data.get('scope', '').startswith('Package and EaModel imports;'):
        fail('The EAGLE setup record needs an explicit completed import scope.')
    cfg = {'method': 'eagle3', 'engine': 'original-eagle',
           'source_revision': data['source_revision'], 'slurm_job_id': data['slurm_job_id'],
           'cuda_runtime': data['cuda_runtime'], 'python': data['python'].split()[0]}
    for package in ('torch', 'transformers', 'accelerate'):
        cfg[package] = data['packages'][package]
    if 'checked_at_utc' in data:
        cfg['checked_at_utc'] = data['checked_at_utc']
    return validate({'stage': 'setup', 'status': 'passed', 'config': cfg,
                     'checks': {'dependency_imports': 'passed'}}), evidence


def output_record(root, directory, method):
    directory = directory.resolve()
    if root.resolve() / 'runs' not in directory.parents:
        fail('Select a completed output-check directory under HPML_ROOT/runs.')
    evidence = directory / 'comparison.json'
    comparison = json.loads(evidence.read_text())
    ordinary_file, speculative_file = directory / 'target.json', directory / f'{method}.json'
    ordinary = json.loads(ordinary_file.read_text())
    actual = json.loads(speculative_file.read_text())
    rows = comparison['checks']
    if not rows or len(rows) != len(actual['prompts']) or len(rows) != len(ordinary['prompts']):
        fail('The completed comparison and both output stages must cover the same prompts.')
    if ordinary['target_revision'] != actual['target_revision']:
        fail('Both stages must identify the same canonical target revision.')
    mark = lambda result: 'passed' if result else 'failed'
    normalized = all(row['normalized_generated_token_ids_match'] and row['normalized_returned_token_ids_match'] for row in rows)
    checks = {'target_loaded': 'passed', 'normalized_token_ids': mark(normalized),
              'normalized_stopping': mark(comparison['normalized_stopping_equality']),
              'raw_boundary': mark(comparison['ordinary_raw_boundary_compliance'] and comparison['speculative_raw_boundary_compliance']),
              'adapted_output': mark(comparison['adapted_output_ready']),
              'native_output': mark(comparison['native_output_ready'])}
    status = 'passed' if comparison['native_output_ready'] else ('partial' if comparison['adapted_output_ready'] else 'failed')
    prompts = [{k: p[k] for k in ('name', 'prompt', 'max_new_tokens')} for p in actual['prompts']]
    engines = {'pard2': 'pard2-custom-transformers', 'eagle3': 'original-eagle'}
    draft_ids = {'pard2': 'amd/PARD2-Llama-3.1-8B', 'eagle3': 'yuhuili/EAGLE3-LLaMA3.1-Instruct-8B'}
    cfg = {'method': method, 'engine': engines[method],
           'target_id': 'meta-llama/Llama-3.1-8B-Instruct', 'target_revision': actual['target_revision'],
           'draft_id': draft_ids[method], 'draft_revision': actual['draft_revision'],
           'source_revision': actual['source_revision'], 'slurm_job_id': actual['slurm_job_id'],
           'torch': actual['torch'], 'cuda_runtime': actual['cuda_runtime'],
           'gpu_model': actual['gpu'], 'gpu_count': 1, 'precision': actual['dtype'],
           'concurrency': 1,
           'output_adapter': comparison['output_adapter'],
           'prompt_set_sha256': hashlib.sha256(json.dumps(prompts,sort_keys=True).encode()).hexdigest(),
           'target_stage_sha256': hashlib.sha256(ordinary_file.read_bytes()).hexdigest(),
           'speculative_stage_sha256': hashlib.sha256(speculative_file.read_bytes()).hexdigest()}
    for key in ('attention_backend', 'cache_capacity', 'draft_length'):
        if key in actual:
            cfg[key] = actual[key]
    if method == 'eagle3':
        load_file = directory / 'drafter-load.json'
        load = json.loads(load_file.read_text())
        loaded = load['drafter_load_checks_pass'] and actual['drafter_load_checks_pass'] and comparison['drafter_load_checks_pass']
        checks['drafter_weights'] = mark(loaded)
        cfg['drafter_load_sha256'] = hashlib.sha256(load_file.read_bytes()).hexdigest()
        if not loaded:
            status = 'failed'
    matched = sum(all(row[k] for k in ('prompt_identity_matches','input_ids_match','output_caps_match','normalized_generated_token_ids_match','normalized_returned_token_ids_match','normalized_stop_reason_matches')) for row in rows)
    return validate({'stage': 'correctness', 'status': status, 'config': cfg, 'checks': checks,
                     'metrics': {'prompts_checked': len(rows), 'prompts_matched': matched}}), evidence


def sglang_record(root, directory, method):
    directory = directory.resolve()
    if root.resolve() / 'runs' not in directory.parents:
        fail('Select a completed SGLang qualification directory under HPML_ROOT/runs.')
    evidence = directory / f'{method}-comparison.json'
    paths = {'ordinary': directory / 'ordinary/output.json',
             'actual': directory / method / 'output.json',
             'requested': directory / method / 'requested-settings.json',
             'server': directory / method / 'server-info.json'}
    comparison = json.loads(evidence.read_text())
    ordinary, actual, requested, server = [json.loads(p.read_text()) for p in paths.values()]
    rows = comparison['checks']
    if not rows or len(rows) != len(ordinary['prompts']) or len(rows) != len(actual['prompts']):
        fail('The SGLang comparison and both stages must cover the same prompts.')
    names = [p['name'] for p in actual['prompts']]
    if len(set(names)) != len(names) or names != [p['name'] for p in ordinary['prompts']] or names != [p['prompt'] for p in rows]:
        fail('SGLang prompt identities and ordering must match the saved comparison.')
    target, draft = actual['artifacts']['target'], actual['artifacts'][method]
    if target != ordinary['artifacts']['target'] or target != requested['artifacts']['target'] or draft != requested['artifacts'][method]:
        fail('SGLang output stages and requested settings must identify the same artifacts.')
    if actual['stage'] != method or requested['stage'] != method or ordinary['stage'] != 'ordinary':
        fail('SGLang evidence must identify the selected method and ordinary control.')
    mark = lambda result: 'passed' if result else 'failed'
    normalized = all(row['normalized_generated_token_ids_match'] and row['normalized_returned_token_ids_match'] for row in rows)
    checks = {'target_loaded': 'passed', 'normalized_token_ids': mark(normalized),
              'normalized_stopping': mark(comparison['normalized_stopping_equality']),
              'raw_boundary': mark(comparison['ordinary_raw_boundary_compliance'] and comparison['speculative_raw_boundary_compliance']),
              'adapted_output': mark(comparison['adapted_output_ready']),
              'native_output': mark(comparison['native_output_ready']),
              'native_finish_reasons': mark(comparison['native_finish_reasons_compliant'] and comparison['native_finish_reasons_match']),
              'native_returned_text': mark(comparison['native_returned_text_matches'])}
    status = 'passed' if all(v == 'passed' for v in checks.values()) else ('partial' if comparison['adapted_output_ready'] else 'failed')
    prompts = [{k: p[k] for k in ('name', 'prompt', 'max_new_tokens')} for p in actual['prompts']]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = {'method': 'eagle3' if method == 'eagle3-sglang' else method,
           'engine': 'sglang', 'engine_version': server['version'],
           'target_id': target['repo_id'], 'target_revision': target['revision'],
           'draft_id': draft['repo_id'], 'draft_revision': draft['revision'],
           'slurm_job_id': requested['slurm_job_id'], 'precision': server['dtype'],
           'attention_backend': server['attention_backend'],
           'gpu_count': server['tp_size'] * server['pp_size'] * server['dp_size'],
           'concurrency': server['max_running_requests'], 'context_length': server['context_length'],
           'speculative_num_draft_tokens': server['speculative_num_draft_tokens'],
           'speculative_steps': server['speculative_num_steps'],
           'speculative_topk': server['speculative_eagle_topk'],
           'radix_cache_enabled': not server['disable_radix_cache'],
           'overlap_schedule_enabled': not server['disable_overlap_schedule'],
           'compilation_mode': 'prefill=' + server['cuda_graph_backend_prefill'] + ',decode=' + server['cuda_graph_backend_decode'],
           'raw_producer_boundary': 'SGLang native generate output_ids after scheduler stopping',
           'output_adapter': comparison['output_adapter'],
           'container_origin': requested['container_origin'],
           'container_sha256': requested['container_sif_sha256_record'].split()[0],
           'qualification_source_sha256': requested['qualification_source_sha256'],
           'prompt_set_sha256': hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
           'target_stage_sha256': sha(paths['ordinary']), 'speculative_stage_sha256': sha(paths['actual']),
           'requested_settings_sha256': sha(paths['requested']), 'server_info_sha256': sha(paths['server'])}
    matched = sum(all(row[k] for k in ('prompt_identity_matches', 'input_ids_match', 'output_caps_match', 'normalized_generated_token_ids_match', 'normalized_returned_token_ids_match', 'normalized_stop_reason_matches')) for row in rows)
    return validate({'stage': 'correctness', 'status': status, 'config': cfg, 'checks': checks,
                     'metrics': {'prompts_checked': len(rows), 'prompts_matched': matched}}), evidence


def hspec_record(root, directory, method):
    directory = directory.resolve()
    if root.resolve() / 'runs' not in directory.parents:
        fail('Select a completed patched-vLLM qualification directory under HPML_ROOT/runs.')
    reference = 'target-v1' if method == 'peagle' else 'target-v2'
    evidence = directory / f'{method}-comparison.json'
    ordinary_path, actual_path = directory / reference / 'output.json', directory / method / 'output.json'
    requested_path = directory / method / 'requested-settings.json'
    ordinary, actual, requested, comparison = [json.loads(p.read_text()) for p in (ordinary_path, actual_path, requested_path, evidence)]
    rows = comparison['checks']
    names = [p['name'] for p in actual['prompts']]
    if (not rows or len(set(names)) != len(names) or names != [p['name'] for p in ordinary['prompts']]
            or names != [p['prompt'] for p in rows]):
        fail('Patched-vLLM output stages and comparison must cover the same ordered prompts.')
    target, draft = requested['target_provenance'], requested['draft_provenance']
    if (ordinary['stage'] != reference or actual['stage'] != method or requested['stage'] != method
            or ordinary['model_runner'] != actual['model_runner']
            or ordinary['target_revision'] != actual['target_revision']
            or actual['target_revision'] != target['revision'] or actual['draft_revision'] != draft['revision']):
        fail('The method, ordinary control, runner and artifact identities must match the saved comparison.')
    ordinary_settings = json.loads((directory / reference / 'requested-settings.json').read_text())
    if ordinary_settings['target_provenance'] != target:
        fail('Both patched-vLLM stages must identify the same target provenance.')
    mark = lambda result: 'passed' if result else 'failed'
    checks = {'target_loaded': 'passed',
              'normalized_token_ids': mark(all(row['normalized_generated_token_ids_match'] and row['normalized_returned_token_ids_match'] for row in rows)),
              'normalized_stopping': mark(comparison['normalized_stopping_equality']),
              'raw_boundary': mark(comparison['ordinary_raw_boundary_compliance'] and comparison['speculative_raw_boundary_compliance']),
              'adapted_output': mark(comparison['adapted_output_ready']),
              'native_output': mark(comparison['native_output_ready']),
              'native_finish_reasons': mark(comparison['native_engine_finish_reasons_compliant'] and comparison['native_engine_finish_reasons_match']),
              'native_returned_text': mark(comparison['native_returned_text_matches']),
              'checkpoint_key_coverage': mark(comparison['checkpoint_input_key_coverage_complete'])}
    status = {0: 'passed', 1: 'failed', 2: 'failed', 3: 'partial'}.get(comparison['exit_code'])
    if status is None:
        fail('The completed comparison requires a recognized qualification exit code.')
    if status == 'partial' and not comparison['adapted_output_ready']:
        fail('A partial output qualification requires a passing adapted-output check.')
    environment, arguments = requested['source_environment_record'], requested['engine_arguments']
    state_path = directory / method / 'loaded-model-state.json'
    workers = json.loads(state_path.read_text())['workers']
    if len(workers) != 1:
        fail('The current patched-vLLM selector requires one recorded worker.')
    worker = workers[0]
    backends = sorted({layer['backend'] for model in worker['models'].values() for layer in model['attention_layers']})
    prompts = [{k: p[k] for k in ('name', 'prompt', 'max_new_tokens')} for p in actual['prompts']]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    audit_files = [p for stage in (reference, method) for p in sorted((directory / stage).glob('load-audit-*.json'))]
    if not all(any(p.parent.name == stage for p in audit_files) for stage in (reference, method)):
        fail('Preserve the actual loading audit files for both completed stages.')
    cfg = {'method': method, 'engine': 'vllm-hspec-patched', 'engine_version': actual['vllm'],
           'runner': actual['model_runner'], 'actual_runner_class': worker['runner_class'],
           'producer_class': worker['producer_class'],
           'target_id': target['repo_id'], 'target_revision': target['revision'],
           'draft_id': draft['repo_id'], 'draft_revision': draft['revision'],
           'source_revision': environment['source_revisions']['HSPEC'],
           'vllm_source_revision': environment['source_revisions']['VLLM'],
           'speculators_source_revision': environment['source_revisions']['SPECULATORS'],
           'vllm_patch_sha256': environment['patch_sha256']['VLLM'],
           'speculators_patch_sha256': environment['patch_sha256']['SPECULATORS'],
           'vllm_wheel_sha256': environment['vllm_wheel_sha256'],
           'flashinfer_cache_wheel_sha256': environment['flashinfer_cache_wheel_sha256'],
           'slurm_job_id': actual['slurm_job_id'], 'python': actual['python'].split()[0],
           'torch': actual['torch'], 'transformers': actual['transformers'],
           'speculators': actual['speculators'], 'cuda_runtime': actual['cuda_runtime'],
           'gpu_model': actual['gpu'], 'gpu_count': arguments['tensor_parallel_size'],
           'precision': actual['dtype'], 'attention_backend': ','.join(backends),
           'requested_attention_backend': actual['requested_attention_backend'],
           'concurrency': arguments['max_num_seqs'], 'context_length': arguments['max_model_len'],
           'speculative_num_draft_tokens': arguments['speculative_config']['num_speculative_tokens'],
           'compilation_mode': 'eager' if arguments['enforce_eager'] else 'engine-configured',
           'raw_producer_boundary': 'vLLM CompletionOutput.token_ids after native scheduler stopping',
           'weight_audit_scope': 'checkpoint input names shapes dtypes and loader completion',
           'output_adapter': comparison['output_adapter'],
           'prompt_set_sha256': hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
           'target_stage_sha256': sha(ordinary_path), 'speculative_stage_sha256': sha(actual_path),
           'requested_settings_sha256': sha(requested_path),
           'ordinary_loaded_model_state_sha256': sha(directory / reference / 'loaded-model-state.json'),
           'loaded_model_state_sha256': sha(state_path),
           'ordinary_checkpoint_keys_sha256': sha(directory / reference / 'checkpoint-keys.json'),
           'checkpoint_keys_sha256': sha(directory / method / 'checkpoint-keys.json'),
           'load_audit_set_sha256': hashlib.sha256(json.dumps(
               {str(p.relative_to(directory)): sha(p) for p in audit_files}, sort_keys=True).encode()).hexdigest()}
    matched = sum(all(row[k] for k in ('prompt_identity_matches', 'input_ids_match', 'output_caps_match', 'normalized_generated_token_ids_match', 'normalized_returned_token_ids_match', 'normalized_stop_reason_matches')) for row in rows)
    return validate({'stage': 'correctness', 'status': status, 'config': cfg, 'checks': checks,
                     'metrics': {'prompts_checked': len(rows), 'prompts_matched': matched}}), evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='Explicit aggregate JSON record; source file stays local.')
    source.add_argument('--pard2-output', type=Path, help='Select correctness fields from one completed PARD-2 output-check directory.')
    source.add_argument('--eagle-output', type=Path, help='Select correctness and drafter-load fields from one completed EAGLE-3 output-check directory.')
    source.add_argument('--sglang-dflash-output', type=Path, help='Select one completed SGLang DFlash correctness comparison.')
    source.add_argument('--sglang-eagle-output', type=Path, help='Select one completed SGLang EAGLE-3 correctness comparison.')
    source.add_argument('--peagle-output', type=Path, help='Select a completed patched-vLLM P-EAGLE comparison with ordinary Model Runner V1.')
    source.add_argument('--hspec-output', type=Path, help='Select a completed patched-vLLM H-Spec comparison with ordinary Model Runner V2.')
    source.add_argument('--dspark-output', type=Path, help='Select a completed patched-vLLM DSpark comparison with ordinary Model Runner V2.')
    source.add_argument('--eagle-check', choices=['setup'], help="Select fields from EAGLE's saved setup/import record.")
    source.add_argument('--pard2-check', choices=['setup', 'gpu', 'draft'], help='Select fields from an actual PARD-2 check record.')
    parser.add_argument('--record', action='store_true', help='Write an offline W&B run after validation.')
    parser.add_argument('--project', default='hpml-pretrained-baselines')
    parser.add_argument('--entity', help='Verified W&B user/team, optional for offline records.')
    parser.add_argument('--name', required=True, help='Short run label.')
    args = parser.parse_args()
    for value in (args.project, args.name, args.entity):
        if value is not None and not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value):
            fail('Project, entity and name use letters, numbers, underscores, dots and hyphens.')
    root = Path(os.environ.get('HPML_ROOT', ''))
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        fail('Set HPML_ROOT to the existing private project directory.')
    if root.stat().st_uid != os.getuid() or stat.S_IMODE(root.stat().st_mode) & 0o077:
        fail('HPML_ROOT must be private and owned by this account.')
    if args.pard2_output:
        record, evidence = output_record(root, args.pard2_output, 'pard2')
    elif args.eagle_output:
        record, evidence = output_record(root, args.eagle_output, 'eagle3')
    elif args.sglang_dflash_output:
        record, evidence = sglang_record(root, args.sglang_dflash_output, 'dflash')
    elif args.sglang_eagle_output:
        record, evidence = sglang_record(root, args.sglang_eagle_output, 'eagle3-sglang')
    elif args.peagle_output:
        record, evidence = hspec_record(root, args.peagle_output, 'peagle')
    elif args.hspec_output:
        record, evidence = hspec_record(root, args.hspec_output, 'hspec')
    elif args.dspark_output:
        record, evidence = hspec_record(root, args.dspark_output, 'dspark')
    elif args.eagle_check:
        record, evidence = eagle_record(root, args.eagle_check)
    elif args.pard2_check:
        record, evidence = pard_record(root, args.pard2_check)
    else:
        evidence = args.input
        record = validate(json.loads(evidence.read_text()))
    record['evidence_sha256'] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    print(json.dumps(record, indent=2))
    if not args.record:
        return
    # Enforce offline mode, independent of a parent's environment settings.
    os.environ.update(WANDB_MODE='offline', WANDB_DISABLE_CODE='true', WANDB_DISABLE_GIT='true', WANDB_CONSOLE='off', WANDB_ERROR_REPORTING='false')
    output = root / 'tracking'
    if output.is_symlink():
        fail('Use an ordinary tracking directory.')
    output.mkdir(mode=0o700, exist_ok=True)
    os.umask(0o077)
    os.chdir(output)
    import wandb
    if wandb.__version__ != '0.30.0':
        fail('Use the separate tracking environment with wandb 0.30.0.')
    settings = wandb.Settings(disable_code=True, disable_git=True, save_code=False,
                             console='off', capture_loggers={}, x_disable_meta=True,
                             x_disable_stats=True, x_disable_machine_info=True, quiet=True)
    with wandb.init(project=args.project, entity=args.entity, name=args.name,
                    job_type=record['stage'], mode='offline', dir=str(output),
                    config={**record['config'], 'evidence_sha256': record['evidence_sha256']},
                    settings=settings) as run:
        run.summary['stage'] = record['stage']
        run.summary['status'] = record['status']
        for name, result in record['checks'].items():
            run.summary[f'checks/{name}'] = result
        if record['metrics']:
            run.log(record['metrics'])
        path = str(Path(run.dir).parent)
    print(f'Offline run saved: {path}')
    print('Review the selected record and verify the destination account before explicit W&B sync.')


if __name__ == '__main__':
    main()
