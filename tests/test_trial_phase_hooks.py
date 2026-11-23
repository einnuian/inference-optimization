import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_phase_hooks as hooks


class FakeEvent:
    def __init__(self, cuda, enable_timing):
        if not enable_timing:
            raise AssertionError("Diagnostic events need timing enabled.")
        self.cuda = cuda
        self.synced = False

    def record(self, stream):
        self.cuda.clock += 0.002
        self.timestamp = self.cuda.clock
        self.cuda.operations.append(("record", stream.cuda_stream))

    def synchronize(self):
        self.synced = True
        self.cuda.operations.append(("event_sync", self.timestamp))

    def elapsed_time(self, end):
        if self.cuda.fail_elapsed:
            raise RuntimeError("The fake CUDA event reports an error.")
        if not end.synced:
            raise AssertionError("CUDA event resolution requires a synchronized endpoint.")
        return (end.timestamp - self.timestamp) * 1000


class FakeCUDA:
    def __init__(self):
        self.clock = 0.0
        self.stream = SimpleNamespace(cuda_stream=7)
        self.operations = []
        self.resets = 0
        self.fail_elapsed = False

    def is_available(self):
        return True

    def current_device(self):
        return 0

    def current_stream(self, device):
        return self.stream

    def Event(self, enable_timing):
        return FakeEvent(self, enable_timing)

    def synchronize(self, device):
        self.operations.append(("device_sync", device))

    def reset_peak_memory_stats(self, device):
        self.resets += 1

    def memory_allocated(self, device):
        return 128

    def memory_reserved(self, device):
        return 256

    def max_memory_allocated(self, device):
        return 192

    def max_memory_reserved(self, device):
        return 384


class PhaseHooksTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.cuda = FakeCUDA()
        self.environment = patch.dict(os.environ, {"HPML_DIAGNOSTIC_DIR": self.directory.name,
            "HPML_CONFIGURATION": "test-configuration",
            "HPML_DIAGNOSTIC_EAGLE_TREE": "0",
                                                  "SLURM_JOB_ID": "123", "HPML_GPU_UUID": "GPU-test"})
        self.environment.start()
        self.torch_patch = patch.object(hooks, "_TORCH", SimpleNamespace(cuda=self.cuda))
        self.torch_patch.start()
        self.local_patch = patch.object(hooks, "_LOCAL", threading.local())
        self.local_patch.start()
        hooks._MEMORY_SCOPES.clear()
        hooks._ACTIVE_DEVICES.clear()

    def tearDown(self):
        hooks.drain_events()
        import trial_round_hooks
        trial_round_hooks._close_events()
        trial_round_hooks._HANDLES.clear()
        hooks._MEMORY_SCOPES.clear()
        hooks._ACTIVE_DEVICES.clear()
        self.local_patch.stop()
        self.torch_patch.stop()
        self.environment.stop()
        self.directory.cleanup()

    def records(self):
        hooks.drain_events()
        return [json.loads(line) for path in Path(self.directory.name).glob("phase-*.jsonl")
                for line in path.read_text().splitlines()]

    def test_disabled_helpers_preserve_methods_and_avoid_torch_import(self):
        class Worker:
            def forward_batch_generation(self, batch):
                return batch

        original = Worker.forward_batch_generation
        with patch.dict(os.environ, {"HPML_DIAGNOSTIC_DIR": ""}), patch.object(hooks, "_TORCH", None), \
                patch.object(importlib, "import_module", side_effect=AssertionError("Unexpected import")):
            self.assertIsNone(hooks.begin_round(["one"], "decode"))
            self.assertIsNone(hooks.begin_phase("draft"))
            self.assertIsNone(hooks.end_phase("draft"))
            self.assertIsNone(hooks.end_round())
            self.assertIsNone(hooks.reset_memory())
            self.assertIsNone(hooks.read_memory())
            self.assertFalse(hooks.install_sglang_worker(Worker))
            self.assertIs(Worker.forward_batch_generation, original)

    def test_phases_defer_synchronization_and_preserve_raw_scope(self):
        hooks.begin_round(["request-1"], "speculative", phase="worker_iteration", tokens={"request-1": 4})
        self.cuda.operations.clear()
        hooks.begin_phase("target_execution")
        hooks.begin_phase("draft_proposal")
        hooks.end_phase("draft_proposal")
        hooks.end_phase("target_execution")
        self.assertFalse(any(operation[0].endswith("sync") for operation in self.cuda.operations))
        record = hooks.end_round()
        self.assertEqual(record["request_ids"], ["request-1"])
        self.assertEqual(record["tokens"], {"request-1": 4})
        self.assertEqual(record["configuration"], "test-configuration")
        self.assertEqual(record["slurm_job_id"], "123")
        self.assertEqual(record["phases"][1]["parent_sequence"], record["phases"][0]["sequence"])
        for phase in [record["outer"]] + record["phases"]:
            self.assertGreaterEqual(phase["gpu_elapsed_seconds"], 0)
            self.assertGreaterEqual(phase["cpu_end_ns"], phase["cpu_start_ns"])
            self.assertEqual(phase["stream_id"], 7)
        self.assertEqual(record["memory"]["max_memory_allocated_bytes"], 192)
        self.assertEqual(len(self.records()), 1)

    def test_nested_rounds_and_out_of_order_phase_closure_are_rejected(self):
        hooks.begin_round(["one"], "decode")
        with self.assertRaises(RuntimeError):
            hooks.begin_round(["two"], "decode")
        hooks.begin_phase("outer")
        hooks.begin_phase("inner")
        with self.assertRaises(RuntimeError):
            hooks.end_phase("outer")
        hooks.end_phase("inner")
        hooks.end_phase("outer")
        hooks.end_round()

    def test_memory_resets_once_for_each_new_request_scope(self):
        for identifier in ("one", "one", "two"):
            hooks.begin_round([identifier], "decode")
            hooks.end_round()
        self.assertEqual(self.cuda.resets, 2)
        records = self.records()
        self.assertEqual(records[0]["memory"]["scope"]["reset_id"], records[1]["memory"]["scope"]["reset_id"])
        self.assertNotEqual(records[1]["memory"]["scope"]["reset_id"], records[2]["memory"]["scope"]["reset_id"])
        self.assertEqual(records[2]["memory"]["scope"]["request_ids"], ["two"])

    def test_vllm_phase_and_memory_ids_match_the_client_with_native_ids_preserved(self):
        client_id = "hpml-" + "b" * 32
        native_id = "cmpl-" + client_id + "-0"
        hooks.begin_round([native_id], "speculative", tokens={native_id: 8})
        record = hooks.end_round()
        self.assertEqual(record["request_ids"], [client_id])
        self.assertEqual(record["native_request_ids"], [native_id])
        self.assertEqual(record["memory"]["scope"]["request_ids"], [client_id])
        self.assertEqual(record["tokens"], {native_id: 8})

    def test_timing_errors_fail_and_release_the_context(self):
        hooks.begin_round(["one"], "decode")
        self.cuda.fail_elapsed = True
        with self.assertRaises(RuntimeError):
            hooks.end_round()
        self.assertIsNone(hooks._active())
        self.assertEqual(hooks._ACTIVE_DEVICES, {})
        self.assertEqual(self.records()[0]["event"], "diagnostic_capture_error")
        self.cuda.fail_elapsed = False
        hooks.begin_round(["two"], "decode")
        hooks.end_round()

    def test_sglang_wrappers_preserve_results_and_capture_named_phases(self):
        class Draft:
            def draft(self, value):
                return value + 1

            def _draft_extend_for_decode(self, value):
                return value + 2

        class Verify:
            def verify(self, value):
                return value + 3

        class Worker:
            def forward_batch_generation(self, batch, value=0):
                return Verify().verify(Draft()._draft_extend_for_decode(Draft().draft(value)))

        mode = SimpleNamespace(is_extend=lambda: True, is_idle=lambda: False)
        batch = SimpleNamespace(reqs=[SimpleNamespace(rid="sg-1")], forward_mode=mode,
                                input_ids=SimpleNamespace(numel=lambda: 5))
        hooks.install_sglang_eagle_draft(Draft)
        hooks.install_sglang_eagle_verify(Verify)
        self.assertTrue(hooks.install_sglang_worker(Worker))
        self.assertFalse(hooks.install_sglang_worker(Worker))
        self.assertEqual(Worker().forward_batch_generation(batch=batch, value=4), 10)
        record = self.records()[0]
        self.assertEqual(record["kind"], "prefill")
        self.assertEqual(record["request_ids"], ["sg-1"])
        self.assertEqual([phase["name"] for phase in record["phases"]],
                         ["draft_proposal", "draft_extension", "target_verification"])
        mode.is_idle = lambda: True
        self.assertEqual(Worker().forward_batch_generation(batch, value=4), 10)
        self.assertEqual(len(self.records()), 1)

    def test_vllm_v2_preserves_the_context_until_sampling(self):
        class Speculator:
            def propose(self, value):
                return value + "-draft"

        class Runner:
            def __init__(self):
                self.speculator = Speculator()

            def execute_model(self, scheduler_output):
                return None

            def sample_tokens(self, value):
                return self.speculator.propose(value)

        hooks.install_vllm_runner(Runner)
        runner = Runner()
        scheduler = SimpleNamespace(num_scheduled_tokens={"v2-1": 3}, scheduled_spec_decode_tokens={"v2-1": [4, 5]})
        self.assertIsNone(runner.execute_model(scheduler_output=scheduler))
        self.assertIsNotNone(hooks._active())
        self.assertEqual(self.records(), [])
        self.assertEqual(runner.sample_tokens("answer"), "answer-draft")
        record = self.records()[0]
        self.assertEqual(record["kind"], "speculative")
        self.assertEqual([phase["name"] for phase in record["phases"]], ["target_execution", "draft_proposal"])
        self.assertIsNone(hooks._active())

    def test_sglang_decode_links_native_round_and_device_completion(self):
        class Worker:
            def forward_batch_generation(self, batch):
                hooks.begin_phase("draft")
                hooks.end_phase("draft")
                hooks.begin_phase("verify")
                hooks.end_phase("verify")
                return "generated"

        request = SimpleNamespace(rid="sg-round", spec_verify_ct=4,
                                  is_retracted=False, finished=lambda: False)
        batch = SimpleNamespace(reqs=[request], forward_mode=SimpleNamespace(is_extend=lambda: False))
        with patch.dict(os.environ, {"HPML_CONFIGURATION": "dflash"}):
            hooks.install_sglang_worker(Worker)
            self.assertEqual(Worker().forward_batch_generation(batch), "generated")
        row = self.records()[0]
        self.assertEqual(row["verification_round_indices"], {"sg-round": 5})
        self.assertEqual(request.spec_verify_ct, 4)
        self.assertEqual(row["clock"], "time.perf_counter_ns")
        self.assertGreaterEqual(row["synchronized_cpu_end_ns"], row["outer"]["cpu_end_ns"])
        self.assertTrue(row["memory"]["device_synchronized_before_read"])
        self.assertIn(("device_sync", 0), self.cuda.operations)

    def test_vllm_v1_bound_draft_and_immediate_output_are_supported(self):
        class Runner:
            def propose_draft_token_ids(self, value):
                return value + 1

            def execute_model(self, scheduler_output, value=0):
                return self.propose_draft_token_ids(value)

            def sample_tokens(self):
                return "unused"

        hooks.install_vllm_runner(Runner)
        scheduler = SimpleNamespace(num_scheduled_tokens={"v1-1": 1}, scheduled_spec_decode_tokens={})
        self.assertEqual(Runner().execute_model(scheduler, value=9), 10)
        record = self.records()[0]
        self.assertEqual(record["kind"], "prefill_or_ar")
        self.assertEqual(record["phases"][1]["name"], "draft_proposal")
        self.assertEqual(record["phases"][1]["parent_sequence"], record["phases"][0]["sequence"])
        self.assertIsNone(hooks._active())

    def peagle_runner(self):
        class Runner:
            def __init__(self):
                self.input_batch = SimpleNamespace(req_ids=["p-one"], prev_sampled_token_ids=None)
                self.prev_positions = SimpleNamespace(np=[0])
                self._draft_token_ids = None
                self.next_draft = [[10, 11]]

            def _prepare_input_ids(self, scheduler_output, num_reqs):
                return "prepared"

            def propose_draft_token_ids(self):
                return self.next_draft

            def execute_model(self, scheduler_output):
                self._prepare_input_ids(scheduler_output, 1)
                return None

            def sample_tokens(self):
                self._draft_token_ids = self.propose_draft_token_ids()
                self.input_batch.prev_sampled_token_ids = [[9]]
                return "sampled"

        hooks.install_vllm_runner(Runner)
        return Runner()

    def test_peagle_links_the_exact_producer_and_later_verification(self):
        with patch.dict(os.environ, {"HPML_CONFIGURATION": "peagle"}):
            runner = self.peagle_runner()
            initial = SimpleNamespace(num_scheduled_tokens={"p-one": 8},
                scheduled_spec_decode_tokens={}, hpml_diagnostic_step_id="step-prefill")
            runner.execute_model(initial)
            runner.sample_tokens()
            verified = SimpleNamespace(num_scheduled_tokens={"p-one": 3},
                scheduled_spec_decode_tokens={"p-one": [-1, -1]}, hpml_diagnostic_step_id="step-verify")
            runner.next_draft = [[20, 21]]
            runner.execute_model(verified)
            runner.sample_tokens()
        producer, consumer = self.records()
        link = consumer["verification_proposals"]["p-one"]
        self.assertEqual(link["producer_observation_id"], producer["observation_id"])
        self.assertEqual(link["producer_scheduler_step_id"], "step-prefill")
        self.assertEqual(link["token_ids"], [10, 11])
        self.assertTrue(link["tensor_identity_verified"])
        self.assertEqual(consumer["produced_drafts"]["p-one"]["token_ids"], [20, 21])
        self.assertEqual(consumer["scheduler_step_id"], "step-verify")
        verification = next(phase for phase in consumer["phases"] if phase["name"] == "verification_and_sampling")
        draft = next(phase for phase in consumer["phases"] if phase["name"] == "draft_proposal")
        self.assertLessEqual(verification["cpu_end_ns"], draft["cpu_start_ns"])
        self.assertGreaterEqual(consumer["synchronized_cpu_end_ns"], verification["cpu_end_ns"])

    def test_peagle_rejects_a_draft_tensor_without_its_producer(self):
        with patch.dict(os.environ, {"HPML_CONFIGURATION": "peagle"}):
            runner = self.peagle_runner()
            initial = SimpleNamespace(num_scheduled_tokens={"p-one": 8},
                scheduled_spec_decode_tokens={}, hpml_diagnostic_step_id="step-prefill")
            runner.execute_model(initial)
            runner.sample_tokens()
            runner._draft_token_ids = [[10, 11]]
            verified = SimpleNamespace(num_scheduled_tokens={"p-one": 3},
                scheduled_spec_decode_tokens={"p-one": [-1, -1]}, hpml_diagnostic_step_id="step-verify")
            with self.assertRaisesRegex(RuntimeError, "recorded draft tensor"):
                runner.execute_model(verified)
        self.assertIsNone(hooks._active())
        self.assertEqual(self.records()[-1]["event"], "diagnostic_capture_error")

    def test_peagle_rejects_mismatched_previous_request_rows(self):
        with patch.dict(os.environ, {"HPML_CONFIGURATION": "peagle"}):
            runner = self.peagle_runner()
            initial = SimpleNamespace(num_scheduled_tokens={"p-one": 8},
                scheduled_spec_decode_tokens={}, hpml_diagnostic_step_id="step-prefill")
            runner.execute_model(initial)
            runner.sample_tokens()
            runner.prev_positions.np = [-1]
            verified = SimpleNamespace(num_scheduled_tokens={"p-one": 3},
                scheduled_spec_decode_tokens={"p-one": [-1, -1]}, hpml_diagnostic_step_id="step-verify")
            with self.assertRaisesRegex(RuntimeError, "previous draft row"):
                runner.execute_model(verified)


if __name__ == "__main__":
    unittest.main()
