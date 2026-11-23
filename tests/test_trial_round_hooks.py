"""Exercise diagnostic round capture with small fake engines."""

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_round_hooks as hooks


class SGRequest:
    def __init__(self, request_id, finished=False, retracted=False):
        self.rid = request_id
        self.output_ids = [7]
        self.spec_verify_ct = 0
        self.is_retracted = retracted
        self._finished = finished

    def finished(self):
        return self._finished


class VLLMRequest:
    def __init__(self, finished=False, stale=False, keep=None, remove=False):
        self._output_token_ids = [7]
        self.async_tokens_to_discard = 1 if stale else 0
        self._finished = finished
        self.keep = keep
        self.remove = remove

    def is_finished(self):
        return self._finished


def sglang_processor_class():
    class Processor:
        def _resolve_spec_v2_tokens(self, result, batch):
            for request in batch.reqs:
                if not request.is_retracted and not request.finished():
                    request.spec_verify_ct += 1
            return result.predictions

    return Processor


def vllm_scheduler_class():
    class Scheduler:
        def __init__(self, requests):
            self.requests = requests
            self.num_sampled_tokens_per_step = 1

        def update_from_output(self, scheduler_output, model_runner_output):
            for request_id in scheduler_output.num_scheduled_tokens:
                request = self.requests.get(request_id)
                if request is None or request.is_finished() or request.async_tokens_to_discard:
                    continue
                index = model_runner_output.req_id_to_index[request_id]
                generated = model_runner_output.sampled_token_ids[index]
                if request.keep is not None:
                    del generated[request.keep:]
                request._output_token_ids.extend(generated)
                if request.remove:
                    del self.requests[request_id]
            return {"updated": True}

    return Scheduler


def vllm_outputs(proposals, generated, invalid=None):
    request_ids = list(proposals)
    scheduled = SimpleNamespace(
        num_scheduled_tokens={request_id: len(tokens) + 1 for request_id, tokens in proposals.items()},
        scheduled_spec_decode_tokens=proposals,
        num_invalid_spec_tokens=invalid,
    )
    runner = SimpleNamespace(
        req_id_to_index={request_id: index for index, request_id in enumerate(request_ids)},
        sampled_token_ids=[list(generated[request_id]) for request_id in request_ids],
    )
    return scheduled, runner


class TrialRoundHookTests(unittest.TestCase):
    def setUp(self):
        self._close_handles()
        hooks._ROUNDS.clear()
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.environment = mock.patch.dict(os.environ, {
            "HPML_DIAGNOSTIC_DIR": str(self.directory),
            "HPML_CONFIGURATION": "dflash",
        })
        self.environment.start()

    def tearDown(self):
        self._close_handles()
        hooks._ROUNDS.clear()
        self.environment.stop()
        self.temporary.cleanup()

    @staticmethod
    def _close_handles():
        for handle in hooks._HANDLES.values():
            handle.close()
        hooks._HANDLES.clear()

    def rows(self, backend):
        hooks.drain_events()
        rows = []
        for path in self.directory.glob(backend + "-rounds-*.jsonl"):
            rows.extend(json.loads(line) for line in path.read_text().splitlines())
        return rows

    def test_absent_environment_keeps_both_classes_unchanged(self):
        processor = sglang_processor_class()
        scheduler = vllm_scheduler_class()
        sg_original = processor._resolve_spec_v2_tokens
        vllm_original = scheduler.update_from_output
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(hooks.install_sglang_processor(processor))
            self.assertFalse(hooks.install_vllm_scheduler(scheduler))
        self.assertIs(processor._resolve_spec_v2_tokens, sg_original)
        self.assertIs(scheduler.update_from_output, vllm_original)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_sglang_records_success_and_zero_acceptance_rounds(self):
        processor = sglang_processor_class()
        self.assertTrue(hooks.install_sglang_processor(processor))
        wrapped = processor._resolve_spec_v2_tokens
        self.assertTrue(hooks.install_sglang_processor(processor))
        self.assertIs(processor._resolve_spec_v2_tokens, wrapped)
        request = SGRequest("sg-one")
        batch = SimpleNamespace(reqs=[request])
        result = SimpleNamespace(speculative_num_draft_tokens=5,
                                 num_correct_drafts_per_req_cpu=[2],
                                 predictions=[[10, 11, 12]])
        actual = processor()._resolve_spec_v2_tokens(result, batch)
        self.assertIs(actual, result.predictions)
        request.output_ids.extend(actual[0])
        result.num_correct_drafts_per_req_cpu = [0]
        result.predictions = [[13]]
        processor()._resolve_spec_v2_tokens(result, batch)
        rows = self.rows("sglang")
        self.assertEqual([row["round"] for row in rows], [1, 2])
        self.assertEqual([row["native_verification_count"] for row in rows], [1, 2])
        self.assertEqual([row["accepted_draft_tokens"] for row in rows], [2, 0])
        self.assertEqual([row["retained_target_tokens"] for row in rows], [1, 1])
        self.assertEqual(rows[1]["committed_token_ids"], [13])
        self.assertEqual(rows[1]["previous_output_count"], 4)
        self.assertEqual(rows[1]["proposed_depth"], 4)
        self.assertEqual(rows[1]["accepted_prefix_depth"], 0)
        self.assertNotIn("delivered_tokens", rows[1])
        self.assertLessEqual(rows[0]["cpu_call_started_ns"], rows[0]["cpu_call_finished_ns"])

    def test_sglang_preserves_unknown_eagle_tree_depth(self):
        os.environ["HPML_CONFIGURATION"] = "eagle3"
        processor = sglang_processor_class()
        hooks.install_sglang_processor(processor)
        result = SimpleNamespace(speculative_num_draft_tokens=16,
                                 num_correct_drafts_per_req_cpu=[2],
                                 predictions=[[10, 11, 12]])
        processor()._resolve_spec_v2_tokens(result, SimpleNamespace(reqs=[SGRequest("tree")]))
        row = self.rows("sglang")[0]
        self.assertEqual(row["proposed_draft_tokens"], 15)
        self.assertEqual(row["proposal_structure"], "tree")
        self.assertEqual(row["proposed_count_basis"], "candidate_nodes")
        self.assertIsNone(row["proposed_depth"])
        self.assertIsNone(row["accepted_prefix_depth"])
        self.assertFalse(row["depth_metrics_available"])

    def test_sglang_skips_finished_and_retracted_requests(self):
        processor = sglang_processor_class()
        hooks.install_sglang_processor(processor)
        requests = [SGRequest("finished", finished=True), SGRequest("retracted", retracted=True)]
        result = SimpleNamespace(speculative_num_draft_tokens=5,
                                 num_correct_drafts_per_req_cpu=[0, 0], predictions=[[], []])
        processor()._resolve_spec_v2_tokens(result, SimpleNamespace(reqs=requests))
        self.assertEqual([request.spec_verify_ct for request in requests], [0, 0])
        self.assertEqual(self.rows("sglang"), [])

    def test_sglang_emits_only_after_final_state_update(self):
        processor = sglang_processor_class()
        observed = []

        def process_batch_result_decode(instance, batch, result):
            predictions = instance._resolve_spec_v2_tokens(result, batch)
            observed.append(self.rows("sglang"))
            batch.reqs[0].output_ids.extend(predictions[0])
            batch.reqs[0]._finished = True
            batch.reqs[0].finished_len = 2
            return "state-updated"

        processor.process_batch_result_decode = process_batch_result_decode
        hooks.install_sglang_processor(processor)
        request = SGRequest("sg-final")
        result = SimpleNamespace(speculative_num_draft_tokens=5,
            num_correct_drafts_per_req_cpu=[1], predictions=[[10, 99]])
        self.assertEqual(processor().process_batch_result_decode(SimpleNamespace(reqs=[request]), result), "state-updated")
        self.assertEqual(observed, [[]])
        row = self.rows("sglang")[0]
        self.assertTrue(row["state_update_completed"])
        self.assertTrue(row["native_finished"])
        self.assertEqual(row["native_finished_len"], 2)
        self.assertEqual(row["native_postupdate_token_ids"], [10, 99])
        self.assertGreaterEqual(row["state_update_completed_ns"], row["cpu_call_finished_ns"])
        self.assertEqual(row["clock"], "time.perf_counter_ns")
        self.assertEqual(row["native_verification_count"], 1)

    def test_sglang_failed_state_update_keeps_completed_round_absent(self):
        processor = sglang_processor_class()

        def process_batch_result_decode(instance, batch, result):
            instance._resolve_spec_v2_tokens(result, batch)
            raise RuntimeError("The final state update failed.")

        processor.process_batch_result_decode = process_batch_result_decode
        hooks.install_sglang_processor(processor)
        result = SimpleNamespace(speculative_num_draft_tokens=5,
            num_correct_drafts_per_req_cpu=[0], predictions=[[10]])
        with self.assertRaisesRegex(RuntimeError, "state update failed"):
            processor().process_batch_result_decode(SimpleNamespace(reqs=[SGRequest("failed")]), result)
        self.assertEqual(self.rows("sglang"), [])
        self.assertIsNone(hooks._SG_PENDING.rows)

    def test_vllm_preserves_pretrim_tokens_after_request_removal(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        request = VLLMRequest(keep=1, remove=True)
        scheduler = scheduler_class({"one": request})
        scheduled, runner = vllm_outputs({"one": [10, 11, 12]}, {"one": [10, 11, 99]})
        result = scheduler.update_from_output(scheduled, runner)
        self.assertEqual(result, {"updated": True})
        self.assertNotIn("one", scheduler.requests)
        self.assertEqual(runner.sampled_token_ids[0], [10])
        row = self.rows("vllm")[0]
        self.assertEqual(row["committed_token_ids"], [10, 11, 99])
        self.assertEqual(row["committed_tokens"], 3)
        self.assertEqual(row["accepted_draft_tokens"], 2)
        self.assertEqual(row["retained_target_tokens"], 1)
        self.assertEqual(row["native_posttrim_token_ids"], [10])
        self.assertEqual(row["native_posttrim_tokens"], 1)
        self.assertEqual(row["previous_output_count"], 1)

    def test_vllm_schedule_identity_survives_through_final_state_update(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        scheduler_class = vllm_scheduler_class()
        scheduled, runner = vllm_outputs({"one": [10, 11]}, {"one": [10, 99]})
        scheduled.hpml_diagnostic_step_id = None
        scheduler_class.schedule = lambda instance: scheduled
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({"one": VLLMRequest()})
        actual = scheduler.schedule()
        self.assertEqual(len(actual.hpml_diagnostic_step_id), 32)
        scheduler.update_from_output(actual, runner)
        row = self.rows("vllm")[0]
        self.assertEqual(row["scheduler_step_id"], actual.hpml_diagnostic_step_id)
        self.assertTrue(row["state_update_completed"])
        self.assertEqual(row["state_update_completed_ns"], row["cpu_call_finished_ns"])

    def test_vllm_completion_id_matches_the_recorder_and_preserves_engine_identity(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        client_id = "hpml-" + "a" * 32
        native_id = "cmpl-" + client_id + "-0"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({native_id: VLLMRequest()})
        scheduled, runner = vllm_outputs({native_id: [10]}, {native_id: [10, 99]})
        scheduler.update_from_output(scheduled, runner)
        row = self.rows("vllm")[0]
        self.assertEqual(row["request_id"], client_id)
        self.assertEqual(row["native_request_id"], native_id)
        self.assertEqual(hooks.canonical_request_id("cmpl-" + client_id + "-1"), "cmpl-" + client_id + "-1")
        self.assertEqual(hooks.canonical_request_id("cmpl-ordinary-id-0"), "cmpl-ordinary-id-0")

    def test_vllm_subtracts_invalid_proposals_and_preserves_placeholders(self):
        os.environ["HPML_CONFIGURATION"] = "dspark"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({"one": VLLMRequest()})
        scheduled, runner = vllm_outputs({"one": [10, 11, 12, -1]},
                                         {"one": [10, 11, 99]}, {"one": 1})
        scheduler.update_from_output(scheduled, runner)
        row = self.rows("vllm")[0]
        self.assertEqual(row["scheduled_proposal_count"], 4)
        self.assertEqual(row["invalid_proposal_count"], 1)
        self.assertEqual(row["proposed_draft_tokens"], 3)
        self.assertEqual(row["proposed_depth"], 3)
        self.assertEqual(row["accepted_prefix_depth"], 2)
        self.assertEqual(row["scheduled_proposal_token_ids"], [10, 11, 12, -1])
        self.assertTrue(row["scheduled_proposal_contains_placeholders"])

    def test_vllm_skips_stale_and_finished_requests(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({"stale": VLLMRequest(stale=True),
                                     "finished": VLLMRequest(finished=True)})
        scheduled, runner = vllm_outputs({"stale": [10], "finished": [10]},
                                         {"stale": [10, 99], "finished": [10, 99]})
        scheduler.update_from_output(scheduled, runner)
        self.assertEqual(self.rows("vllm"), [])

    def test_vllm_keeps_zero_acceptance_round_with_empty_delivery(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({"one": VLLMRequest(keep=0)})
        scheduled, runner = vllm_outputs({"one": [10, 11, 12]}, {"one": [99]})
        scheduler.update_from_output(scheduled, runner)
        rows = self.rows("vllm")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["accepted_draft_tokens"], 0)
        self.assertEqual(rows[0]["retained_target_tokens"], 1)
        self.assertEqual(rows[0]["committed_token_ids"], [99])
        self.assertEqual(rows[0]["native_posttrim_token_ids"], [])
        self.assertTrue(rows[0]["native_posttrim_empty"])

    def test_vllm_rejects_acceptance_beyond_valid_proposals(self):
        os.environ["HPML_CONFIGURATION"] = "peagle"
        scheduler_class = vllm_scheduler_class()
        hooks.install_vllm_scheduler(scheduler_class)
        scheduler = scheduler_class({"one": VLLMRequest()})
        scheduled, runner = vllm_outputs({"one": [10, -1]}, {"one": [10, 11, 99]}, {"one": 1})
        with self.assertRaisesRegex(ValueError, "acceptance count exceeds"):
            scheduler.update_from_output(scheduled, runner)
        self.assertEqual(self.rows("vllm"), [])


if __name__ == "__main__":
    unittest.main()
