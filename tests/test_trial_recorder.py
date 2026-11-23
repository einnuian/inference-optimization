"""Check timing, stream boundaries, original IDs and diagnostic denominators."""

import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from trial_recorder import ResponseTrace, block_metrics, percentile, sse_events, request_payload
from trial_analysis import round_metrics
from trial_resources import resource_ratios


class StreamingTests(unittest.TestCase):
    def test_sse_metadata_comments_multiline_and_done(self):
        wire = b': ping\r\n\r\ndata: {"choices":\r\ndata: []}\r\n\r\ndata: [DONE]\r\n\r\n'
        times = iter([20, 30])
        self.assertEqual(list(sse_events(io.BytesIO(wire), lambda: next(times))), [(20, {"choices": []}), (30, None)])

    def test_truncated_stream_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "before its"):
            list(sse_events(io.BytesIO(b'data: {}\n\n')))

    def test_sglang_cumulative_ttft_and_tpot(self):
        trace = ResponseTrace("sglang", 1_000_000_000, [99])
        trace.accept(1_100_000_000, {"text": "", "output_ids": [], "meta_info": {}})
        trace.accept(1_200_000_000, {"text": "Hi", "output_ids": [10], "meta_info": {}})
        trace.accept(1_500_000_000, {"text": "Hi there", "output_ids": [10, 11, 12, 99],
                                   "meta_info": {"finish_reason": {"type": "stop", "matched": 99}, "completion_tokens": 4}})
        trace.accept(1_600_000_000, None)
        row = trace.result()
        self.assertEqual(row["output_token_ids"], [10, 11, 12])
        self.assertEqual(row["provider_completion_tokens"], 4)
        self.assertAlmostEqual(row["ttft_seconds"], .2)
        self.assertAlmostEqual(row["end_to_end_seconds"], .6)
        self.assertAlmostEqual(row["tpot_seconds"], .2)
        self.assertEqual(row["inter_chunk_seconds"], [.3])

    def test_vllm_delta_ids_and_usage_only_event(self):
        trace = ResponseTrace("vllm", 0, [99])
        trace.accept(100, {"choices": [{"index": 0, "text": "A", "token_ids": [5]}]})
        trace.accept(200, {"choices": [{"index": 0, "text": "B", "token_ids": [5], "finish_reason": "length"}]})
        trace.accept(250, {"choices": [], "usage": {"completion_tokens": 2}})
        trace.accept(300, None)
        row = trace.result()
        self.assertEqual(row["output_token_ids"], [5, 5])
        self.assertEqual(len(row["inter_chunk_seconds"]), 1)
        self.assertEqual(row["output_text"], "AB")

    def test_one_token_tpot_is_undefined(self):
        trace = ResponseTrace("sglang", 0, [])
        trace.accept(100, {"text": "A", "output_ids": [1], "meta_info": {"finish_reason": {"type": "length"}}})
        trace.accept(200, None)
        self.assertIsNone(trace.result()["tpot_seconds"])
        self.assertIsNone(trace.result()["max_inter_chunk_seconds"])

    def test_incremental_sglang_and_zero_content(self):
        trace = ResponseTrace("sglang", 0, [99], True)
        trace.accept(100, {"text": "", "output_ids": [99], "meta_info": {"finish_reason": {"type": "stop"}}})
        trace.accept(200, None)
        self.assertEqual(trace.result()["output_tokens"], 0)
        self.assertIsNone(trace.result()["ttft_seconds"])

    def test_cumulative_rewrites_are_rejected(self):
        trace = ResponseTrace("sglang", 0, [])
        trace.accept(100, {"text": "A", "output_ids": [1]})
        with self.assertRaisesRegex(ValueError, "rewrote"):
            trace.accept(200, {"text": "B", "output_ids": [2]})

    def test_tokens_after_stop_are_rejected(self):
        trace = ResponseTrace("sglang", 0, [99])
        trace.accept(100, {"text": "AB", "output_ids": [1, 99, 2], "meta_info": {"finish_reason": {"type": "stop"}}})
        trace.accept(200, None)
        with self.assertRaisesRegex(ValueError, "after a canonical"):
            trace.result()

    def test_missing_original_ids_are_rejected(self):
        trace = ResponseTrace("vllm", 0, [])
        with self.assertRaisesRegex(ValueError, "original"):
            trace.accept(100, {"choices": [{"index": 0, "text": "A"}]})

    def test_common_payloads_preserve_input_ids(self):
        for backend in ("sglang", "vllm"):
            endpoint, payload = request_payload(backend, {"input_token_ids": [1, 2]}, {"stop_token_ids": [99]}, "hpml-target", "request-1")
            self.assertTrue(payload["stream"])
            self.assertEqual(payload.get("input_ids", payload.get("prompt")), [1, 2])
            parameters = payload.get("sampling_params", payload)
            self.assertEqual(parameters["temperature"], 0)
            self.assertFalse(parameters["ignore_eos"])


class CalculationTests(unittest.TestCase):
    def test_resource_intervals_and_real_charges(self):
        result = resource_ratios(100, 20, 2)
        self.assertEqual(result["allocated_gpu_seconds_per_million_output_tokens"], 200000)
        self.assertEqual(result["output_tokens_per_gpu_second"], 5)
        self.assertEqual(result["gpu_cost_per_million_output_tokens"], 20000)

    def test_unavailable_charges_stay_unavailable(self):
        self.assertIsNone(resource_ratios(100, 20)["gpu_cost_per_million_output_tokens"])

    def test_zero_output_preserves_undefined_ratios(self):
        result = resource_ratios(0, 20, 2)
        self.assertIsNone(result["allocated_gpu_seconds_per_million_output_tokens"])
        self.assertIsNone(result["gpu_cost_per_million_output_tokens"])
        self.assertEqual(result["output_tokens_per_gpu_second"], 0)

    def test_invalid_allocation_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            resource_ratios(100, 0)

    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(percentile([1, 2, 3], .95), 2.9)
        self.assertIsNone(percentile([], .95))

    def test_otps_uses_whole_three_request_interval(self):
        rows = [{"output_tokens": 10, "end_to_end_seconds": 1, "ttft_seconds": .2,
                 "tpot_seconds": .1, "output_rate": 10, "max_inter_chunk_seconds": .3,
                 "inter_chunk_seconds": [.1, .3]}] * 3
        self.assertEqual(block_metrics(rows, 4)["otps"], 7.5)

    def test_round_denominators_and_bonus_tokens(self):
        rows = [
            {"request_id": "a", "proposed_draft_tokens": 3, "accepted_draft_tokens": 0,
             "retained_target_tokens": 1, "committed_tokens": 1, "delivered_tokens": 1,
             "proposed_depth": 3, "accepted_prefix_depth": 0},
            {"request_id": "a", "proposed_draft_tokens": 3, "accepted_draft_tokens": 2,
             "retained_target_tokens": 1, "committed_tokens": 3, "delivered_tokens": 2,
             "proposed_depth": 3, "accepted_prefix_depth": 2},
            {"request_id": "b", "proposed_draft_tokens": 1, "accepted_draft_tokens": 1,
             "retained_target_tokens": 1, "committed_tokens": 2, "delivered_tokens": 2,
             "proposed_depth": 1, "accepted_prefix_depth": 1}]
        result = round_metrics(rows)
        self.assertAlmostEqual(result["draft_token_acceptance_percent"], 300 / 7)
        self.assertEqual(result["mean_acceptance_length"], 2)
        self.assertEqual(result["accepted_draft_tokens_per_round"], 1)
        self.assertEqual(result["positions"][1]["prefix_acceptance_probability"], .5)
        self.assertEqual(result["positions"][1]["conditional_acceptance_percent"], 100)
        self.assertEqual(result["positions"][2]["conditional_acceptance_percent"], 0)
        self.assertEqual(result["timing"]["draft_seconds"]["n"], 0)

    def test_impossible_round_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            round_metrics([{"request_id": "a", "proposed_draft_tokens": 1, "accepted_draft_tokens": 2,
                            "retained_target_tokens": 1, "committed_tokens": 3, "delivered_tokens": 3,
                            "proposed_depth": 1, "accepted_prefix_depth": 1}])


if __name__ == "__main__":
    unittest.main(verbosity=2)
