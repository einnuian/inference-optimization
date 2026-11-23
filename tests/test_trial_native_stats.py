import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_native_stats as native


def exposition(rounds=10, proposed=40, accepted=20, success=3, suffix="_total", position=True):
    labels = '{engine="0",model_name="hpml-target"}'
    lines = ["vllm:spec_decode_num_drafts%s%s %d" % (suffix, labels, rounds),
             "vllm:spec_decode_num_draft_tokens%s%s %d" % (suffix, labels, proposed),
             "vllm:spec_decode_num_accepted_tokens%s%s %d" % (suffix, labels, accepted),
             'vllm:request_success_total{engine="0",model_name="hpml-target",finished_reason="stop"} %d' % success,
             "process_start_time_seconds 1234567"]
    if position:
        lines.append('vllm:spec_decode_num_accepted_tokens_per_pos%s{engine="0",model_name="hpml-target",position="0"} %d'
                     % (suffix, accepted))
    return "\n".join(lines) + "\n"


def observation(raw, when=1, gpu="GPU-example"):
    return {"schema_version": 1, "metrics_url": "http://127.0.0.1:31000/metrics", "gpu_uuid": gpu,
            "observed_monotonic_ns": when, "raw_prometheus": raw,
            "samples": native.parse_prometheus(raw)}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def monotonic_ns(self):
        return round(self.now * 1e9)

    def time_ns(self):
        return 1700000000000000000 + self.monotonic_ns()

    def sleep(self, seconds):
        self.now += seconds


class PrometheusTests(unittest.TestCase):
    def test_labels_exemplars_and_raw_evidence(self):
        raw = '# HELP sample Example\nsample{model="a,b\\\"c",path="a\\\\b",line="a\\nb"} 1.5e2 123 # {trace_id="abc"} 1\n# EOF\n'
        sample = native.parse_prometheus(raw)[0]
        self.assertEqual(sample["labels"], {"model": 'a,b"c', "path": "a\\b", "line": "a\nb"})
        self.assertEqual(sample["value"], 150)
        self.assertEqual(sample["timestamp_text"], "123")
        self.assertEqual(sample["raw"], raw.splitlines()[1])
        self.assertIn("trace_id", sample["exemplar_text"])

    def test_nonfinite_unrelated_samples_remain_json_safe(self):
        samples = native.parse_prometheus('unrelated NaN\nanother +Inf\n')
        self.assertEqual(samples[0]["value"], "NaN")
        json.dumps(samples, allow_nan=False)

    def test_duplicate_and_malformed_samples_are_rejected(self):
        for raw in ('x{a="1",a="2"} 1', 'x{a="1} 1', 'x{a="a\\t"} 1', 'x 1\nx 2', 'x 1_0'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                native.parse_prometheus(raw)

    def test_only_loopback_metrics_urls_are_accepted(self):
        self.assertEqual(native._metrics_url("http://localhost:31000/v1"), "http://localhost:31000/metrics")
        self.assertEqual(native._metrics_url("http://[::1]:31000"), "http://[::1]:31000/metrics")
        for url in ("https://example.com", "http://localhost.example.com", "file:///metrics",
                    "http://user:password@127.0.0.1", "http://127.0.0.1/v1/completions", "http://127.0.0.1?x=1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                native._metrics_url(url)
        with self.assertRaises(ValueError):
            native._NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://example.com")


class DeltaTests(unittest.TestCase):
    def test_counter_deltas_preserve_position_evidence_and_units(self):
        before = observation(exposition())
        after = observation(exposition(15, 60, 30, 6), when=2)
        result = native.delta(before, after)
        self.assertEqual(result["totals"], {"num_drafts": 5, "num_draft_tokens": 20, "num_accepted_tokens": 10})
        self.assertEqual(result["acceptance_rate"], 0.5)
        self.assertEqual(result["acceptance_rate_unit"], "fraction")
        self.assertEqual(result["accepted_draft_tokens_per_round"], 2)
        self.assertEqual(result["position_counter_deltas"][0]["delta"], 10)
        self.assertEqual(result["position_counter_deltas"][0]["labels"]["position"], "0")
        self.assertNotIn("tau", result)

    def test_no_total_aliases_and_zero_denominators(self):
        raw = exposition(suffix="", position=False)
        result = native.delta(observation(raw), observation(raw, when=2))
        self.assertIsNone(result["acceptance_rate"])
        self.assertIsNone(result["accepted_draft_tokens_per_round"])
        self.assertEqual(result["position_counter_deltas"], [])

    def test_label_series_are_merged_for_the_recorded_gpu(self):
        first = exposition(position=False)
        second = exposition(15, 60, 30, 6, position=False)
        before = observation(first + first.replace('engine="0"', 'engine="1"').replace('process_start_time_seconds 1234567\n', ''))
        after = observation(second + second.replace('engine="0"', 'engine="1"').replace('process_start_time_seconds 1234567\n', ''), when=2)
        result = native.delta(before, after)
        self.assertEqual(result["totals"]["num_drafts"], 10)
        self.assertEqual(len(result["counter_deltas"]), 6)

    def test_missing_series_resets_nonfinite_and_duplicate_aliases_are_rejected(self):
        before = observation(exposition())
        cases = [exposition(position=False), exposition(9, 40, 20),
                 exposition().replace('} 10\n', '} NaN\n', 1),
                 exposition() + 'vllm:spec_decode_num_drafts{engine="0",model_name="hpml-target"} 10\n',
                 exposition().replace('engine="0"', 'engine="1"'),
                 exposition().replace("process_start_time_seconds 1234567", "process_start_time_seconds 1234568")]
        for raw in cases:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                native.delta(before, observation(raw, when=2))

    def test_gpu_identity_and_sample_order_are_required(self):
        before = observation(exposition())
        with self.assertRaises(ValueError):
            native.delta(before, observation(exposition(), when=2, gpu="GPU-other"))
        with self.assertRaises(ValueError):
            native.delta(before, before)
        without_gpu = observation(exposition(), gpu=None)
        without_gpu_after = observation(exposition(15, 60, 30), when=2, gpu=None)
        with self.assertRaises(ValueError):
            native.delta(without_gpu, without_gpu_after)
        with self.assertRaises(ValueError):
            native.delta(before, without_gpu_after)
        self.assertEqual(native.delta(without_gpu, without_gpu_after, gpu_uuid="GPU-recorded")["gpu_uuid"], "GPU-recorded")


class SnapshotTests(unittest.TestCase):
    def fake_time(self, clock):
        return patch.multiple(native.time, monotonic=clock.monotonic, monotonic_ns=clock.monotonic_ns,
                              time_ns=clock.time_ns, sleep=clock.sleep)

    def test_snapshot_waits_for_completion_and_counter_stability(self):
        clock = FakeClock()
        fetched_at = []

        def fetch(url, timeout):
            fetched_at.append(clock.now)
            self.assertEqual(url, "http://127.0.0.1:31000/metrics")
            if clock.now < 6.5:
                return exposition(success=2)
            if clock.now < 7:
                return exposition(success=3)
            return exposition(15, 60, 30, 3)

        with self.fake_time(clock), patch.object(native, "_fetch_metrics", side_effect=fetch):
            result = native.snapshot("http://127.0.0.1:31000", expected_completions=3, gpu_uuid="GPU-example")
        self.assertGreaterEqual(min(fetched_at), 6)
        self.assertGreaterEqual(clock.now, 8)
        self.assertGreaterEqual(result["settling"]["observed_stability_seconds"], 1)
        self.assertEqual(result["observed_completions"], 3)
        self.assertIn("vllm:spec_decode_num_drafts", result["raw_prometheus"])
        json.dumps(result, allow_nan=False)

    def test_extra_requests_are_rejected(self):
        clock = FakeClock()
        with self.fake_time(clock), patch.object(native, "_fetch_metrics", return_value=exposition(success=4)):
            with self.assertRaises(ValueError):
                native.snapshot("http://localhost:31000", expected_completions=3, settle_seconds=0)

    def test_timeout_preserves_the_last_observation(self):
        clock = FakeClock()
        with self.fake_time(clock), patch.object(native, "_fetch_metrics", return_value=exposition(success=2)):
            with self.assertRaises(native.SnapshotTimeout) as caught:
                native.snapshot("http://localhost:31000", expected_completions=3, timeout=2, settle_seconds=0)
        self.assertEqual(caught.exception.last_observation["observed_completions"], 2)
        self.assertIn("raw_prometheus", caught.exception.last_observation)


if __name__ == "__main__":
    unittest.main()
