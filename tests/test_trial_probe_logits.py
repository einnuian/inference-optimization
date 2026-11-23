"""Check the score probe's settings and interpretation gates without GPU execution."""

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import socket
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_probe_logits as probe


def output():
    return {"prompt_id": probe.PROMPT_ID, "input_token_ids": [1, 2],
            "native_output_token_ids": [74694, 12958, 198, 475, 99],
            "output_token_ids": [74694, 12958, 198, 475], "output_tokens": 4,
            "output_text": "```python\nimport", "finish_reason": "stop", "native_stop_reason": 99,
            "server_metadata": {"spec_verify_ct": 1, "output_token_logprobs": [None, None, None, [-.3, 475, None]],
                "output_token_ids_logprobs": [None, None, None, [[-.4, 755, None], [-.3, 475, None]]],
                "output_top_logprobs": [None, None, None, [[-.3, 475, None], [-.4, 755, None]]]}}


def fixture(directory):
    configuration = directory / "primary/dflash"
    configuration.mkdir(parents=True)
    prepared = directory / "prepared.json"
    bundle = {"input_sha256": probe.INPUT_SHA256, "target_revision": probe.TARGET_REVISION,
              "temperature": 0, "max_new_tokens": 2048, "stop_token_ids": [99],
              "prompts": [{"id": probe.PROMPT_ID, "input_token_ids": [1, 2],
                           "messages": [{"role": "user", "content": "Exact fixture input."}]}]}
    prepared.write_text(json.dumps(bundle))
    launch = {"configuration": "dflash", "slurm_job_id": "123", "hostname": socket.gethostname(),
              "target_provenance": {"revision": probe.TARGET_REVISION},
              "command": ["python3", "-m", "sglang.launch_server", "--port", "31000",
                          "--speculative-algorithm", "DFLASH", "--dtype", "bfloat16"]}
    server_record = directory / "launch.json"
    server_record.write_text(json.dumps(launch))
    (configuration / "run.json").write_text(json.dumps({"configuration": "dflash", "sglang_incremental": False, "gpu_uuid": "GPU-aaaa",
        "prepared_sha256": hashlib.sha256(prepared.read_bytes()).hexdigest(), "server_settings": launch}))
    for repetition in range(1, 4):
        (configuration / ("%02d-humaneval.json" % repetition)).write_text(json.dumps({"requests": [output()]}))
    return SimpleNamespace(primary_results=directory / "primary", prepared=prepared, configuration="dflash",
                           server_record=server_record, url="http://127.0.0.1:31000")


class ScoreProbeTests(unittest.TestCase):
    def test_request_keeps_the_primary_tokens_settings_and_enabled_method(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            bundle, row, run, launch, expected, endpoint, payload, sources = probe.prepare(args)
            self.assertEqual(endpoint, "/generate")
            self.assertEqual(payload["input_ids"], [1, 2])
            self.assertTrue(payload["stream"])
            self.assertEqual(payload["sampling_params"]["max_new_tokens"], 2048)
            self.assertEqual(payload["sampling_params"]["temperature"], 0)
            self.assertEqual(payload["sampling_params"]["stop_token_ids"], [99])
            self.assertEqual(payload["token_ids_logprob"], [755, 475])
            self.assertEqual(payload["top_logprobs_num"], 5)
            self.assertTrue(payload["return_logprob"])
            self.assertEqual(len(sources), 6)
            launch["command"][launch["command"].index("DFLASH")] = "EAGLE3"
            args.server_record.write_text(json.dumps(launch))
            with self.assertRaisesRegex(ValueError, "primary setting"):
                probe.prepare(args)

    def test_input_or_primary_repeat_changes_block_the_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            path = args.primary_results / "dflash/02-humaneval.json"
            changed = output()
            changed["input_token_ids"] = [1, 3]
            path.write_text(json.dumps({"requests": [changed]}))
            with self.assertRaisesRegex(ValueError, "exact selected input"):
                probe.prepare(args)
            changed = output()
            changed["output_text"] = "Different response."
            path.write_text(json.dumps({"requests": [changed]}))
            with self.assertRaisesRegex(ValueError, "repetitions differ"):
                probe.prepare(args)

    def test_score_interpretation_requires_preserved_output_and_aligned_token_ids(self):
        expected = output()
        observed = copy.deepcopy(expected)
        comparison = probe.compare_output(observed, expected)
        result = probe.inspect_scores(observed, comparison)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["reported_top_token_ids"], [475])
        self.assertAlmostEqual(result["def_minus_import_log_probability"], -.1)
        observed["output_token_ids"][3] = 755
        self.assertFalse(probe.inspect_scores(observed, probe.compare_output(observed, expected))["eligible"])
        observed = copy.deepcopy(expected)
        observed["server_metadata"]["output_token_logprobs"][3][1] = 755
        self.assertFalse(probe.inspect_scores(observed, probe.compare_output(observed, expected))["eligible"])

    def test_missing_or_nonfinite_scores_remain_uninterpreted(self):
        observed = output()
        comparison = probe.compare_output(observed, observed)
        observed["server_metadata"]["output_token_ids_logprobs"][3][0][0] = float("nan")
        self.assertFalse(probe.inspect_scores(observed, comparison)["eligible"])
        observed["server_metadata"] = {}
        self.assertFalse(probe.inspect_scores(observed, comparison)["eligible"])

    def test_speculative_scores_require_observed_verification_rounds(self):
        observed = output()
        comparison = probe.compare_output(observed, observed)
        self.assertTrue(probe.inspect_scores(observed, comparison, "dflash")["eligible"])
        observed["server_metadata"]["spec_verify_ct"] = 0
        self.assertFalse(probe.inspect_scores(observed, comparison, "dflash")["eligible"])

    def test_raw_wire_response_and_ancillary_tokens_are_saved_without_touching_primary_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = fixture(Path(temporary))
            args.output = Path(temporary) / "probe.json"
            prepared = probe.prepare(args)
            before = probe.sha256(args.primary_results / "dflash/01-humaneval.json")
            expected = output()
            payload = {"output_ids": expected["native_output_token_ids"], "text": expected["output_text"],
                       "meta_info": {**expected["server_metadata"], "finish_reason": {"type": "stop", "matched": 99},
                                     "completion_tokens": 5}}
            response = io.BytesIO(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
            response.headers = {"Content-Type": "text/event-stream"}
            opener = mock.Mock()
            opener.open.return_value = response
            with mock.patch("download_checkpoints.require_compute_allocation"), \
                    mock.patch.dict(os.environ, {"SLURM_JOB_ID": "123", "CUDA_VISIBLE_DEVICES": "0"}), \
                    mock.patch.object(probe.subprocess, "check_output", return_value="GPU-aaaa\n"), \
                    mock.patch.object(probe, "build_opener", return_value=opener):
                evidence = probe.run_probe(args, prepared)
            self.assertEqual(evidence["status"], "qualified")
            self.assertEqual(evidence["raw_events"][0]["payload"], payload)
            self.assertIsNone(evidence["raw_events"][-1]["payload"])
            self.assertEqual(evidence["accounting"]["output_tokens"], 4)
            self.assertEqual(evidence["accounting"]["native_completion_tokens"], 5)
            self.assertFalse(evidence["accounting"]["included_in_primary_timing"])
            self.assertEqual(probe.sha256(args.primary_results / "dflash/01-humaneval.json"), before)
            self.assertEqual(json.loads(args.output.read_text())["raw_events"], evidence["raw_events"])


if __name__ == "__main__":
    unittest.main()
