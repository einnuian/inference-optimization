import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from trial_view import BENCHMARKS, TrialStore, page_bytes


class TrialViewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prepared = self.root / "prepared.json"
        self.prompts = [{"id": name + "/" + str(index), "benchmark": name,
                         "messages": [{"role": "user", "content": "Read this exact prompt."}],
                         "input_token_ids": [1, 2, 3]}
                        for name in BENCHMARKS for index in range(3)]
        self.prepared.write_text(json.dumps({"prompts": self.prompts}))
        self.results = self.root / "results"
        self.directory = self.results / "eagle3"
        self.directory.mkdir(parents=True)
        self.store = TrialStore(self.prepared, self.results, refresh_seconds=0)

    def event(self, kind, identifier="mt-bench/0", repetition=1):
        return {"event": kind, "configuration": "eagle3", "prompt_id": identifier,
                "benchmark": "mt-bench", "repetition": repetition, "request_id": "request-1",
                "unix_ns": 1_000_000 if kind == "submitted" else 2_000_000,
                "output_tokens": 0, "ttft_seconds": None, "end_to_end_seconds": 0.1}

    def block(self):
        return {"benchmark": "mt-bench", "repetition": 1,
                "requests": [{"prompt_id": "mt-bench/" + str(index), "request_id": "request-%d" % (index + 1),
                              "output_text": "Response %d." % index, "output_tokens": 3,
                              "ttft_seconds": 0.02, "end_to_end_seconds": 0.1, "output_rate": 30,
                              "finish_reason": "stop"} for index in range(3)]}

    def test_partial_progress_and_result_files_eventually_reconcile(self):
        path = self.directory / "live.jsonl"
        encoded = json.dumps(self.event("submitted"))
        path.write_text(encoded[:25])
        self.assertEqual(self.store.snapshot(0)["rows"][3]["status"], "waiting")
        with path.open("a") as handle:
            handle.write(encoded[25:] + "\n")
        state = self.store.snapshot(0)
        self.assertEqual(state["rows"][3]["status"], "submitted")
        self.assertEqual(state["current"]["prompt_id"], "mt-bench/0")
        with path.open("a") as handle:
            handle.write(json.dumps(self.event("completed")) + "\n")
        state = self.store.snapshot(0)
        self.assertEqual(state["completed_requests"], 1)
        self.assertEqual(state["saved_requests"], 0)
        self.assertEqual(state["rows"][3]["output_tokens"], 0)
        saved = self.directory / "01-mt-bench.json"
        saved.write_text('{"benchmark":')
        self.assertEqual(self.store.snapshot()["saved_requests"], 0)
        saved.write_text(json.dumps(self.block()))
        state = self.store.snapshot(0)
        self.assertEqual(state["completed_requests"], 3)
        self.assertEqual(state["saved_requests"], 3)
        self.assertEqual(state["rows"][3]["status"], "saved")
        self.assertEqual(state["rows"][3]["output_text"], "Response 0.")
        self.assertIsNone(state["current"])

    def test_saved_warmup_and_progress_count_once(self):
        path = self.directory / "live.jsonl"
        path.write_text(json.dumps(self.event("submitted", "warmup-1", 0)) + "\n")
        (self.directory / "warmup.json").write_text(json.dumps([{"prompt_id": "warmup-1"}]))
        state = self.store.snapshot()
        self.assertEqual(state["warmup_completed"], 1)
        self.assertEqual(state["completed_requests"], 0)
        self.assertIsNone(state["current"])
        with path.open("a") as handle:
            handle.write(json.dumps(self.event("completed", "warmup-1", 0)) + "\n")
        self.assertEqual(self.store.snapshot()["warmup_completed"], 1)

    def test_mislabelled_blocks_and_external_symlinks_are_ignored(self):
        block = self.block()
        block["requests"][1]["prompt_id"] = "alpaca/0"
        path = self.directory / "01-mt-bench.json"
        path.write_text(json.dumps(block))
        self.assertEqual(self.store.snapshot()["saved_requests"], 0)
        path.unlink()
        outside = self.root / "outside.json"
        outside.write_text(json.dumps(self.block()))
        path.symlink_to(outside)
        self.assertEqual(self.store.snapshot()["saved_requests"], 0)

    def test_prompt_content_cannot_end_the_embedded_json_script(self):
        payload = '</script><script>alert("example")</script>'
        prompts = self.store.prompts
        prompts[0]["text"] = payload
        page = page_bytes(prompts, 0, "test-nonce").decode()
        self.assertNotIn(payload, page)
        self.assertIn(r"\u003c/script\u003e", page)
        self.assertIn('nonce="test-nonce"', page)


if __name__ == "__main__":
    unittest.main()
