"""Check asynchronous progress durability and preserved engine metadata."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
from trial_recorder import LiveWriter, ResponseTrace


class BrokenHandle:
    def write(self, text):
        raise OSError("The progress disk is unavailable.")

    def flush(self):
        pass

    def close(self):
        pass


class TrialLiveTests(unittest.TestCase):
    def test_events_preserve_the_submission_snapshot_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "live.jsonl"
            with LiveWriter(path) as writer:
                event = {"event": "submitted", "request_id": "one", "unix_ns": 10}
                writer.put(event)
                event["event"] = "changed"
                writer.put({"event": "completed", "request_id": "one", "unix_ns": 20})
                writer.drain()
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([row["event"] for row in rows], ["submitted", "completed"])
            self.assertFalse(writer.thread.is_alive())
            with self.assertRaises(RuntimeError):
                writer.put({"event": "submitted"})

    def test_writer_failure_reaches_the_caller_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = LiveWriter(Path(directory) / "live.jsonl")
            original = writer.handle
            writer.handle = BrokenHandle()
            try:
                writer.put({"event": "submitted"})
                writer.put({"event": "completed"})
                with self.assertRaisesRegex(RuntimeError, "could not be saved"):
                    writer.drain()
                with self.assertRaisesRegex(RuntimeError, "could not be saved"):
                    writer.close()
                self.assertFalse(writer.thread.is_alive())
            finally:
                original.close()

    def test_nested_mutable_event_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with LiveWriter(Path(directory) / "live.jsonl") as writer:
                with self.assertRaises(ValueError):
                    writer.put({"mutable": []})

    def test_sglang_metadata_survives_response_collection(self):
        trace = ResponseTrace("sglang", 100, [128009])
        metadata = {"completion_tokens": 2, "spec_verify_ct": 1,
                    "spec_num_correct_drafts": 1,
                    "spec_correct_drafts_histogram": [0, 1],
                    "finish_reason": {"type": "stop", "matched": 128009}}
        trace.accept(120, {"output_ids": [42, 128009], "text": "Answer", "meta_info": metadata})
        trace.accept(130, None)
        result = trace.result()
        self.assertEqual(result["server_metadata"], metadata)
        self.assertEqual(result["events"][0]["meta_info"], metadata)
        self.assertEqual(result["output_token_ids"], [42])


if __name__ == "__main__":
    unittest.main()
