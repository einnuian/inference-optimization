import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_round_hooks as hooks


class DiagnosticWriterTests(unittest.TestCase):
    def tearDown(self):
        for writer in hooks._HANDLES.values():
            writer.close()
        hooks._HANDLES.clear()

    def test_disk_delay_stays_in_the_writer_and_records_are_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.jsonl"
            entered, release = threading.Event(), threading.Event()
            original_open = os.open
            thread_ids = []

            def slow_open(*args, **kwargs):
                thread_ids.append(threading.get_ident())
                entered.set()
                if not release.wait(timeout=3):
                    raise RuntimeError("The test must release the writer.")
                return original_open(*args, **kwargs)

            with patch.object(hooks.os, "open", side_effect=slow_open):
                record = {"tokens": [1, 2], "round": 1}
                hooks._write_event(path, record)
                self.assertTrue(entered.wait(timeout=1))
                record["tokens"].append(3)
                hooks._write_event(path, {"round": 2})
                self.assertNotEqual(thread_ids, [threading.get_ident()])
                release.set()
                hooks.drain_events()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(rows, [{"tokens": [1, 2], "round": 1}, {"round": 2}])
            self.assertEqual(len(thread_ids), 1)

    def test_write_errors_surface_at_drain_and_close_stops_the_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = hooks._DiagnosticWriter(Path(temporary) / "trace.jsonl")
            with patch.object(hooks.os, "open", side_effect=PermissionError("write denied")):
                writer.put('{"round": 1}\n')
                with self.assertRaisesRegex(RuntimeError, "evidence writer failed"):
                    writer.drain()
                with self.assertRaisesRegex(RuntimeError, "evidence writer failed"):
                    writer.close()
            self.assertTrue(writer.closed)
            self.assertFalse(writer.thread.is_alive())


if __name__ == "__main__":
    unittest.main()
