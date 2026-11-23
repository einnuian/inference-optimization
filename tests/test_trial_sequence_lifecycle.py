"""Check sequence cleanup and recovery without a model server."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_sequence as sequence


class SequenceLifecycleTests(unittest.TestCase):
    def test_linux_group_members_include_live_workers_and_ignore_zombies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for pid, state, group in ((101, "Z", 101), (102, "S", 101), (103, "R", 999)):
                entry = root / str(pid)
                entry.mkdir()
                (entry / "stat").write_text("%d (worker (name)) %s 1 %d %d 0\n" % (pid, state, group, group))
            (root / "self").mkdir()
            self.assertEqual(sequence.linux_group_members(101, root, os.getuid()), [102])

    def test_linux_group_probe_uses_process_records(self):
        with mock.patch.object(sequence.sys, "platform", "linux"):
            with mock.patch.object(sequence, "linux_group_members", return_value=[102]):
                with mock.patch.object(sequence.os, "killpg", side_effect=PermissionError) as signal_group:
                    self.assertTrue(sequence.group_alive(101))
        signal_group.assert_not_called()

    def test_shutdown_waits_for_workers_after_the_leader_exits(self):
        process = mock.Mock(pid=101)
        process.poll.return_value = 0
        with mock.patch.object(sequence, "group_alive", side_effect=[True, True, False]):
            with mock.patch.object(sequence.os, "killpg") as signal_group:
                with mock.patch.object(sequence.time, "sleep"):
                    sequence.stop_server(process)
        signal_group.assert_called_once_with(101, signal.SIGINT)
        process.wait.assert_called_once_with(timeout=5)

    def test_shutdown_accepts_an_empty_live_group(self):
        process = mock.Mock(pid=101)
        with mock.patch.object(sequence, "group_alive", return_value=False):
            with mock.patch.object(sequence.os, "killpg") as signal_group:
                sequence.stop_server(process)
        signal_group.assert_not_called()
        process.wait.assert_called_once_with(timeout=5)

    def test_termination_and_hangup_begin_cleanup_once(self):
        for signum in (signal.SIGTERM, signal.SIGHUP):
            handlers = {}

            def register(selected, handler):
                handlers[selected] = handler
                return signal.SIG_DFL

            with mock.patch.object(sequence.signal, "signal", side_effect=register):
                previous = sequence.install_interrupt_handlers()
            self.assertEqual(previous[signum], signal.SIG_DFL)
            with self.assertRaises(sequence.SequenceInterrupted):
                handlers[signum](signum, None)
            self.assertIsNone(handlers[signum](signum, None))

    def test_interrupted_request_client_is_terminated_and_reaped(self):
        child = mock.MagicMock()
        child.__enter__.return_value = child
        child.wait.side_effect = [sequence.SequenceInterrupted(signal.SIGTERM), 0]
        with mock.patch.object(sequence.subprocess, "Popen", return_value=child):
            with self.assertRaises(sequence.SequenceInterrupted):
                sequence.run_checked(["request-client"])
        child.terminate.assert_called_once()
        child.kill.assert_not_called()
        self.assertEqual(child.wait.call_args_list, [mock.call(), mock.call(timeout=5)])

    def test_unresponsive_request_client_is_killed_and_reaped(self):
        child = mock.MagicMock()
        child.__enter__.return_value = child
        child.wait.side_effect = [sequence.SequenceInterrupted(signal.SIGHUP),
                                  subprocess.TimeoutExpired(["request-client"], 5), -9]
        with mock.patch.object(sequence.subprocess, "Popen", return_value=child):
            with self.assertRaises(sequence.SequenceInterrupted):
                sequence.run_checked(["request-client"])
        child.terminate.assert_called_once()
        child.kill.assert_called_once()
        self.assertEqual(child.wait.call_args_list[-1], mock.call(timeout=5))

    def test_port_probe_enables_reuse_and_still_rejects_an_active_listener(self):
        probe = mock.MagicMock()
        probe.__enter__.return_value = probe
        with mock.patch.object(sequence.socket, "socket", return_value=probe):
            self.assertTrue(sequence.port_free(31000))
            probe.bind.side_effect = OSError("The address is occupied.")
            self.assertFalse(sequence.port_free(31000))
        probe.setsockopt.assert_any_call(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def test_partial_results_are_preserved_with_unique_failed_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempts = root / "attempts"
            attempts.mkdir()
            destination = root / "eagle3"
            archived = []
            for content in (b'{"partial":1}\n', b'{"partial":2}\n'):
                destination.mkdir()
                (destination / "saved.json").write_bytes(content)
                with mock.patch("builtins.print"):
                    status = sequence.preserve_partial(destination, attempts, "eagle3", ValueError("Incomplete run."))
                self.assertFalse(destination.exists())
                path = Path(status["preserved_result_rename"]["to"])
                self.assertEqual((path / "saved.json").read_bytes(), content)
                self.assertEqual(json.loads((path.parent / "status.json").read_text())["status"], "failed")
                archived.append(path)
            self.assertNotEqual(archived[0], archived[1])
            self.assertEqual((archived[0] / "saved.json").read_bytes(), b'{"partial":1}\n')


if __name__ == "__main__":
    unittest.main()
