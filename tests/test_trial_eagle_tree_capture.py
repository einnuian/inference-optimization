"""Check bounded raw EAGLE captures without allocating a GPU."""

import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_phase_hooks as hooks


class SmallTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)
        self.dtype = "torch.int64"
        self.device = "cuda:0"
        self.copies = 0

    def numel(self):
        return len(self.values)

    def detach(self):
        return self

    def cpu(self):
        self.copies += 1
        return self

    def tolist(self):
        return self.values


class EagleTreeCaptureTests(unittest.TestCase):
    def setUp(self):
        self.records = []
        self.context = {"observation_id": "observation-1", "phases": []}
        self.info = types.SimpleNamespace(
            draft_token=SmallTensor([10, 11, 12]),
            positions=SmallTensor([8, 9, 9]),
            retrieve_index=SmallTensor([0, 1, 2]),
            retrieve_next_token=SmallTensor([1, -1, -1]),
            retrieve_next_sibling=SmallTensor([-1, 2, -1]),
            draft_token_num=3,
        )
        self.batch = types.SimpleNamespace(
            spec_info=self.info, reqs=[types.SimpleNamespace(rid="prompt-1")]
        )
        self.enter = mock.patch.dict(os.environ, {"HPML_DIAGNOSTIC_EAGLE_TREE": "1"})
        self.enter.start()
        self.addCleanup(self.enter.stop)
        for patcher in (
            mock.patch.object(hooks, "_active", return_value=self.context),
            mock.patch.object(hooks, "_metadata", return_value={"diagnostic_only": True}),
            mock.patch.object(hooks, "_emit", side_effect=lambda context, record: self.records.append(record)),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_pre_verification_capture_preserves_raw_fields_and_request_identity(self):
        capture = hooks._capture_eagle_tree((object(), self.batch), {})
        record = self.records[0]
        self.assertEqual(record["event"], "eagle_tree_before_verify")
        self.assertEqual(record["request_ids_from_verify_batch"], ["prompt-1"])
        self.assertEqual(record["raw_tree"]["positions"]["value"], [8, 9, 9])
        self.assertEqual(record["raw_tree"]["draft_token_num"]["value"], 3)
        self.assertEqual(record["missing_tree_fields"], [])
        self.assertEqual(record["capture_id"], capture[0])
        self.assertEqual(self.info.draft_token.copies, 1)
        self.assertGreaterEqual(record["cpu_capture_end_ns"], record["cpu_capture_start_ns"])

    def test_oversize_tensor_is_explicitly_omitted_before_copy(self):
        self.info.positions = SmallTensor(list(range(257)))
        hooks._capture_eagle_tree((), {"batch": self.batch})
        field = self.records[0]["raw_tree"]["positions"]
        self.assertEqual(field["value_count"], 257)
        self.assertIn("omitted", field)
        self.assertNotIn("value", field)
        self.assertEqual(self.info.positions.copies, 0)

    def test_result_capture_preserves_tuple_positions_and_available_acceptance(self):
        capture = hooks._capture_eagle_tree((object(), self.batch), {})
        accepted = SmallTensor([2])
        result = (SmallTensor([10, 11]), types.SimpleNamespace(accept_lens=accepted))
        self.info.accept_length_cpu = [2]
        hooks._capture_eagle_result(capture, result)
        record = self.records[1]
        items = record["raw_result"]["sequence_items"]
        self.assertEqual(record["capture_id"], self.records[0]["capture_id"])
        self.assertEqual(items[0]["index"], 0)
        self.assertEqual(items[0]["raw"]["value"], [10, 11])
        self.assertEqual(items[1]["fields"]["accept_lens"]["value"], [2])
        self.assertEqual(record["spec_info_after_verify"]["accept_length_cpu"]["value"], [2])

    def test_disabled_tree_capture_keeps_tensors_on_their_current_device(self):
        with mock.patch.dict(os.environ, {"HPML_DIAGNOSTIC_EAGLE_TREE": "0"}):
            capture = hooks._capture_eagle_tree((object(), self.batch), {})
        self.assertIsNone(capture)
        self.assertEqual(self.records, [])
        self.assertEqual(self.info.draft_token.copies, 0)


if __name__ == "__main__":
    unittest.main()
