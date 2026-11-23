"""Validate the team adapter's source selection, vocabulary, and stop semantics."""

import json
from contextlib import nullcontext
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baselines"))
import team_peagle_trial as team
import team_peagle_train as training
import team_peagle_import as importer


class TeamAdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_exact_notebook_definitions_compile_without_running_cells(self):
        path = self.root / "pinned.ipynb"
        team.export_notebook(path)
        code = team.notebook_definitions(path)
        self.assertEqual(team.sha256(path), team.NOTEBOOK_SHA256)
        self.assertIn("Drafter", code.co_names)
        self.assertIn("train", code.co_names)
        self.assertNotIn("drafters", code.co_names)
        self.assertNotIn("login", code.co_names)
        self.assertNotIn("AutoModelForCausalLM", code.co_names)

    def test_notebook_hash_change_fails_before_execution(self):
        path = self.root / "other.ipynb"
        path.write_text('{"cells": []}')
        with self.assertRaisesRegex(ValueError, "notebook bytes"):
            team.notebook_definitions(path)

    def test_source_export_preserves_existing_files(self):
        path = self.root / "existing.ipynb"
        path.write_text("user content")
        with self.assertRaises(FileExistsError):
            team.export_notebook(path)
        self.assertEqual(path.read_text(), "user content")

    def test_vocabulary_preserves_first_occurrence_ties_and_appends_stops(self):
        rows = [{"answer": [9, 7, 9, 7, 5]}, {"answer": [5, 4]}]
        self.assertEqual(team.draft_vocabulary(rows, [99], count=2), [9, 7, 99])

    def test_training_rows_require_original_integer_ids(self):
        path = self.root / "data.jsonl"
        path.write_text(json.dumps({"prompt": [1, 2], "answer": [3, 4]}) + "\n")
        self.assertEqual(len(team.training_rows(path, 1)), 1)
        with self.assertRaisesRegex(ValueError, "too few"):
            team.training_rows(path, 2)
        path.write_text(json.dumps({"prompt": [1], "answer": [True]}) + "\n")
        with self.assertRaisesRegex(ValueError, "token-ID"):
            team.training_rows(path, 1)

    def test_first_eos_trims_the_remainder_of_a_speculative_block(self):
        self.assertEqual(team.retained_chunk([3, 99, 4, 5], 8, {99}), ([3, 99], "stop", 99))

    def test_length_boundary_precedes_a_later_eos(self):
        self.assertEqual(team.retained_chunk([3, 4, 99], 2, {99}), ([3, 4], "length", None))
        self.assertEqual(team.retained_chunk([3, 99], 2, {99}), ([3, 99], "stop", 99))

    def test_partial_group_keeps_its_open_finish_state(self):
        self.assertEqual(team.retained_chunk([3, 4], 4, {99}), ([3, 4], None, None))

    def test_real_ar_generator_keeps_each_emitted_group_independent(self):
        class Logits:
            def __getitem__(self, key):
                return self
            def argmax(self):
                return self
            def item(self):
                return 3
        engine = object.__new__(team.TeamEngine)
        engine.diagnostic_dir = None
        engine.drafter = None
        engine.ns = {
            "torch": types.SimpleNamespace(no_grad=nullcontext, tensor=lambda data, device: data),
            "EOS": {99}, "target_step": lambda ids, cache: (None, Logits(), None),
        }
        groups = list(engine.generate([1, 2], 3, "test"))
        self.assertEqual([ids for ids, _, _ in groups], [[3], [3], [3]])
        self.assertEqual(groups[-1][1], "length")

    def test_runtime_versions_are_pinned(self):
        team.validate_runtime_versions("2.6.0+cu124", "12.4", "4.57.6", (3, 10, 22))
        with self.assertRaisesRegex(ValueError, "pinned team"):
            team.validate_runtime_versions("2.11.0+cu129", "12.9", "4.57.6", (3, 10, 22))

    def compact_metadata(self):
        return {
            "id": "request-1", "team_rounds_encoding": "accepted-delivered-pairs-v1",
            "team_rounds_compact": [[0, 1], [2, 2]], "spec_verify_ct": 2,
            "spec_num_correct_drafts": 2, "spec_num_proposed_drafts": 16,
            "team_initial_target_token_id": 10, "team_initial_delivered_tokens": 1,
            "team_delivered_output_tokens": 4,
        }

    def test_compact_counts_expand_with_pretrim_and_delivered_meanings(self):
        rows = team.expand_team_rounds(self.compact_metadata(), "request-1", True)
        self.assertEqual([row["committed_tokens"] for row in rows], [1, 3])
        self.assertEqual([row["delivered_tokens"] for row in rows], [1, 2])
        self.assertEqual([row["round_index"] for row in rows], [0, 1])
        metadata = self.compact_metadata()
        metadata["team_delivered_output_tokens"] = 5
        with self.assertRaisesRegex(ValueError, "Delivered round counts"):
            team.expand_team_rounds(metadata, "request-1", True)

    def test_artifact_expansion_preserves_raw_stream_metadata(self):
        metadata = self.compact_metadata()
        request = {
            "request_id": "request-1", "output_tokens": 4, "native_output_token_ids": [10, 11, 12, 13],
            "server_metadata": metadata, "events": [{"meta_info": metadata}],
        }
        path = self.root / "warmup.json"
        path.write_text(json.dumps([request]))
        team.finalize_round_artifacts(self.root, "team-ffn-75")
        saved = json.loads(path.read_text())[0]
        self.assertEqual(len(saved["server_metadata"]["team_rounds"]), 2)
        self.assertNotIn("team_rounds", saved["events"][0]["meta_info"])
        self.assertEqual(saved["events"][0]["meta_info"]["team_rounds_compact"], [[0, 1], [2, 2]])
        self.assertFalse(list(self.root.glob("*.expanding")))

    def test_original_checkpoint_shapes_match_all_three_widths(self):
        class Parameter:
            dtype = "torch.float32"
            def __init__(self, shape):
                self.shape = shape
            def is_floating_point(self):
                return True
        counts = []
        for width in (1.0, .75, .5):
            state = {"update": 50, "drafter": {
                name: Parameter(shape) for name, shape in importer.expected_shapes(width).items()
            }}
            checked = importer.validate_state(state, width)
            counts.append(checked["parameter_count"])
            self.assertEqual(checked["updates"], 50)
        self.assertEqual(counts, [906022912, 729862144, 553701376])
        state["update"] = 49
        with self.assertRaisesRegex(ValueError, "50 updates"):
            importer.validate_state(state, .5)

    def test_overlap_reports_training_and_held_out_matches(self):
        texts = [
            {"selection_index": 0, "split": "training", "text": "  HELLO \n world "},
            {"selection_index": 200, "split": "held_out", "text": "Café"},
        ]
        evaluation = {"prompts": [
            {"id": "one", "benchmark": "mt-bench", "messages": [{"content": "hello world"}]},
            {"id": "two", "benchmark": "alpaca", "messages": [{"content": "Cafe\u0301"}]},
            {"id": "three", "benchmark": "gsm8k", "messages": [{"content": "Different question"}]},
        ]}
        result = training.overlap_report(texts, evaluation)
        self.assertEqual([r["evaluation_prompt_id"] for r in result["matches"]], ["one", "two"])
        self.assertEqual([r["split"] for r in result["matches"]], ["training", "held_out"])
        self.assertEqual(texts[0]["text"], "  HELLO \n world ")

    def test_prompt_recovery_requires_exact_token_roundtrip(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "<user>" + messages[0]["content"] + "</user><assistant>"
            def decode(self, tokens, **kwargs):
                return "<user>Question?</user><assistant>"
        namespace = {"tok": Tokenizer(), "chat_ids": lambda content: [1, 2, 3]}
        result = training.recover_prompt_texts(namespace, [{"prompt": [1, 2, 3]}])
        self.assertEqual(result[0]["text"], "Question?")
        namespace["chat_ids"] = lambda content: [1, 2]
        with self.assertRaisesRegex(ValueError, "roundtrip"):
            training.recover_prompt_texts(namespace, [{"prompt": [1, 2, 3]}])

    def test_saved_pilot_settings_and_checkpoint_final_save(self):
        self.assertEqual(team.PILOT["N_TRAIN"], 200)
        self.assertEqual(team.PILOT["N_VAL"], 50)
        self.assertEqual(team.PILOT["EPOCHS"] * team.PILOT["N_TRAIN"] // team.PILOT["ACCUM"], 50)
        self.assertEqual(team.PILOT["K"], 8)
        self.assertEqual(training.DATASET_REVISION, "8049631c405ae6576f93f445c6b8166f76f5505a")
        raw = subprocess.check_output(["git", "show", f"{team.NOTEBOOK_COMMIT}:peagle_ffn.ipynb"], cwd=ROOT)
        notebook = json.loads(raw)
        source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        self.assertIn("if update % 100 == 0 or update == total:", source)
        self.assertIn("torch.manual_seed(0)", source)

    def payload(self):
        return {"rid": "test", "input_ids": [1, 2], "stream": True, "sampling_params": {
            "temperature": 0, "ignore_eos": False, "skip_special_tokens": True,
            "top_p": 1, "top_k": -1, "stop_token_ids": [99], "max_new_tokens": 2048,
        }}

    def test_shared_streaming_request_is_accepted(self):
        self.assertEqual(team.validate_generation_payload(self.payload(), {99}, 100), ([1, 2], 2048, "test"))

    def test_sampling_and_stop_changes_are_rejected(self):
        for field, value in (("temperature", .5), ("stop_token_ids", [98]), ("max_new_tokens", 2049)):
            payload = self.payload()
            payload["sampling_params"][field] = value
            with self.assertRaises(ValueError):
                team.validate_generation_payload(payload, {99}, 100)

    def qualification_args(self):
        from trial_recorder import INPUT_SHA256, TARGET_REVISION
        prepared = self.root / "prepared.json"
        prepared.write_text(json.dumps({
            "input_sha256": INPUT_SHA256, "target_revision": TARGET_REVISION,
            "warmups": [{"id": f"warmup-{i}", "input_token_ids": [1, 2]} for i in range(3)],
            "prompts": [{"id": "longest-input", "input_token_ids": list(range(20))}],
        }))
        return types.SimpleNamespace(
            prepared=prepared, output=self.root / "qualification.json", configuration="team-ar")

    def fake_engine(self, wrong=False):
        class Engine:
            def __init__(self, args):
                self.drafter = None
                self.provenance = {"test": True}
                self.ns = {"EOS": {99}, "standard": lambda prompt, maximum: ([3] * maximum, [])}

            def generate(self, prompt, maximum, request_id):
                for index in range(maximum):
                    yield [4 if wrong and index == 0 else 3], "length" if index == maximum - 1 else None, None
        return Engine

    def test_qualification_checks_all_twelve_cases(self):
        args = self.qualification_args()
        with mock.patch.object(team, "TeamEngine", self.fake_engine()):
            team.qualify(args)
        result = json.loads(args.output.read_text())
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 12)
        self.assertEqual([row["prompt_id"] for row in result["checks"][-3:]], ["longest-input"] * 3)

    def test_qualification_saves_differences_and_rejects_them(self):
        args = self.qualification_args()
        with mock.patch.object(team, "TeamEngine", self.fake_engine(wrong=True)):
            with self.assertRaisesRegex(ValueError, "differs from the pinned notebook"):
                team.qualify(args)
        self.assertFalse(json.loads(args.output.read_text())["passed"])

    def test_paired_comparison_rejects_different_measurement_phases(self):
        method, reference = self.root / "method", self.root / "reference"
        method.mkdir()
        reference.mkdir()
        for path, phase in ((method, "primary"), (reference, "diagnostic")):
            (path / "run.json").write_text(json.dumps({
                "backend": "transformers-team", "phase": phase, "gpu_uuid": "GPU-one",
            }))
        args = types.SimpleNamespace(method=method, reference=reference, output=self.root / "paired.json")
        with self.assertRaisesRegex(ValueError, "phase"):
            team.compare(args)
        self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
