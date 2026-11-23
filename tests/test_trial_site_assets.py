"""Check published recording completeness, exact inputs, and safe asset refreshes."""

import hashlib
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_site_assets as assets
from test_trial_wandb import configuration, save


def fixture(directory, name="ar-sglang"):
    root = directory / "trial-first30-20261122"
    root.mkdir()
    config = configuration(root, name)
    prompts = []
    for benchmark in assets.BENCHMARKS:
        for index in range(3):
            content = "Exact prompt %s %d.\nPreserve this second line." % (benchmark, index)
            prompts.append({"id": "%s-%d" % (benchmark, index), "benchmark": benchmark,
                            "sourceIndex": index + (7 if benchmark == "mbpp" else 0),
                            "sourceItemId": None, "promptSha256": hashlib.sha256(content.encode()).hexdigest(),
                            "input_token_ids": [1, index + 2], "messages": [{"role": "user", "content": content}]})
        for repetition in range(1, 4):
            path = config / ("%02d-%s.json" % (repetition, benchmark))
            block = json.loads(path.read_text())
            for row in block["requests"]:
                row["events"] = [{"received_ns": row["submitted_ns"] + 100_000_000,
                                  "text": "Fixture response.", "token_ids": [10, 11]}]
            save(path, block)
    prepared = directory / "prepared.json"
    save(prepared, {"input_sha256": assets.INPUT_SHA256, "target_revision": assets.TARGET_REVISION,
                    "prompts": prompts, "warmups": []})
    run = json.loads((config / "run.json").read_text())
    run["prepared_sha256"] = assets.digest(prepared)
    save(config / "run.json", run)
    site = directory / "site"
    (site / "app").mkdir(parents=True)
    (site / "lib").mkdir()
    (site / "package.json").write_text("{}")
    return root, prepared, site


class RecordingAssetTests(unittest.TestCase):
    def test_large_json_blocks_keep_logical_identity_and_exact_original_archive_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            source = root / "ar-sglang/01-gsm8k.json"
            block = json.loads(source.read_text())
            block["requests"][0]["server_metadata"]["fixture_padding"] = "Exact metadata. " * 10_000
            save(source, block)
            original = source.read_bytes()
            result = assets.package_trial(root, prepared, site)
            folder = site / "public/measurements" / root.name
            entry = next(row for row in result["files"]
                         if row.get("sourcePath") == "ar-sglang/01-gsm8k.json")
            self.assertEqual(entry["compression"], "gzip")
            self.assertEqual(entry["path"], "ar-sglang/01-gsm8k.json.gz")
            self.assertEqual(entry["uncompressedBytes"], len(original))
            self.assertEqual(entry["uncompressedSha256"], hashlib.sha256(original).hexdigest())
            self.assertEqual(gzip.decompress((folder / entry["path"]).read_bytes()), original)
            self.assertFalse((folder / entry["sourcePath"]).exists())
            with zipfile.ZipFile(folder / "all-recordings.zip") as zipped:
                self.assertEqual(zipped.read(root.name + "/ar-sglang/01-gsm8k.json"), original)
            self.assertTrue(any(row["path"] == "ar-sglang/run.json" for row in result["files"]))

    def test_complete_team_recordings_publish_with_an_unfinished_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary), "team-ffn-100")
            partial = root / "team-ar"
            partial.mkdir()
            save(partial / "run.json", {"partial": True})
            result = assets.package_trial(root, prepared, site)
            self.assertEqual(result["requestCount"], 90)
            self.assertEqual([row["id"] for row in result["configurations"]], ["team-ffn-100"])
            self.assertEqual(result["configurations"][0]["label"], "Pilot FFN 100%")
            folder = site / "public/measurements" / root.name
            with zipfile.ZipFile(folder / "all-recordings.zip") as zipped:
                self.assertFalse(any("/team-ar/" in name for name in zipped.namelist()))
            (root / "team-ffn-100/03-gsm8k.json").unlink()
            with self.assertRaisesRegex(ValueError, "30 completed benchmark blocks"):
                assets.package_trial(root, prepared, site)

    def test_large_trace_downloads_are_compressed_and_archive_preserves_original_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            original = b'{"event":"memory sample","bytes":123456789}\n' * 10_000
            (root / "gpu-memory.jsonl").write_bytes(original)
            with mock.patch.object(assets, "MAX_ASSET_BYTES", 64 * 1024):
                result = assets.package_trial(root, prepared, site)
            folder = site / "public/measurements" / root.name
            entry = next(row for row in result["files"] if row["path"] == "gpu-memory.jsonl.gz")
            self.assertEqual(gzip.decompress((folder / entry["path"]).read_bytes()), original)
            self.assertEqual(entry["sourcePath"], "gpu-memory.jsonl")
            self.assertEqual(entry["uncompressedSha256"], hashlib.sha256(original).hexdigest())
            with zipfile.ZipFile(folder / "all-recordings.zip") as zipped:
                self.assertEqual(zipped.read(root.name + "/gpu-memory.jsonl"), original)

    def test_completed_responses_and_exact_prompts_are_published_with_raw_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            partial = root / "dflash"
            partial.mkdir()
            save(partial / "run.json", {"partial": True})
            (root / "credentials.json").write_text('{"secret":"excluded"}')
            (root / "diagnostic-runtime").mkdir()
            (root / "diagnostic-runtime" / "library.so").write_bytes(b"excluded")
            save(root / "resource-summary.json", {"allocated_gpu_seconds": 80})
            result = assets.package_trial(root, prepared, site)
            self.assertEqual(result["requestCount"], 90)
            self.assertEqual([row["id"] for row in result["configurations"]], ["ar-sglang"])
            self.assertEqual(next(row for row in result["benchmarks"] if row["id"] == "mbpp")["prompts"][0]["ordinal"], 1)
            folder = site / "public" / "measurements" / root.name
            self.assertEqual((folder / "prompts.json").read_text(), json.dumps(json.loads(prepared.read_text()), ensure_ascii=False, indent=2) + "\n")
            with zipfile.ZipFile(folder / "all-recordings.zip") as zipped:
                names = zipped.namelist()
                self.assertIn(root.name + "/resource-summary.json", names)
                self.assertFalse(any("credential" in name or "diagnostic-runtime" in name or "/dflash/" in name for name in names))
                block = json.loads(zipped.read(root.name + "/ar-sglang/01-gsm8k.json"))
                self.assertEqual(block["requests"][0]["server_metadata"]["spec_verify_ct"], 1)
                self.assertEqual(block["requests"][0]["events"][0]["text"], "Fixture response.")
            for entry in result["files"] + [result["archive"]]:
                self.assertEqual(entry["sha256"], assets.digest(folder / entry["path"]))
                self.assertLessEqual(entry["bytes"], assets.MAX_ASSET_BYTES)

    def test_invalid_completed_run_fails_and_preserves_the_previous_assets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            assets.package_trial(root, prepared, site)
            folder = site / "public" / "measurements" / root.name
            original_archive = assets.digest(folder / "all-recordings.zip")
            original_index = (site / "lib/trial-recordings.json").read_bytes()
            (root / "ar-sglang/03-gsm8k.json").unlink()
            with self.assertRaisesRegex(ValueError, "30 completed benchmark blocks"):
                assets.package_trial(root, prepared, site)
            self.assertEqual(assets.digest(folder / "all-recordings.zip"), original_archive)
            self.assertEqual((site / "lib/trial-recordings.json").read_bytes(), original_index)

    def test_changed_prompt_file_and_out_of_order_events_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            original = prepared.read_bytes()
            prepared.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "prompt file differs"):
                assets.package_trial(root, prepared, site)
            prepared.write_bytes(original)
            path = root / "ar-sglang/01-gsm8k.json"
            block = json.loads(path.read_text())
            block["requests"][0]["events"][0]["received_ns"] = block["requests"][0]["completed_ns"] + 1
            save(path, block)
            with self.assertRaisesRegex(ValueError, "Stream arrival times"):
                assets.package_trial(root, prepared, site)

    def test_repeat_packaging_is_reproducible_and_size_failure_preserves_previous_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            first = assets.package_trial(root, prepared, site)
            second = assets.package_trial(root, prepared, site)
            self.assertEqual(first, second)
            with mock.patch.object(assets, "MAX_ASSET_BYTES", 100):
                with self.assertRaisesRegex(ValueError, "hosting limit"):
                    assets.package_trial(root, prepared, site)
            self.assertEqual(json.loads((site / "lib/trial-recordings.json").read_text()), first)
            self.assertEqual([path.name for path in (site / "public/measurements").iterdir()], [root.name])

    def test_unowned_destination_and_linked_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, prepared, site = fixture(Path(temporary))
            destination = site / "public/measurements" / root.name
            destination.mkdir(parents=True)
            (destination / "keep.txt").write_text("User-owned file.")
            with self.assertRaisesRegex(ValueError, "outside this packager's ownership"):
                assets.package_trial(root, prepared, site)
            self.assertEqual((destination / "keep.txt").read_text(), "User-owned file.")
            source = root / "resource-summary.json"
            source.symlink_to(prepared)
            with self.assertRaisesRegex(ValueError, "regular local file"):
                assets.package_trial(root, prepared, site)


if __name__ == "__main__":
    unittest.main()
