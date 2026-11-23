"""Exercise canonical target acquisition with tiny, offline synthetic files.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import download_checkpoints as downloader
import import_meta_target as importer


class TargetProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        temporary = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="hpml-provenance-test-"))
        self.home = Path(temporary).resolve() / "huggingface"
        self.home.mkdir()
        self.target = next(a for a in downloader.ARTIFACTS if a.name == "target")
        self.snapshot = (self.home / "hub" / ("models--" + self.target.repo_id.replace("/", "--"))
                         / "snapshots" / self.target.revision)
        self.destination = self.home / "hpml-provenance/target" / f"{self.target.revision}.json"
        self.payloads = {}
        for name in downloader.TARGET_FILES:
            if name == "model.safetensors.index.json":
                value = {"weight_map": {f"tensor_{i}": shard
                                         for i, shard in enumerate(downloader.WEIGHT_SHA256)}}
                self.payloads[name] = json.dumps(value).encode()
            elif name.endswith(".json"):
                self.payloads[name] = json.dumps({"fixture": name}).encode()
            else:
                self.payloads[name] = f"Synthetic fixture for {name}.\n".encode()
        sizes = {name: len(data) for name, data in self.payloads.items()}
        small = {name: hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
                 for name, data in self.payloads.items() if name in downloader.SMALL_FILE_BLOBS}
        weights = {name: hashlib.sha256(self.payloads[name]).hexdigest()
                   for name in downloader.WEIGHT_SHA256}
        for dictionary, replacement in ((downloader.TARGET_FILE_BYTES, sizes),
                                         (downloader.SMALL_FILE_BLOBS, small),
                                         (downloader.WEIGHT_SHA256, weights)):
            self.stack.enter_context(patch.dict(dictionary, replacement, clear=True))
        for module in (downloader, importer):
            self.stack.enter_context(patch.object(module, "require_compute_allocation", return_value=("123", "fixture-node")))
            self.stack.enter_context(patch.object(module, "require_scratch_cache", return_value=self.home))
        self.hub = types.ModuleType("huggingface_hub")
        self.hub.__version__ = "fixture-version"
        self.hub.get_token = Mock(return_value="synthetic-test-credential")
        self.hub.snapshot_download = Mock(side_effect=self.download_fixture)
        self.stack.enter_context(patch.dict(sys.modules, {"huggingface_hub": self.hub}))
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(redirect_stderr(io.StringIO()))
        old_umask = os.umask(0o077)
        self.addCleanup(os.umask, old_umask)

    def write_fixture(self, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for name, data in self.payloads.items():
            (directory / name).write_bytes(data)

    def download_fixture(self, **kwargs):
        self.assertEqual(kwargs["repo_id"], self.target.repo_id)
        self.assertEqual(kwargs["revision"], self.target.revision)
        self.assertEqual(kwargs["endpoint"], "https://huggingface.co")
        self.assertIs(kwargs["token"], True)
        self.assertEqual(set(kwargs["allow_patterns"]), set(self.payloads))
        self.write_fixture(self.snapshot)
        return str(self.snapshot)

    def run_cli(self):
        return downloader.main(["--download", "--only", "target"])

    def run_import(self):
        source = self.home / "meta-import"
        self.write_fixture(source)
        with patch.object(sys, "argv", ["import_meta_target.py", "--meta-files", str(source)]):
            return importer.main()

    def acquisition(self):
        return downloader.target_acquisition(self.target, "Synthetic direct-Meta acquisition fixture.")

    def test_fresh_cli_records_complete_canonical_acquisition(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.hub.snapshot_download.call_args.kwargs["force_download"])
        record = json.loads(self.destination.read_text())
        self.assertTrue(downloader.canonical_target_origin(record, self.target, self.snapshot))
        self.assertEqual(record["acquisition"]["weight_source_repo"], self.target.repo_id)
        self.assertEqual(record["acquisition"]["weight_source_revision"], self.target.revision)
        self.assertEqual(len(record["required_file_bytes"]), 13)
        self.assertEqual(len(record["acquisition"]["weight_sha256"]), 4)
        self.assertEqual(len(record["acquisition"]["canonical_small_file_git_blobs"]), 9)
        self.assertEqual(record["huggingface_hub_version"], "fixture-version")

    def test_browser_import_publishes_one_complete_record(self):
        replace = downloader.os.replace
        observed = []

        def observe(source, destination):
            record = json.loads(Path(source).read_text())
            self.assertTrue(downloader.canonical_target_origin(record, self.target, self.snapshot))
            self.assertEqual(record["import_tool"], "baselines/import_meta_target.py")
            self.assertFalse(self.destination.exists())
            observed.append(record)
            replace(source, destination)

        with patch.object(downloader.os, "replace", side_effect=observe):
            self.assertEqual(self.run_import(), 0)
        self.assertEqual(len(observed), 1)
        self.assertFalse(list(self.destination.parent.glob("*.tmp")))

    def test_repeated_cli_preserves_detailed_browser_origin(self):
        self.assertEqual(self.run_import(), 0)
        original = json.loads(self.destination.read_text())
        original["acquisition"]["transfer_receipt"] = {"note": "Retain the original detailed record."}
        original["researcher_note"] = "Retain additional provenance fields."
        self.destination.write_text(json.dumps(original))
        for _ in range(2):
            self.assertEqual(self.run_cli(), 0)
            record = json.loads(self.destination.read_text())
            self.assertFalse(self.hub.snapshot_download.call_args.kwargs["force_download"])
            self.assertEqual(record["acquisition"], original["acquisition"])
            self.assertEqual(record["import_tool"], original["import_tool"])
            self.assertEqual(record["researcher_note"], original["researcher_note"])

    def test_legacy_unattributed_cache_requires_canonical_refresh(self):
        self.write_fixture(self.snapshot)
        self.destination.parent.mkdir(parents=True)
        self.destination.write_text(json.dumps({
            "artifact": "target", "repo_id": self.target.repo_id, "revision": self.target.revision,
            "snapshot_path": str(self.snapshot.resolve()), "required_file_bytes": downloader.TARGET_FILE_BYTES,
            "import_tool": "legacy-unattributed-import.py",
        }))
        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(self.hub.snapshot_download.call_args.kwargs["force_download"])
        self.assertTrue(downloader.canonical_target_origin(
            json.loads(self.destination.read_text()), self.target, self.snapshot))
        self.assertNotIn("import_tool", json.loads(self.destination.read_text()))

    def test_same_size_corruption_of_each_file_preserves_existing_provenance(self):
        self.assertEqual(self.run_cli(), 0)
        original = self.destination.read_bytes()
        for name, data in self.payloads.items():
            with self.subTest(name=name):
                (self.snapshot / name).write_bytes(bytes([data[0] ^ 1]) + data[1:])
                with self.assertRaises(downloader.DownloadError):
                    downloader.save_provenance(self.home, self.target, self.snapshot, None,
                                               "124", "fixture-node", "fixture-version",
                                               acquisition=self.acquisition())
                self.assertEqual(self.destination.read_bytes(), original)
                self.assertFalse(list(self.destination.parent.glob("*.tmp")))
                (self.snapshot / name).write_bytes(data)

    def test_corrupt_fresh_cli_download_never_publishes_provenance(self):
        def corrupt(**kwargs):
            snapshot = self.download_fixture(**kwargs)
            name = next(iter(downloader.WEIGHT_SHA256))
            data = self.payloads[name]
            (self.snapshot / name).write_bytes(b"!" + data[1:])
            return snapshot
        self.hub.snapshot_download.side_effect = corrupt
        self.assertEqual(self.run_cli(), 1)
        self.assertFalse(self.destination.exists())

    def test_atomic_replacement_failure_keeps_previous_record(self):
        self.assertEqual(self.run_cli(), 0)
        original = self.destination.read_bytes()
        with patch.object(downloader.os, "replace", side_effect=OSError("Synthetic publication failure")):
            self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.destination.read_bytes(), original)
        self.assertFalse(list(self.destination.parent.glob("*.tmp")))

    def test_content_identity_requires_an_acquisition_record(self):
        self.write_fixture(self.snapshot)
        with self.assertRaises(downloader.DownloadError):
            downloader.save_provenance(self.home, self.target, self.snapshot, None,
                                       "123", "fixture-node", "fixture-version")
        self.assertFalse(self.destination.exists())

    def test_partial_browser_import_waits_for_all_four_weights(self):
        source = self.home / "meta-import"
        self.write_fixture(source)
        weights = list(downloader.WEIGHT_SHA256)
        for selected in (weights[:1], weights[1:]):
            argv = ["import_meta_target.py", "--meta-files", str(source), "--only-weights", *selected]
            with patch.object(sys, "argv", argv):
                self.assertEqual(importer.main(), 0)
            if len(selected) == 1:
                self.assertFalse(self.destination.exists())
        self.assertTrue(downloader.canonical_target_origin(
            json.loads(self.destination.read_text()), self.target, self.snapshot))

    def test_partial_browser_import_rejects_unknown_origin_of_cached_shards(self):
        self.write_fixture(self.snapshot)
        source = self.home / "meta-import"
        self.write_fixture(source)
        weights = list(downloader.WEIGHT_SHA256)
        for name in weights[1:]:
            (source / name).unlink()
        argv = ["import_meta_target.py", "--meta-files", str(source), "--only-weights", weights[0]]
        with patch.object(sys, "argv", argv):
            self.assertEqual(importer.main(), 1)
        self.assertFalse(self.destination.exists())
        self.assertTrue(all((self.snapshot / name).is_file() for name in weights))
        for name in weights[1:]:
            (source / name).write_bytes(self.payloads[name])
        (source / weights[1]).write_bytes(b"!" + self.payloads[weights[1]][1:])
        with patch.object(sys, "argv", argv):
            self.assertEqual(importer.main(), 1)
        self.assertFalse(self.destination.exists())
        (source / weights[1]).write_bytes(self.payloads[weights[1]])
        with patch.object(sys, "argv", argv):
            self.assertEqual(importer.main(), 0)
        self.assertTrue(downloader.canonical_target_origin(
            json.loads(self.destination.read_text()), self.target, self.snapshot))

    def test_partial_browser_import_preserves_established_meta_origin(self):
        self.assertEqual(self.run_import(), 0)
        original = json.loads(self.destination.read_text())["acquisition"]
        source = self.home / "meta-import"
        weights = list(downloader.WEIGHT_SHA256)
        for name in weights[1:]:
            (source / name).unlink()
        argv = ["import_meta_target.py", "--meta-files", str(source), "--only-weights", weights[0]]
        with patch.object(sys, "argv", argv):
            self.assertEqual(importer.main(), 0)
        self.assertEqual(json.loads(self.destination.read_text())["acquisition"], original)

    def test_public_drafter_keeps_anonymous_complete_snapshot_scope(self):
        artifact = next(a for a in downloader.ARTIFACTS if a.name == "eagle3")
        snapshot = (self.home / "hub" / ("models--" + artifact.repo_id.replace("/", "--"))
                    / "snapshots" / artifact.revision)

        def public_download(**kwargs):
            self.assertEqual(kwargs["repo_id"], artifact.repo_id)
            self.assertIs(kwargs["token"], False)
            self.assertIs(kwargs["force_download"], False)
            self.assertIsNone(kwargs["allow_patterns"])
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text('{"fixture": true}')
            (snapshot / "pytorch_model.bin").write_bytes(b"Synthetic released draft fixture")
            return str(snapshot)

        self.hub.snapshot_download.side_effect = public_download
        self.assertEqual(downloader.main(["--download", "--only", "eagle3"]), 0)
        self.hub.get_token.assert_not_called()
        record = json.loads((self.home / "hpml-provenance/eagle3" / f"{artifact.revision}.json").read_text())
        self.assertEqual(record["snapshot_scope"], "complete repository at pinned revision")
        self.assertNotIn("acquisition", record)


if __name__ == "__main__":
    unittest.main()
