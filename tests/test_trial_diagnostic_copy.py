"""Verify that diagnostic copies preserve the installed package's contents."""

import errno
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baselines"))
import trial_prepare_diagnostics as preparation
import trial_entrypoint as entrypoint


def identity(path):
    metadata = path.stat()
    return metadata.st_dev, metadata.st_ino


class DiagnosticCopyTests(unittest.TestCase):
    def test_vllm_step_identity_is_a_declared_serializable_dataclass_field(self):
        original = "from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass SchedulerOutput:\n    num_spec_tokens_to_schedule: int = 0\n"
        patched = preparation.patch_source("vllm", "v1/core/sched/output.py", original)
        namespace = {}
        exec(compile(patched, "output.py", "exec"), namespace)
        instance = namespace["SchedulerOutput"](hpml_diagnostic_step_id="step-one")
        encoded = json.loads(json.dumps(dataclasses.asdict(instance)))
        self.assertEqual(encoded["hpml_diagnostic_step_id"], "step-one")
        self.assertIsNone(namespace["SchedulerOutput"]().hpml_diagnostic_step_id)
        with self.assertRaisesRegex(ValueError, "marker must occur once"):
            preparation.patch_source("vllm", "v1/core/sched/output.py", patched)

    def test_sglang_reference_worker_patch_and_complete_manifest_are_required(self):
        original = "class TpModelWorker:\n    def forward_batch_generation(self, batch):\n        return batch\n"
        patched = preparation.patch_source("sglang", "srt/managers/tp_worker.py", original)
        compile(patched, "tp_worker.py", "exec")
        self.assertIn("install_sglang_worker(TpModelWorker)", patched)
        with mock.patch.dict(os.environ, {"HPML_DIAGNOSTIC_DIR": ""}):
            namespace = {}
            exec(patched, namespace)
            token = object()
            self.assertIs(namespace["TpModelWorker"]().forward_batch_generation(token), token)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            parent = root / "results/trial-first30-20261122/diagnostic-runtime/sglang"
            package = parent / "sglang"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            relatives = ["srt/managers/scheduler_components/batch_result_processor.py",
                         "srt/speculative/eagle_worker_v2.py", "srt/speculative/dflash_worker_v2.py",
                         "srt/managers/tp_worker.py"]
            patches = []
            for relative in relatives:
                path = package / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(patched if relative.endswith("tp_worker.py") else "fixture = True\n")
                patches.append({"relative_path": relative, "patched_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest = {"schema": 1, "engine": "sglang", "patches": patches,
                        "results_relative_package": package.relative_to(root / "results").as_posix()}
            record = parent / "diagnostic-instrumentation.json"
            record.write_text(json.dumps(manifest))
            with mock.patch.dict(os.environ, {"HPML_ROOT": str(root)}):
                self.assertEqual(entrypoint.validate_package(parent, "sglang"), parent)
                manifest["patches"] = patches[:-1]
                record.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "all 4 patched"):
                    entrypoint.validate_package(parent, "sglang")

    def test_python_changes_and_copy_cleanup_preserve_the_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            source.mkdir()
            source_python = source / "runner.py"
            source_python.write_text("value = 1\n")
            source_library = source / "_C.abi3.so"
            library_bytes = b"\x7fELF\x02\x01\x01\x00synthetic-library-content"
            source_library.write_bytes(library_bytes)
            copied = root / "diagnostic"
            linked = []
            shutil.copytree(source, copied, copy_function=preparation.package_copy_function(source, linked))

            copied_python = copied / "runner.py"
            copied_library = copied / "_C.abi3.so"
            self.assertNotEqual(identity(source_python), identity(copied_python))
            self.assertEqual(identity(source_library), identity(copied_library))
            self.assertEqual(copied_library.read_bytes(), library_bytes)
            self.assertEqual(len(linked), 1)
            self.assertEqual(linked[0]["relative_path"], "_C.abi3.so")
            self.assertEqual(linked[0]["source_path"], str(source_library.resolve()))
            self.assertEqual(linked[0]["size_bytes"], len(library_bytes))

            copied_python.write_text("value = 2\n")
            self.assertEqual(source_python.read_text(), "value = 1\n")
            previous_inode = identity(copied_python)
            preparation.require_independent_file(source_python, copied_python)
            preparation.atomic_replace(copied_python, b"value = 3\n")
            self.assertNotEqual(identity(copied_python), previous_inode)
            self.assertEqual(copied_python.read_text(), "value = 3\n")
            self.assertEqual(source_python.read_text(), "value = 1\n")
            self.assertEqual(list(copied.glob(".hpml-diagnostic-*.tmp")), [])

            shutil.rmtree(copied)
            self.assertTrue(source_library.is_file())
            self.assertEqual(source_library.read_bytes(), library_bytes)
            self.assertEqual(source_python.read_text(), "value = 1\n")

    def test_non_elf_file_with_shared_library_suffix_gets_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            source.mkdir()
            original = source / "example.so"
            original.write_text("This fixture contains ordinary text.\n")
            copied = root / "copy.so"
            linked = []
            preparation.package_copy_function(source, linked)(original, copied)
            self.assertNotEqual(identity(original), identity(copied))
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertEqual(linked, [])
            copied.write_text("The copied text has changed.\n")
            self.assertEqual(original.read_text(), "This fixture contains ordinary text.\n")

    def test_cross_filesystem_library_falls_back_to_an_independent_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            source.mkdir()
            original = source / "libexample.so.1"
            original.write_bytes(b"\x7fELFsynthetic-versioned-library")
            copied = root / "libexample.so.1"
            linked = []
            with mock.patch.object(preparation.os, "link", side_effect=OSError(errno.EXDEV, "Different filesystems")) as link:
                with mock.patch("builtins.print") as notice:
                    preparation.package_copy_function(source, linked)(original, copied)
            link.assert_called_once_with(original.resolve(), copied)
            self.assertTrue(any("another filesystem" in str(call) for call in notice.call_args_list))
            self.assertNotEqual(identity(original), identity(copied))
            self.assertEqual(original.read_bytes(), copied.read_bytes())
            self.assertEqual(linked, [])

    def test_patch_guard_rejects_a_python_file_that_shares_its_source_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("value = 1\n")
            copied = root / "copy.py"
            os.link(source, copied)
            with self.assertRaisesRegex(RuntimeError, "independent inode"):
                preparation.require_independent_file(source, copied)
            self.assertEqual(source.read_text(), "value = 1\n")


if __name__ == "__main__":
    unittest.main()
