"""Capture vLLM loading evidence while delegating every load to released code.

Use this worker for setup qualification. Its loading records describe original
checkpoint names, pre-mapper loader inputs, destination names and declared skips.
The worker leaves tensors and inference methods unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


def tensor_description(name, tensor):
    return {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}


def inventory(model):
    return {
        "class": f"{type(model).__module__}.{type(model).__name__}",
        "parameters": [tensor_description(k, v) for k, v in model.named_parameters()],
        "buffers": [tensor_description(k, v) for k, v in model.named_buffers()],
    }


class AuditedWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise RuntimeError("Qualification requires the released pretrained weights.")
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.models.utils import AutoWeightsLoader

        records, active = [], []
        original_load = DefaultModelLoader.load_weights
        original_get = DefaultModelLoader.get_all_weights
        original_auto = AutoWeightsLoader.load_weights

        def observed_get(loader, model_config, model):
            for name, tensor in original_get(loader, model_config, model):
                if active:
                    active[-1]["checkpoint_inputs"].append(tensor_description(name, tensor))
                yield name, tensor

        def observed_auto(loader, weights, *, mapper=None):
            event = {
                "module_class": f"{type(loader.module).__module__}.{type(loader.module).__name__}",
                "module_prefix": None, "inputs": [],
                "input_name_boundary": "Inputs supplied to AutoWeightsLoader before its optional WeightsMapper; the model's outer loader may already rename keys.",
                "skip_prefixes": list(loader.skip_prefixes),
                "skip_substrings": list(loader.skip_substrs),
                "ignore_unexpected_prefixes": list(loader.ignore_unexpected_prefixes),
                "ignore_unexpected_suffixes": list(loader.ignore_unexpected_suffixes),
                "mapper": repr(mapper),
            }
            if active:
                event["module_prefix"] = next(
                    (name for name, module in active[-1]["_model"].named_modules()
                     if module is loader.module), None)
                active[-1]["automatic_loader_calls"].append(event)

            def observed_weights():
                for name, tensor in weights:
                    event["inputs"].append(tensor_description(name, tensor))
                    yield name, tensor

            try:
                result = original_auto(loader, observed_weights(), mapper=mapper)
                event["loaded_destination_names"] = sorted(result) if result is not None else None
                return result
            finally:
                event["effective_skip_prefixes"] = list(loader.skip_prefixes)
                event["effective_skip_substrings"] = list(loader.skip_substrs)
                event["effective_ignore_unexpected_prefixes"] = list(loader.ignore_unexpected_prefixes)
                event["effective_ignore_unexpected_suffixes"] = list(loader.ignore_unexpected_suffixes)

        def observed_load(loader, model, model_config):
            event = {
                "model_path": str(model_config.model),
                "model_class": f"{type(model).__module__}.{type(model).__name__}",
                "checkpoint_inputs": [], "automatic_loader_calls": [],
                "model_load_weights_return": None, "completed": False,
                "_model": model,
            }
            records.append(event)
            active.append(event)
            original_model_load = model.load_weights
            had_instance_method = "load_weights" in model.__dict__
            original_instance_method = model.__dict__.get("load_weights")

            def observed_model_load(weights):
                result = original_model_load(weights)
                event["model_load_weights_return"] = sorted(result) if result is not None else None
                return result

            model.load_weights = observed_model_load
            try:
                result = original_load(loader, model, model_config)
                event["completed"] = True
                return result
            except Exception as error:
                event["error"] = f"{type(error).__name__}: {error}"
                raise
            finally:
                if had_instance_method:
                    model.load_weights = original_instance_method
                else:
                    del model.load_weights
                event["state_after_load"] = inventory(model)
                event.pop("_model")
                active.pop()

        DefaultModelLoader.get_all_weights = observed_get
        DefaultModelLoader.load_weights = observed_load
        AutoWeightsLoader.load_weights = observed_auto
        self.hpml_loading_records = records
        try:
            return super().load_model(load_dummy_weights=False)
        finally:
            DefaultModelLoader.get_all_weights = original_get
            DefaultModelLoader.load_weights = original_load
            AutoWeightsLoader.load_weights = original_auto
            destination = Path(os.environ["HPML_HSPEC_AUDIT_DIR"]) / f"load-audit-{os.getpid()}.json"
            with destination.open("x") as handle:
                json.dump({"models": records, "scope": (
                    "Complete observed checkpoint-name and automatic-loader traces. "
                    "Declared skipped tensors and shared target parameters require the documented loader interpretation. "
                    "These records describe keys, shapes and dtypes; tensor-value identity is a separate check."
                )}, handle, indent=2)
                handle.write("\n")

    def hpml_model_state(self):
        runner = self.model_runner
        target = runner.get_model()
        producer = getattr(runner, "speculator", None)
        if producer is None:
            producer = getattr(runner, "drafter", None)
        draft = getattr(producer, "model", None) if producer is not None else None
        models = {"target": target}
        if draft is not None:
            models["draft"] = draft
        result = {
            "runner_class": f"{type(runner).__module__}.{type(runner).__name__}",
            "producer_class": f"{type(producer).__module__}.{type(producer).__name__}" if producer is not None else None,
            "models": {}, "shared_target_parameters": [],
        }
        target_pointers = {value.data_ptr(): name for name, value in target.named_parameters()}
        for role, model in models.items():
            details = inventory(model)
            details["attention_layers"] = []
            for name, module in model.named_modules():
                backend = getattr(module, "attn_backend", None)
                if backend is not None and hasattr(backend, "get_name"):
                    details["attention_layers"].append({
                        "name": name, "backend": backend.get_name(),
                        "implementation": type(getattr(module, "impl", None)).__name__,
                    })
            result["models"][role] = details
            if role == "draft":
                for name, value in model.named_parameters():
                    if value.data_ptr() in target_pointers:
                        result["shared_target_parameters"].append({
                            "draft_name": name, "target_name": target_pointers[value.data_ptr()],
                            "shape": list(value.shape), "dtype": str(value.dtype),
                        })
        return result
