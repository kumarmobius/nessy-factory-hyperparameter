
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import json
import os
import csv
import importlib.util

class BaseComponent(ABC):
    """Abstract base class for all LM Factory components."""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the component's main logic."""
        pass

    def save(self, path: str) -> None:
        """Optional: Save state (e.g., model weights, tokenizer files)."""
        raise NotImplementedError(f"{self.name} does not support save()")

    def load(self, path: str) -> None:
        """Optional: Load state."""
        raise NotImplementedError(f"{self.name} does not support load()")
    
    @staticmethod
    def _dump_json(obj: dict, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    

    @staticmethod
    def count_learnable_parameters(model: Any) -> int:
        """Return learnable parameter count if possible; else -1."""
        try:
            total = 0
            for p in model.parameters():
                if getattr(p, "requires_grad", False):
                    total += int(p.numel())
            return total
        except Exception:
            return -1


class BaseExporter(BaseComponent):
    """Abstract base class for dataset exporters."""

    @abstractmethod
    def run(self, dataset: str, split: str = "train",
            text_fields: Optional[str] = None,
            max_rows: Optional[int] = None,
            streaming: bool = False,
            output_file: str = "exported.txt") -> Dict[str, Any]:
        """
        Export dataset to newline-delimited text.
        Must return metadata (dict) describing the run.
        """
        pass



class BaseTokenizer(BaseComponent):
    """Abstract base class for tokenizers."""

    @abstractmethod
    def run(self, text_file: str, vocab_size: int = 32000,
            min_frequency: int = 2,
            special_tokens: Optional[str] = "[PAD],[UNK],[BOS],[EOS]",
            add_bos_eos: bool = True) -> Any:
        """Train tokenizer on given text file."""
        pass

    @abstractmethod
    def encode(self, text: str) -> Any:
        """Encode string into token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: Any) -> str:
        """Decode token IDs back to string."""
        pass

    def save(self, path: str) -> None:
        """Save trained tokenizer model (e.g., tokenizer.json)."""
        raise NotImplementedError(f"{self.name} does not support save()")

    def load(self, path: str) -> None:
        """Load a tokenizer model from file."""
        raise NotImplementedError(f"{self.name} does not support load()")





class BaseModel(BaseComponent):
    """
    Minimal interface for a runtime model. We keep this generic so it does not
    require importing deep learning libraries in the base module.
    """

    @abstractmethod
    def forward(self, input_ids: Any, *args, **kwargs) -> Any:
        """Forward pass. Should return logits and optionally loss."""
        raise NotImplementedError

class BaseModelBuilder(BaseComponent):
    """
    Interface for constructing an untrained model given tokenizer & config,
    and emitting artifacts: weights, config JSON, reusable model code, schema JSON.
    """

    # ---- High-level entrypoint (recommended) ----
    @abstractmethod
    def run(
        self,
        tokenizer_json: str,
        n_layers: int,
        layer_pattern: str,
        model_weights_out: str,
        model_config_out: str,
        model_py_out: str,
        schema_json_out: str,
    ) -> Dict[str, Any]:
        """
        Build the model and write all outputs. Return a minimal schema/summary dict.
        Implementations typically:
          1) read tokenizer to get vocab_size
          2) parse layer_pattern -> layer_types
          3) assemble a config dict
          4) instantiate the model
          5) save weights/config/code/schema
        """
        raise NotImplementedError

    # ---- Utilities commonly needed by builders ----
    @staticmethod
    def pattern_to_layers(pattern: str, n_layers: int) -> List[str]:
        """
        Parse patterns like 'S*3,F*1,S*2' into a list of layer type names
        (e.g., ['sliding_attention', 'sliding_attention', ..., 'full_attention', ...])
        and validate it matches n_layers.
        """
        def norm(tok: str) -> str:
            t = tok.strip().lower()
            if t in ("s", "sliding", "sliding_attention"):
                return "sliding_attention"
            if t in ("f", "full", "full_attention"):
                return "full_attention"
            raise ValueError(f"Unknown layer type token: {tok}")

        blocks: List[Tuple[str, int]] = []
        for part in pattern.split(","):
            part = part.strip()
            if not part:
                continue
            if "*" in part:
                t_raw, c_raw = part.split("*", 1)
                t = norm(t_raw)
                c = int(c_raw)
                if c <= 0:
                    raise ValueError(f"Count must be > 0 in block '{part}'")
            else:
                t, c = norm(part), 1
            blocks.append((t, c))

        layer_list: List[str] = []
        for t, c in blocks:
            layer_list.extend([t] * c)

        if len(layer_list) != n_layers:
            raise ValueError(
                f"Pattern expands to {len(layer_list)} layers, but n_layers={n_layers}."
            )
        return layer_list

    # @staticmethod
    # def count_learnable_parameters(model: Any) -> int:
    #     """
    #     Best-effort parameter counting. If model exposes `.parameters()` that yield
    #     tensors with `numel()` and `requires_grad`, we use that. Otherwise returns -1.
    #     """
    #     try:
    #         total = 0
    #         for p in getattr(model, "parameters")():
    #             # works for PyTorch-like modules
    #             if getattr(p, "requires_grad", False):
    #                 total += int(p.numel())
    #         return total
    #     except Exception:
    #         return -1

    # ---- Artifact writers (generic, overridable) ----
    def save_config(self, config: Dict[str, Any], path: str) -> None:
        self._dump_json(config, path)

    def save_code(self, code_str: str, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code_str)

    def save_schema(self, schema: Dict[str, Any], path: str) -> None:
        self._dump_json(schema, path)

    def save_weights(self, model: Any, path: str) -> None:
        """
        Default implementation supports PyTorch-style models by calling
        torch.save(model.state_dict(), path). Subclasses may override for
        other frameworks (e.g., safetensors, flax, tf).
        """
        try:
            import torch  # local import to avoid hard dep in base
            state_dict = model.state_dict()  # type: ignore[attr-defined]
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(state_dict, path)
        except Exception as e:
            raise RuntimeError(
                f"{self.name} could not save weights automatically. "
                f"Override save_weights() in your builder. Original error: {e}"
            )


class BaseTrainer(BaseComponent):
    """Interface + reusable utilities for training pipelines."""

    # ------- Interface -------
    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute the training loop. Return summary/metrics dict."""
        raise NotImplementedError

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(f"{self.name} does not support evaluate()")

    # ------- Generic utilities (covering your trainer needs) -------

    @staticmethod
    def pick_device() -> Tuple[str, str]:
        """Return (device, device_type) like ('cuda','cuda') or ('cpu','cpu')."""
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            return dev, ("cuda" if dev == "cuda" else "cpu")
        except Exception:
            return "cpu", "cpu"

    @staticmethod
    def autocast_context(device_type: str):
        """
        Return (ctx, scaler, dtype_used):
          - ctx: a context manager (null on CPU, torch.amp.autocast on CUDA)
          - scaler: GradScaler if fp16 autocast enabled, else a no-op with .is_enabled()
          - dtype_used: 'bf16' | 'fp16' | 'fp32'
        """
        class _NoScaler:
            def is_enabled(self): return False
            def step(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass

        try:
            import torch
            if device_type == "cuda":
                # Prefer bf16 if supported, else fp16
                if torch.cuda.is_bf16_supported():
                    return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16), _NoScaler(), "bf16"
                else:
                    return torch.amp.autocast(device_type="cuda", dtype=torch.float16), torch.cuda.amp.GradScaler(), "fp16"
            else:
                from contextlib import nullcontext
                return nullcontext(), _NoScaler(), "fp32"
        except Exception:
            from contextlib import nullcontext
            return nullcontext(), _NoScaler(), "fp32"

    @staticmethod
    def scheduler_warmup_cosine(optimizer, total_updates: int, warmup_steps: int, min_lr: float):
        """Linear warmup for warmup_steps, then cosine to min_lr for remaining steps."""
        try:
            import torch
            from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
            warm = max(1, min(int(warmup_steps), int(total_updates)))
            rest = max(1, int(total_updates) - warm)
            sched_warm = LinearLR(optimizer, total_iters=warm)
            sched_cos  = CosineAnnealingLR(optimizer, T_max=rest, eta_min=min_lr)
            return SequentialLR(optimizer, schedulers=[sched_warm, sched_cos], milestones=[warm])
        except Exception as e:
            raise RuntimeError(f"Failed to build warmup→cosine scheduler: {e}")

    @staticmethod
    def clip_grad_norm(parameters, max_norm: float) -> None:
        """Clip gradients if torch is available; no-op otherwise."""
        try:
            import torch
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)
        except Exception:
            pass

    @staticmethod
    def estimate_loss_loop(model, get_batch_fn, eval_iters: int, ctx) -> Dict[str, float]:
        """
        Evaluate average loss on 'train' and 'val'. Assumes model(X,Y)->(_, loss).
        get_batch_fn(split) must return (X, Y).
        """
        out = {}
        try:
            import torch
            model.eval()
            with torch.inference_mode():
                for split in ["train", "val"]:
                    losses = []
                    for _ in range(int(eval_iters)):
                        X, Y = get_batch_fn(split)
                        with ctx:
                            _, loss = model(X, Y)
                        losses.append(float(loss.item()))
                    out[split] = sum(losses) / max(1, len(losses))
        finally:
            # Return to train mode if available
            try: model.train()
            except Exception: pass
        return out

    # ------- Artifact helpers (JSON, CSV, checkpoints, schema-string) -------

    @staticmethod
    def save_checkpoint(model, path: str) -> None:
        """Save state_dict if torch-like; creates dirs."""
        try:
            import torch
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            torch.save(model.state_dict(), path)
        except Exception as e:
            raise RuntimeError(f"Could not save checkpoint to {path}: {e}")

    @staticmethod
    def save_final(model, path: str) -> None:
        BaseTrainer.save_checkpoint(model, path)

    def save_report(self, report: Dict[str, Any], path: str) -> None:
        self._dump_json(report, path)

    @staticmethod
    def save_curve_csv(rows: Iterable[Tuple[int, float, float, float]], path: str) -> None:
        """
        rows: iterable of (update_idx, train_loss, val_loss, lr)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["update", "train_loss", "val_loss", "lr"])
            for r in rows:
                w.writerow(r)

    @staticmethod
    def save_schema_string(obj: Any, path: str) -> None:
        """
        Save compact JSON (no spaces) to be consumed downstream as a String param.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))

    # ------- Model/code loading utilities (for model_py_in) -------

    @staticmethod
    def find_model_py_file(path_or_dir: str) -> str:
        """
        If a file is given, return it. If a dir is given, pick the first .py file.
        """
        if os.path.isfile(path_or_dir):
            return path_or_dir
        if not os.path.isdir(path_or_dir):
            raise ValueError(f"Path {path_or_dir} is neither a file nor a directory")
        py_files = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith(".py")]
        if not py_files:
            raise FileNotFoundError(f"No .py file found in directory {path_or_dir}")
        return py_files[0]

    @staticmethod
    def load_model_class_from_file(py_path: str, class_name: str) -> Any:
        """
        Dynamically import a class from a python file path.
        """
        if not py_path.endswith(".py"):
            # tolerate paths missing extension (create a temp copy with .py)
            tmp = py_path + ".py"
            import shutil as _sh
            _sh.copyfile(py_path, tmp)
            py_path = tmp
        spec = importlib.util.spec_from_file_location("lmf_model_module", py_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load spec from {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, class_name):
            raise AttributeError(f"{py_path} does not define {class_name}")
        return getattr(mod, class_name)

    # ------- Misc small helpers -------

    # @staticmethod
    # def count_learnable_parameters(model: Any) -> int:
    #     """Return learnable parameter count if possible; else -1."""
    #     try:
    #         total = 0
    #         for p in model.parameters():
    #             if getattr(p, "requires_grad", False):
    #                 total += int(p.numel())
    #         return total
    #     except Exception:
    #         return -1

    @staticmethod
    def validate_min_lr(lr: float, min_lr: float) -> None:
        if min_lr > lr:
            raise ValueError(f"min_lr ({min_lr}) must be <= learning_rate ({lr})")

