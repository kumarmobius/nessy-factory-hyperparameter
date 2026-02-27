# language_model/trainer_gemma.py
from __future__ import annotations

import os, json, math, shutil, importlib.util, csv
from typing import Any, Dict
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from .base import BaseTrainer
from .registry import register

def _find_model_py_file(path_or_dir: str) -> str:
    if os.path.isfile(path_or_dir):
        return path_or_dir
    if not os.path.isdir(path_or_dir):
        raise ValueError(f"Path {path_or_dir} is neither a file nor a directory")
    py_files = [os.path.join(path_or_dir, f) for f in os.listdir(path_or_dir) if f.endswith(".py")]
    if not py_files:
        raise FileNotFoundError(f"No .py file found in directory {path_or_dir}")
    return py_files[0]

def _load_model_class_from_file(py_path: str, class_name: str = "Gemma3Model"):
    if not py_path.endswith(".py"):
        tmp = py_path + ".py"
        shutil.copyfile(py_path, tmp)
        py_path = tmp
    spec = importlib.util.spec_from_file_location("gemma3_model", py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, class_name):
        raise AttributeError(f"{py_path} does not define {class_name}")
    return getattr(mod, class_name)

def _choose_bin_dtype(vocab_size: int):
    if vocab_size <= (1 << 16): return np.uint16
    if vocab_size <= (1 << 32): return np.uint32
    return np.uint64

def _tokenize_to_bins(tokenizer_path: str, train_txt: str, val_fraction: float, num_proc: int) -> Dict[str, Any]:
    tok = Tokenizer.from_file(tokenizer_path)
    vocab_size = tok.get_vocab_size()
    BIN_DTYPE = _choose_bin_dtype(vocab_size)

    ds_single = load_dataset("text", data_files={"train": train_txt})
    split = ds_single["train"].train_test_split(test_size=float(val_fraction), seed=42)
    ds = {"train": split["train"], "val": split["test"]}

    def proc(ex):
        ids = tok.encode(ex["text"]).ids
        return {"ids": ids, "len": len(ids)}

    nproc = max(1, min(int(num_proc), (os.cpu_count() or 1)))
    tokenized = {sp: d.map(proc, remove_columns=d.column_names, desc=f"tokenizing {sp}", num_proc=nproc)
                 for sp, d in ds.items()}

    def write_split(dset, filename):
        total = int(np.sum(dset["len"], dtype=np.uint64))
        if total == 0: raise ValueError(f"{filename}: no tokens to write")
        arr = np.memmap(filename, dtype=BIN_DTYPE, mode="w+", shape=(total,))
        shard_size = max(1, len(dset) // 1024)
        total_shards = max(1, (len(dset) + shard_size - 1) // shard_size)
        idx = 0
        for i in tqdm(range(total_shards), desc=f"writing {filename}"):
            start, stop = i * shard_size, min(len(dset), (i + 1) * shard_size)
            if start >= stop: continue
            batch = dset.select(range(start, stop)).with_format(type="numpy")
            ids_list = batch["ids"]
            if not ids_list: continue
            arr_batch = np.concatenate([np.asarray(x, dtype=BIN_DTYPE) for x in ids_list])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        return total

    train_tokens = write_split(tokenized["train"], "train.bin")
    val_tokens   = write_split(tokenized["val"],   "val.bin")

    meta = {"vocab_size": vocab_size, "train_tokens": int(train_tokens), "val_tokens": int(val_tokens),
            "bin_dtype": str(BIN_DTYPE.__name__)}
    with open("bin_meta.json", "w", encoding="utf-8") as f: json.dump(meta, f, indent=2)
    return meta

def _make_get_batch(device, device_type, block_size: int, batch_size: int):
    with open("bin_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    dtype_map = {"uint16": np.uint16, "uint32": np.uint32, "uint64": np.uint64}
    BIN_DTYPE = dtype_map[meta["bin_dtype"]]
    TRAIN_BIN, VAL_BIN = "train.bin", "val.bin"

    def get_batch(split: str):
        path = TRAIN_BIN if split == "train" else VAL_BIN
        data = np.memmap(path, dtype=BIN_DTYPE, mode="r")
        if len(data) <= block_size:
            raise ValueError(f"{path}: not enough tokens for block_size={block_size}")
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(np.asarray(data[i:i+block_size], dtype=np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(np.asarray(data[i+1:i+block_size+1], dtype=np.int64)) for i in ix])
        if device_type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True); y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y
    return get_batch

@register("gemma_trainer")
class GemmaTrainer(BaseTrainer):
    """
    Registered trainer that follows your YAML behavior.
    Call with `create("gemma_trainer").run(**kwargs)` or import and use directly.
    """

    def run(
        self,
        tokenizer_json: str,
        train_corpus: str,
        model_config: str,
        model_weights: str,
        model_py_in: str,
        model_py_out: str,
        learning_rate: float,
        min_lr: float,
        warmup_steps: int,
        max_iters: int,
        batch_size: int,
        block_size: int,
        grad_accum: int,
        eval_interval: int,
        eval_iters: int,
        weight_decay: float,
        beta2: float,
        clip_grad_norm: float,
        val_fraction: float,
        num_proc: int,
        best_weights: str,
        final_weights: str,
        training_report: str,
        loss_curve_csv: str,
    ) -> Dict[str, Any]:

        # 1) device + AMP
        device, device_type = self.pick_device()
        print(f"[INFO] Training on {device.upper()}")
        ctx, scaler, _dtype_name = self.autocast_context(device_type)

        # 2) model code + config + weights
        model_py_path = _find_model_py_file(model_py_in)
        os.makedirs(os.path.dirname(model_py_out) or ".", exist_ok=True)
        shutil.copyfile(model_py_path, model_py_out)

        with open(model_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["dtype"] = torch.float16 if str(cfg.get("dtype")) == "float16" else torch.float32

        self.validate_min_lr(learning_rate, min_lr)
        if block_size > int(cfg["context_length"]):
            raise ValueError(f"block_size ({block_size}) must be <= context_length ({cfg['context_length']})")
        if "sliding_attention" in cfg.get("layer_types", []):
            if block_size > int(cfg["sliding_window"]):
                raise ValueError(f"block_size ({block_size}) must be <= sliding_window ({cfg['sliding_window']}) for sliding layers")

        tok = Tokenizer.from_file(tokenizer_json)
        if tok.get_vocab_size() != int(cfg["vocab_size"]):
            print(f"[WARN] tokenizer vocab {tok.get_vocab_size()} != model config vocab {cfg['vocab_size']}")

        Gemma3Model = _load_model_class_from_file(model_py_path, "Gemma3Model")
        model = Gemma3Model(cfg).to(device)
        state = torch.load(model_weights, map_location=device)
        model.load_state_dict(state, strict=True)
        model.train()

        total_params = self.count_learnable_parameters(model)
        print(f"[INFO] Trainable parameters: {total_params:,}")

        # 3) data → memmaps + batch fn
        meta = _tokenize_to_bins(tokenizer_json, train_corpus, val_fraction, num_proc)
        get_batch = _make_get_batch(device, device_type, block_size, batch_size)

        # 4) optimizer + warmup→cosine scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, beta2),
                                      weight_decay=weight_decay, eps=1e-9)
        total_updates = math.ceil(max_iters / max(1, grad_accum))
        scheduler = self.scheduler_warmup_cosine(optimizer, total_updates, warmup_steps, min_lr)

        # 5) training loop with periodic eval + checkpoints
        best_val = float("inf")
        train_curve, val_curve, lr_curve = [], [], []
        updates = 0

        pbar = tqdm(range(max_iters), desc="train micro-steps")
        for step in pbar:
            X, Y = get_batch("train")
            with ctx:
                _, loss = model(X, Y)
                (loss / grad_accum).backward()

            if ((step + 1) % grad_accum == 0) or (step + 1 == max_iters):
                self.clip_grad_norm(model.parameters(), clip_grad_norm)
                if scaler.is_enabled():
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                updates += 1

                if updates % max(1, eval_interval) == 0:
                    losses = self.estimate_loss_loop(model, get_batch, eval_iters, ctx)
                    train_curve.append(losses["train"])
                    val_curve.append(losses["val"])
                    lr_curve.append(optimizer.param_groups[0]["lr"])
                    pbar.set_postfix(train=f"{losses['train']:.4f}",
                                     val=f"{losses['val']:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}")
                    if losses["val"] < best_val:
                        best_val = losses["val"]
                        self.save_checkpoint(model, best_weights)

        self.save_final(model, final_weights)

        # 6) artifacts: training report + loss CSV
        report = {
            "best_val_loss": best_val,
            "final_lr": optimizer.param_groups[0]["lr"],
            "updates": updates,
            "max_iters": max_iters,
            "batch_size": batch_size,
            "block_size": block_size,
            "grad_accum": grad_accum,
            "train_tokens": meta.get("train_tokens"),
            "val_tokens": meta.get("val_tokens"),
            "total_updates": total_updates,
            "warmup_updates": min(warmup_steps, total_updates),
            "total_params": total_params,
        }
        self.save_report(report, training_report)

        os.makedirs(os.path.dirname(loss_curve_csv) or ".", exist_ok=True)
        with open(loss_curve_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["update","train_loss","val_loss","lr"])
            for i,(tr,va,lr) in enumerate(zip(train_curve, val_curve, lr_curve), start=1):
                w.writerow([i, tr, va, lr])

        return report
