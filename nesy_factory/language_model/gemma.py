# language_model/gemma.py
from __future__ import annotations

from typing import Any, Dict
import os
import json

from .base import BaseModelBuilder
from .registry import register


@register("gemma3_builder")
class Gemma3Builder(BaseModelBuilder):
    """
    Build an untrained Gemma3 transformer from a tokenizer and a layer pattern.
    Saves: model weights (.pt), model config (JSON), reusable model code (.py).
    No schema file is produced.
    """

    # ------------------ reusable model code blob ------------------
    MODEL_PY = r'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    angles = torch.cat((angles, angles), dim=1)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    *_, seq_len, head_dim = x.size()
    x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
    cos_seq = cos[:seq_len, :].to(x.dtype).unsqueeze(0).unsqueeze(0)
    sin_seq = sin[:seq_len, :].to(x.dtype).unsqueeze(0).unsqueeze(0)
    rotated = torch.cat((-x2, x1), dim=-1)
    return (x * cos_seq) + (rotated * sin_seq)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
    
    def forward(self, x):
        orig_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        out = x_norm * (1.0 + self.scale.float())
        if self.shift is not None:
            out = out + self.shift.float()
        return out.to(orig_dtype)

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = head_dim if head_dim is not None else d_in // num_heads
        self.scaling = (query_pre_attn_scalar ** -0.5) if (query_pre_attn_scalar is not None) else (self.head_dim ** -0.5)
        
        self.W_query = nn.Linear(d_in, num_heads * self.head_dim, bias=False, dtype=dtype)
        self.W_key   = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj= nn.Linear(num_heads * self.head_dim, d_in, bias=False, dtype=dtype)
        
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
    
    def forward(self, x, cos, sin, mask=None):
        B, T, _ = x.shape
        q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_key(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.W_value(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
        
        if self.q_norm is not None: q = self.q_norm(q)
        if self.k_norm is not None: k = self.k_norm(k)
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        p = torch.softmax(attn, dim=-1)
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        emb_dim, hidden_dim, dtype = cfg["emb_dim"], cfg["hidden_dim"], cfg["dtype"]
        self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=dtype)
        self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=dtype)
    
    def forward(self, x):
        return self.fc3(F.gelu(self.fc1(x)) * self.fc2(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_type):
        super().__init__()
        self.layer_type = layer_type
        self.attn = GroupedQueryAttention(
            d_in=cfg["emb_dim"], num_heads=cfg["n_heads"], num_kv_groups=cfg.get("n_kv_groups", 1),
            head_dim=cfg["head_dim"], qk_norm=cfg.get("qk_norm", False),
            query_pre_attn_scalar=cfg.get("query_pre_attn_scalar", None), dtype=cfg["dtype"]
        )
        self.input_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.post_attn_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.pre_ff_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.post_ff_norm = RMSNorm(cfg["emb_dim"], eps=cfg.get("rms_norm_eps", 1e-6))
        self.ffn = FeedForward(cfg)
    
    def forward(self, x, cos, sin, mask=None):
        residual = x
        x = self.input_norm(x)
        x = self.post_attn_norm(self.attn(x, cos, sin, mask) + residual)
        residual = x
        x = self.pre_ff_norm(x)
        x = self.post_ff_norm(self.ffn(x) + residual)
        return x

class Gemma3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        vocab_size, emb_dim = cfg["vocab_size"], cfg["emb_dim"]
        
        self.token_emb = nn.Embedding(vocab_size, emb_dim, dtype=cfg["dtype"])
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg, layer_type=cfg["layer_types"][i]) 
            for i in range(cfg["n_layers"])
        ])
        self.final_norm = RMSNorm(emb_dim, eps=cfg.get("rms_norm_eps", 1e-6))
        self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=cfg["dtype"])
        
        # FIX: Register RoPE parameters as buffers so they get converted with .half()
        cos_global, sin_global = compute_rope_params(
            cfg["head_dim"], cfg["rope_base"], cfg["context_length"], torch.float32
        )
        cos_local, sin_local = compute_rope_params(
            cfg["head_dim"], cfg["rope_local_base"], cfg["sliding_window"], torch.float32
        )
        
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
    
    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor = None):
        # FIX: Don't force dtype - let embeddings follow the model's current dtype
        x = self.token_emb(input_ids)
        # The embedding layer will automatically output the correct dtype based on its weights
        
        B, T, _ = x.size()
        device = x.device
        
        # Buffers are already on the correct device, but ensure anyway
        mask_full = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
        
        window = self.cfg["sliding_window"]
        mask_sliding = torch.zeros((T, T), dtype=torch.bool, device=device)
        for i in range(T):
            start = max(0, i - window + 1)
            mask_sliding[i, start:i+1] = True
        
        for i, block in enumerate(self.blocks):
            if self.cfg["layer_types"][i] == "sliding_attention":
                x = block(x, self.cos_local, self.sin_local, mask_sliding)
            else:
                x = block(x, self.cos_global, self.sin_global, mask_full)
        
        x = self.final_norm(x)
        logits = self.out_head(x)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
        
        return logits, loss
'''

    # ------------------ builder entrypoint ------------------
    def run(
        self,
        tokenizer_json: str,
        n_layers: int,
        layer_pattern: str,
        model_weights_out: str,
        model_config_out: str,
        model_py_out: str,
    ) -> Dict[str, Any]:
        # Lazy imports so importing the package doesn’t hard-require heavy deps
        from tokenizers import Tokenizer
        import torch
        import torch.nn as nn
        import torch.nn.functional as F  # noqa: F401

        # 1) read tokenizer vocab
        tok = Tokenizer.from_file(tokenizer_json)
        vocab_size = int(tok.get_vocab_size())

        # 2) pattern -> layer types
        layer_types = self.pattern_to_layers(layer_pattern, int(n_layers))

        # 3) assemble config
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if device == "cuda" else torch.float32
        cfg = {
            "vocab_size": vocab_size,
            "context_length": 2048,
            "emb_dim": 640,
            "n_heads": 4,
            "n_layers": int(n_layers),
            "hidden_dim": 2048,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 1,
            "rope_local_base": 10000.0,
            "rope_base": 1000000.0,
            "sliding_window": 512,
            "layer_types": layer_types,
            "dtype": model_dtype,             # torch dtype in-memory
            "query_pre_attn_scalar": 128,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
        }

        # 4) instantiate the model (mirrors MODEL_PY) and init weights
        class RMSNorm(nn.Module):
            def __init__(self, emb_dim, eps=1e-6, bias=False):
                super().__init__()
                self.eps = eps
                self.scale = nn.Parameter(torch.zeros(emb_dim))
                self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None
            def forward(self, x):
                orig_dtype = x.dtype
                x_f = x.float()
                var = x_f.pow(2).mean(dim=-1, keepdim=True)
                x_norm = x_f * torch.rsqrt(var + self.eps)
                out = x_norm * (1.0 + self.scale.float())
                if self.shift is not None:
                    out = out + self.shift.float()
                return out.to(orig_dtype)

        def compute_rope_params(head_dim, theta_base=10_000.0, context_length=4096, dtype=torch.float32):
            assert head_dim % 2 == 0
            inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype) / head_dim))
            positions = torch.arange(context_length, dtype=dtype)
            angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
            angles = torch.cat((angles, angles), dim=1)
            return torch.cos(angles), torch.sin(angles)

        def apply_rope(x, cos, sin):
            *_, seq_len, head_dim = x.size()
            x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
            cos_seq = cos[:seq_len, :].to(x.dtype).unsqueeze(0).unsqueeze(0)
            sin_seq = sin[:seq_len, :].to(x.dtype).unsqueeze(0).unsqueeze(0)
            rotated = torch.cat((-x2, x1), dim=-1)
            return (x * cos_seq) + (rotated * sin_seq)

        class GroupedQueryAttention(nn.Module):
            def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, query_pre_attn_scalar=None, dtype=None):
                super().__init__()
                assert num_heads % num_kv_groups == 0
                self.num_heads = num_heads
                self.num_kv_groups = num_kv_groups
                self.group_size = num_heads // num_kv_groups
                self.head_dim = head_dim if head_dim is not None else d_in // num_heads
                self.scaling = (query_pre_attn_scalar ** -0.5) if (query_pre_attn_scalar is not None) else (self.head_dim ** -0.5)
                self.W_query = nn.Linear(d_in, num_heads * self.head_dim, bias=False, dtype=model_dtype)
                self.W_key   = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=model_dtype)
                self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=model_dtype)
                self.out_proj= nn.Linear(num_heads * self.head_dim, d_in, bias=False, dtype=model_dtype)
                self.q_norm = RMSNorm(self.head_dim) if cfg.get("qk_norm", False) else None
                self.k_norm = RMSNorm(self.head_dim) if cfg.get("qk_norm", False) else None
            def forward(self, x, cos, sin, mask=None):
                B, T, _ = x.shape
                q = self.W_query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.W_key(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
                v = self.W_value(x).view(B, T, self.num_kv_groups, self.head_dim).transpose(1, 2)
                if self.q_norm is not None: q = self.q_norm(q)
                if self.k_norm is not None: k = self.k_norm(k)
                q = apply_rope(q, model.cos_global, model.sin_global)
                k = apply_rope(k, model.cos_global, model.sin_global)
                k = k.repeat_interleave(self.group_size, dim=1)
                v = v.repeat_interleave(self.group_size, dim=1)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_probs = torch.softmax(attn_scores, dim=-1)
                out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
                return self.out_proj(out)

        class FeedForward(nn.Module):
            def __init__(self, cfg_local):
                super().__init__()
                emb_dim, hidden_dim = cfg_local["emb_dim"], cfg_local["hidden_dim"]
                self.fc1 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=model_dtype)
                self.fc2 = nn.Linear(emb_dim, hidden_dim, bias=False, dtype=model_dtype)
                self.fc3 = nn.Linear(hidden_dim, emb_dim, bias=False, dtype=model_dtype)
            def forward(self, x):
                return self.fc3(F.gelu(self.fc1(x)) * self.fc2(x))

        class TransformerBlock(nn.Module):
            def __init__(self, cfg_local, layer_type):
                super().__init__()
                self.layer_type = layer_type
                self.attn = GroupedQueryAttention(cfg_local["emb_dim"], cfg_local["n_heads"], cfg_local.get("n_kv_groups", 1),
                                                  cfg_local["head_dim"], cfg_local.get("qk_norm", False),
                                                  cfg_local.get("query_pre_attn_scalar", None), model_dtype)
                self.input_norm = RMSNorm(cfg_local["emb_dim"], eps=cfg_local.get("rms_norm_eps", 1e-6))
                self.post_attn_norm = RMSNorm(cfg_local["emb_dim"], eps=cfg_local.get("rms_norm_eps", 1e-6))
                self.pre_ff_norm = RMSNorm(cfg_local["emb_dim"], eps=cfg_local.get("rms_norm_eps", 1e-6))
                self.post_ff_norm = RMSNorm(cfg_local["emb_dim"], eps=cfg_local.get("rms_norm_eps", 1e-6))
                self.ffn = FeedForward(cfg_local)
            def forward(self, x, cos, sin, mask=None):
                residual = x
                x = self.input_norm(x)
                attn = self.attn
                B, T, _ = x.shape
                q = attn.W_query(x).view(B, T, attn.num_heads, attn.head_dim).transpose(1, 2)
                k = attn.W_key(x).view(B, T, attn.num_kv_groups, attn.head_dim).transpose(1, 2)
                v = attn.W_value(x).view(B, T, attn.num_kv_groups, attn.head_dim).transpose(1, 2)
                if attn.q_norm is not None: q = attn.q_norm(q)
                if attn.k_norm is not None: k = attn.k_norm(k)
                q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
                k = k.repeat_interleave(attn.group_size, dim=1)
                v = v.repeat_interleave(attn.group_size, dim=1)
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scaling
                if mask is not None:
                    attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_probs = torch.softmax(attn_scores, dim=-1)
                out = torch.matmul(attn_probs, v).transpose(1, 2).contiguous().view(B, T, attn.num_heads * attn.head_dim)
                x = self.post_attn_norm(attn.out_proj(out) + residual)
                residual = x
                x = self.pre_ff_norm(x)
                x = self.post_ff_norm(self.ffn(x) + residual)
                return x

        class Gemma3Model(nn.Module):
            def __init__(self, cfg_local):
                super().__init__()
                self.cfg = cfg_local
                vocab_size, emb_dim = cfg_local["vocab_size"], cfg_local["emb_dim"]
                self.token_emb = nn.Embedding(vocab_size, emb_dim, dtype=model_dtype)
                self.blocks = nn.ModuleList([TransformerBlock(cfg_local, layer_type=layer) for layer in cfg_local["layer_types"]])
                self.final_norm = RMSNorm(emb_dim, eps=cfg_local.get("rms_norm_eps", 1e-6))
                self.out_head = nn.Linear(emb_dim, vocab_size, bias=False, dtype=model_dtype)
                self.cos_global, self.sin_global = compute_rope_params(cfg_local["head_dim"], cfg_local["rope_base"], cfg_local["context_length"], torch.float32)
                self.cos_local,  self.sin_local  = compute_rope_params(cfg_local["head_dim"], cfg_local["rope_local_base"], cfg_local["sliding_window"], torch.float32)
            def _ensure_rope_on_device(self, device):
                if self.cos_global.device != device:
                    self.cos_global = self.cos_global.to(device); self.sin_global = self.sin_global.to(device)
                if self.cos_local.device != device:
                    self.cos_local = self.cos_local.to(device); self.sin_local = self.sin_local.to(device)
            def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor = None):
                x = self.token_emb(input_ids).to(model_dtype)
                B, T, _ = x.size(); device = x.device
                self._ensure_rope_on_device(device)
                mask_full = torch.tril(torch.ones((T, T), dtype=torch.bool, device=device))
                window = self.cfg["sliding_window"]
                mask_sliding = torch.zeros((T, T), dtype=torch.bool, device=device)
                for i in range(T):
                    start = max(0, i - window + 1)
                    mask_sliding[i, start:i+1] = True
                for i, block in enumerate(self.blocks):
                    if self.cfg["layer_types"][i] == "sliding_attention":
                        x = block(x, self.cos_local, self.sin_local, mask_sliding)
                    else:
                        x = block(x, self.cos_global, self.sin_global, mask_full)
                x = self.final_norm(x)
                logits = self.out_head(x)
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="mean")
                return logits, loss

        model = Gemma3Model(cfg).to(device)
        # place rope tensors on device now (forward would also handle this)
        model.cos_global = model.cos_global.to(device)
        model.sin_global = model.sin_global.to(device)
        model.cos_local = model.cos_local.to(device)
        model.sin_local = model.sin_local.to(device)
        model.eval()

        # 5) save artifacts (weights, config, code)
        os.makedirs(os.path.dirname(model_weights_out) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_weights_out)

        cfg_json = {**cfg, "dtype": ("torch.float16" if cfg["dtype"] == torch.float16 else "torch.float32")}
        self.save_config(cfg_json, model_config_out)

        self.save_code(self.MODEL_PY, model_py_out)

        # Return a small summary (not written to disk)
        total_params = self.count_learnable_parameters(model)
        return {
            "vocab_size": cfg_json["vocab_size"],
            "n_layers": cfg_json["n_layers"],
            "layer_pattern": layer_pattern,
            "total_learnable_parameters": int(total_params),
            "weights_path": model_weights_out,
            "config_path": model_config_out,
            "model_py_path": model_py_out,
        }
