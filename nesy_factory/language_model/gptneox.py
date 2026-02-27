# language_model/gptneox.py
from __future__ import annotations
from typing import Any, Dict
import os
import json

from .base import BaseModelBuilder
from .registry import register


MODEL_PY = r'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)

def apply_rotary_pos_emb(q, k, sin, cos):
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot

def make_rotary_sin_cos(seq_len, rotary_dim, device, dtype=torch.float32):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.sin(), emb.cos()

class NeoXAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, rotary_pct=0.25, dropout=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rotary_pct = rotary_pct
        self.rotary_dim = int(self.head_dim * rotary_pct)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1

        self.query_key_value = nn.Linear(hidden_size, self.head_dim * num_heads * 3, bias=True)
        self.dense = nn.Linear(self.head_dim * num_heads, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sin=None, cos=None, attn_mask=None, past_kv=None):
        B, T, H = x.shape
        qkv = self.query_key_value(x)  
        qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        if self.rotary_dim > 0 and sin is not None and cos is not None:
            q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
            k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
            sin_v = sin.view(1, 1, T, self.rotary_dim)
            cos_v = cos.view(1, 1, T, self.rotary_dim)
            q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, sin_v, cos_v)
            q = torch.cat((q_rot, q_pass), dim=-1)
            k = torch.cat((k_rot, k_pass), dim=-1)

        if past_kv is not None:
            past_k, past_v = past_kv

            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v)


        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale 

        q_len, k_len = q.size(2), k.size(2)
        if attn_mask is None:

            past_len = k_len - q_len if k_len >= q_len else 0

            idx_q = torch.arange(q_len, device=x.device)
            idx_k = torch.arange(k_len, device=x.device)
            mask = idx_k.unsqueeze(0) <= (past_len + idx_q.unsqueeze(1)) 
            attn_mask = mask.unsqueeze(0).unsqueeze(0)  

        attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(B, q_len, -1)
        out = self.dense(out)
        return out, present_kv

class NeoXMLP(nn.Module):
    def __init__(self, hidden_size, ff_mult=4, dropout=0.0):
        super().__init__()
        inner = hidden_size * ff_mult
        self.dense_h_to_4h = nn.Linear(hidden_size, inner, bias=True)
        self.dense_4h_to_h = nn.Linear(inner, hidden_size, bias=True)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dense_h_to_4h(x)
        x = self.act(x)
        x = self.dense_4h_to_h(x)
        x = self.dropout(x)
        return x

class NeoXLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, rotary_pct=0.25, ff_mult=4, dropout=0.0, use_parallel_residual=False):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)

        self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.attention = NeoXAttention(hidden_size, num_heads, rotary_pct=rotary_pct, dropout=dropout)
        self.mlp = NeoXMLP(hidden_size, ff_mult=ff_mult, dropout=dropout)
        self.use_parallel_residual = use_parallel_residual

    def forward(self, x, sin=None, cos=None, attn_mask=None, past_kv=None):
        if self.use_parallel_residual:
            norm_x = self.input_layernorm(x)
            attn_out, present_kv = self.attention(norm_x, sin=sin, cos=cos, attn_mask=attn_mask, past_kv=past_kv)
            ff_out = self.mlp(norm_x)
            x = x + attn_out + ff_out
            return x, present_kv
        else:
            norm_x = self.input_layernorm(x)
            attn_out, present_kv = self.attention(norm_x, sin=sin, cos=cos, attn_mask=attn_mask, past_kv=past_kv)
            x = x + attn_out
            norm_x2 = self.post_attention_layernorm(x)
            ff_out = self.mlp(norm_x2)
            x = x + ff_out
            return x, present_kv

class GPTNeoXForCausalLM(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        vocab_size = cfg["vocab_size"]
        hidden_size = cfg["hidden_size"]
        num_layers = cfg["num_layers"]
        num_heads = cfg["num_heads"]
        ff_mult = cfg.get("ff_mult", 4)
        rotary_pct = cfg.get("rotary_pct", 0.25)
        max_seq_len = cfg.get("max_seq_len", 512)
        dropout = cfg.get("dropout", 0.0)
        use_parallel_residual = cfg.get("use_parallel_residual", False)
        tie_word_embeddings = cfg.get("tie_word_embeddings", True)

      
        self.gpt_neox = nn.Module()
        self.gpt_neox.embed_in = nn.Embedding(vocab_size, hidden_size)
        self.gpt_neox.layers = nn.ModuleList([
            NeoXLayer(hidden_size, num_heads, rotary_pct=rotary_pct, ff_mult=ff_mult, dropout=dropout,
                     use_parallel_residual=use_parallel_residual)
            for _ in range(num_layers)
        ])
        self.gpt_neox.final_layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)

        self.gpt_neox.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:

            self.gpt_neox.embed_out.weight = self.gpt_neox.embed_in.weight

        rotary_dim = int((hidden_size // num_heads) * rotary_pct)
        if rotary_dim > 0:
            sin, cos = make_rotary_sin_cos(max_seq_len, rotary_dim, device="cpu", dtype=torch.float32)
            self.register_buffer("sin_cache", sin, persistent=False)
            self.register_buffer("cos_cache", cos, persistent=False)
        else:
            self.sin_cache = None
            self.cos_cache = None

    def forward(self, input_ids, past_kvs=None):
        B, T = input_ids.shape
        x = self.gpt_neox.embed_in(input_ids)
        past_len = past_kvs[0][0].size(2) if past_kvs is not None else 0
        if getattr(self, "sin_cache", None) is not None:
            sin = self.sin_cache[past_len: past_len + T].to(x.device).to(x.dtype)
            cos = self.cos_cache[past_len: past_len + T].to(x.device).to(x.dtype)
        else:
            sin = cos = None

        new_kvs = []
        attn_mask = None 
        for i, layer in enumerate(self.gpt_neox.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, present_kv = layer(x, sin=sin, cos=cos, attn_mask=attn_mask, past_kv=past_kv)
            new_kvs.append(present_kv)

        x = self.gpt_neox.final_layer_norm(x)
        logits = self.gpt_neox.embed_out(x)
        return logits, new_kvs
        
class Gemma3Model(GPTNeoXForCausalLM):
            def forward(self, input_ids, labels=None):
                logits, _ = super().forward(input_ids)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                return (logits, loss)
'''

@register("gptneox_builder")
class GPTNeoXBuilder(BaseModelBuilder):
    MODEL_PY = MODEL_PY
    def run(
        self,
        tokenizer_json: str,
        n_layers: int,
        layer_pattern: str,
        model_weights_out: str,
        model_config_out: str,
        model_py_out: str,
    ) -> Dict[str, Any]:

        from tokenizers import Tokenizer
        import torch
        import torch.nn as nn
        import os
        from textwrap import dedent
        import json
        import math
        from typing import Any, Dict

        tok = Tokenizer.from_file(tokenizer_json)
        vocab_size = int(tok.get_vocab_size())


        layer_types = self.pattern_to_layers(layer_pattern, int(n_layers)) if layer_pattern else ["global_attention"] * int(n_layers)


        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_dtype = torch.float16 if device == "cuda" else torch.float32

        hidden_size = 256
        num_heads = 8
        ff_mult = 4
        max_seq = 3500

        cfg = {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_heads": num_heads,
            "num_layers": int(n_layers),
            "ff_mult": ff_mult,
            "rotary_pct": 0.25,
            "max_seq_len": max_seq,
            "context_length": max_seq, 
            "dropout": 0.0,
            "use_parallel_residual": True,
            "tie_word_embeddings": True,   
            "dtype": "float16" if model_dtype == torch.float16 else "float32",
        }

        env = {}
        exec(dedent(self.MODEL_PY), env, env)
        GPTClass = env["GPTNeoXForCausalLM"]

        model_cfg = {
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "num_layers": cfg["num_layers"],
            "num_heads": cfg["num_heads"],
            "context_length": max_seq,
            "ff_mult": cfg["ff_mult"],
            "rotary_pct": cfg["rotary_pct"],
            "max_seq_len": cfg["max_seq_len"],
            "dropout": cfg["dropout"],
            "use_parallel_residual": cfg["use_parallel_residual"],
            "tie_word_embeddings": cfg["tie_word_embeddings"],
        }
        model = GPTClass(model_cfg).to(device)

        # cast dtype
        if cfg["dtype"] == "float16":
            model.half()

        if getattr(model, "sin_cache", None) is not None:
            model.sin_cache = model.sin_cache.to(device)
            model.cos_cache = model.cos_cache.to(device)

        model.eval()

        os.makedirs(os.path.dirname(model_weights_out) or ".", exist_ok=True)
        torch.save(model.state_dict(), model_weights_out)


        state_check = torch.load(model_weights_out, map_location="cpu")

        self.save_config(cfg, model_config_out)


        model_code = dedent(self.MODEL_PY).lstrip("\n").replace("\x00", "")
        self.save_code(model_code, model_py_out)
        total_params = self.count_learnable_parameters(model)
        return {
            "vocab_size": cfg["vocab_size"],
            "n_layers": cfg["num_layers"],
            "layer_pattern": layer_pattern,
            "total_learnable_parameters": int(total_params),
            "weights_path": model_weights_out,
            "config_path": model_config_out,
            "model_py_path": model_py_out,
        }


# language_model/gptneox.py
# from __future__ import annotations
# from typing import Any, Dict
# import os
# import json

# from .base import BaseModelBuilder
# from .registry import register

# MODEL_PY = r'''
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Any, Dict

# def rotate_half(x):
#     orig_shape = x.shape
#     rotary_dim = orig_shape[-1]
#     x = x.view(*orig_shape[:-1], rotary_dim // 2, 2)
#     x1 = x[..., 0]
#     x2 = x[..., 1]
#     out = torch.stack((-x2, x1), dim=-1).reshape_as(torch.empty(0, device=x.device, dtype=x.dtype).new_empty(orig_shape))
#     return out

# def apply_rotary_pos_emb(q, k, sin, cos):
#     q_rot = (q * cos) + (rotate_half(q) * sin)
#     k_rot = (k * cos) + (rotate_half(k) * sin)
#     return q_rot, k_rot

# def make_rotary_sin_cos(seq_len, rotary_dim, device=torch.device("cpu"), dtype=torch.float32):
#     assert rotary_dim % 2 == 0, "rotary_dim must be even"
#     inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
#     positions = torch.arange(seq_len, device=device, dtype=dtype)
#     freqs = torch.einsum("i,j->ij", positions, inv_freq) 
#     emb = torch.cat((freqs, freqs), dim=-1) 
#     return emb.sin(), emb.cos()

# class NeoXAttention(nn.Module):
#     def __init__(self, hidden_size, num_heads, rotary_pct=0.25, dropout=0.0):
#         super().__init__()
#         assert hidden_size % num_heads == 0
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         self.rotary_pct = rotary_pct
#         self.rotary_dim = int(self.head_dim * rotary_pct)
#         if self.rotary_dim % 2 != 0:
#             self.rotary_dim -= 1
#         self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
#         self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)
#         self.dropout = nn.Dropout(dropout)

#     def _split_heads(self, x):
#         B, T, H = x.shape
#         x = x.view(B, T, self.num_heads, self.head_dim)

#         return x.permute(0, 2, 1, 3)

#     def forward(self, x, sin=None, cos=None, attn_mask=None, past_kv=None):

#         B, T, H = x.shape
#         qkv = self.qkv_proj(x)
#         qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2] 

#         if self.rotary_dim > 0 and sin is not None and cos is not None:
#             sin_v = sin[:T].to(q.dtype).to(q.device)
#             cos_v = cos[:T].to(q.dtype).to(q.device)
#             sin_v = sin_v.view(1, 1, T, self.rotary_dim)
#             cos_v = cos_v.view(1, 1, T, self.rotary_dim)

#             q_rot, q_pass = q[..., :self.rotary_dim], q[..., self.rotary_dim:]
#             k_rot, k_pass = k[..., :self.rotary_dim], k[..., self.rotary_dim:]
#             q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, sin_v, cos_v)
#             q = torch.cat((q_rot, q_pass), dim=-1)
#             k = torch.cat((k_rot, k_pass), dim=-1)

#         if past_kv is not None:
#             past_k, past_v = past_kv

#             k = torch.cat([past_k, k], dim=2)
#             v = torch.cat([past_v, v], dim=2)

#         present_kv = (k, v)

#         scale = 1.0 / math.sqrt(self.head_dim)
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale 

#         if attn_mask is None:
#             q_len = q.size(2)
#             k_len = k.size(2)
#             device = q.device
#             mask = torch.tril(torch.ones((q_len, k_len), device=device, dtype=torch.bool))
#             attn_mask = mask.unsqueeze(0).unsqueeze(0) 


#         attn_scores = attn_scores.masked_fill(~attn_mask, float("-inf"))
#         attn_probs = torch.softmax(attn_scores, dim=-1)
#         attn_probs = self.dropout(attn_probs)
#         out = torch.matmul(attn_probs, v)
#         out = out.permute(0, 2, 1, 3).contiguous().view(B, T, -1) 
#         out = self.out_proj(out)
#         return out, present_kv

# class NeoXMLP(nn.Module):
#     def __init__(self, hidden_size, ff_mult=4, dropout=0.0):
#         super().__init__()
#         inner = hidden_size * ff_mult
#         self.dense_h_to_4h = nn.Linear(hidden_size, inner, bias=True)
#         self.act = nn.GELU()
#         self.dense_4h_to_h = nn.Linear(inner, hidden_size, bias=True)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.dense_h_to_4h(x)
#         x = self.act(x)
#         x = self.dense_4h_to_h(x)
#         x = self.dropout(x)
#         return x

# class NeoXLayer(nn.Module):
#     def __init__(self, hidden_size, num_heads, rotary_pct=0.25, ff_mult=4, dropout=0.0):
#         super().__init__()
#         self.input_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
#         self.post_attention_layernorm = nn.LayerNorm(hidden_size, elementwise_affine=True)
#         self.attention = NeoXAttention(hidden_size, num_heads, rotary_pct=rotary_pct, dropout=dropout)
#         self.mlp = NeoXMLP(hidden_size, ff_mult=ff_mult, dropout=dropout)

#     def forward(self, x, sin=None, cos=None, attn_mask=None, past_kv=None):
#         norm_x = self.input_layernorm(x)
#         attn_out, present_kv = self.attention(norm_x, sin=sin, cos=cos, attn_mask=attn_mask, past_kv=past_kv)
#         ff_out = self.mlp(norm_x)
#         x = x + attn_out + ff_out
#         return x, present_kv

# class GPTNeoXForCausalLM(nn.Module):
#     def __init__(self, cfg: Dict[str, Any]):
#         super().__init__()
#         vocab_size = cfg["vocab_size"]
#         hidden_size = cfg["hidden_size"]
#         num_layers = cfg["num_layers"]
#         num_heads = cfg["num_heads"]
#         ff_mult = cfg.get("ff_mult", 4)
#         rotary_pct = cfg.get("rotary_pct", 0.25)
#         max_seq_len = cfg.get("max_seq_len", 512)
#         dropout = cfg.get("dropout", 0.0)
#         tie_word_embeddings = cfg.get("tie_word_embeddings", True)

#         self.embed_in = nn.Embedding(vocab_size, hidden_size)
#         self.layers = nn.ModuleList([
#             NeoXLayer(hidden_size, num_heads, rotary_pct=rotary_pct, ff_mult=ff_mult, dropout=dropout)
#             for _ in range(num_layers)
#         ])
#         self.final_layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
#         self.embed_out = nn.Linear(hidden_size, vocab_size, bias=False)

#         if tie_word_embeddings:
#             self.embed_out.weight = self.embed_in.weight

#         rotary_dim = int((hidden_size // num_heads) * rotary_pct)
#         if rotary_dim > 0:
#             sin, cos = make_rotary_sin_cos(max_seq_len, rotary_dim, device=torch.device("cpu"), dtype=torch.float32)
#             self.register_buffer("sin_cache", sin)
#             self.register_buffer("cos_cache", cos)
#         else:
#             self.sin_cache = None
#             self.cos_cache = None

#     def forward(self, input_ids, past_kvs=None):
#         B, T = input_ids.shape
#         x = self.embed_in(input_ids)
#         past_len = 0
#         if past_kvs is not None:
#             past_len = past_kvs[0][0].size(2)

#         if getattr(self, "sin_cache", None) is not None:
#             sin = self.sin_cache[past_len: past_len + T].to(x.device).to(x.dtype)
#             cos = self.cos_cache[past_len: past_len + T].to(x.device).to(x.dtype)
#         else:
#             sin = cos = None

#         new_kvs = []
#         attn_mask = None
#         for i, layer in enumerate(self.layers):
#             past_kv = past_kvs[i] if past_kvs is not None else None
#             x, present_kv = layer(x, sin=sin, cos=cos, attn_mask=attn_mask, past_kv=past_kv)
#             new_kvs.append(present_kv)

#         x = self.final_layer_norm(x)
#         logits = self.embed_out(x)
#         return logits, new_kvs

# class Gemma3Model(GPTNeoXForCausalLM):
#     def forward(self, input_ids, labels=None, past_kvs=None):
#         logits, new_kvs = super().forward(input_ids, past_kvs=past_kvs)
#         loss = None
#         if labels is not None:
#             shift_logits = logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             loss = F.cross_entropy(
#                 shift_logits.view(-1, shift_logits.size(-1)),
#                 shift_labels.view(-1),
#                 ignore_index=-100,
#             )
#         return logits, loss
# '''

# @register("gptneox_builder")
# class GPTNeoXBuilder(BaseModelBuilder):
#     MODEL_PY = MODEL_PY
#     def run(
#         self,
#         tokenizer_json: str,
#         n_layers: int,
#         layer_pattern: str,
#         model_weights_out: str,
#         model_config_out: str,
#         model_py_out: str,
#     ) -> Dict[str, Any]:

#         from tokenizers import Tokenizer
#         import torch
#         import torch.nn as nn
#         import os
#         from textwrap import dedent
#         import json
#         import math
#         from typing import Any, Dict

#         tok = Tokenizer.from_file(tokenizer_json)
#         vocab_size = int(tok.get_vocab_size())

#         layer_types = self.pattern_to_layers(layer_pattern, int(n_layers)) if layer_pattern else ["global_attention"] * int(n_layers)

#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model_dtype = torch.float16 if device == "cuda" else torch.float32

#         hidden_size = 256
#         num_heads = 8
#         ff_mult = 4
#         max_seq = 3500
        

#         cfg = {
#             "vocab_size": vocab_size,
#             "hidden_size": hidden_size,
#             "num_heads": num_heads,
#             "num_layers": int(n_layers),
#             "ff_mult": ff_mult,
#             "rotary_pct": 0.25,
#             "max_seq_len": max_seq,
#             "context_length": max_seq,
#             "dropout": 0.0,
#             "tie_word_embeddings": True,
#             "dtype": "float16" if model_dtype == torch.float16 else "float32",
#         }

#         env = {}
#         exec(dedent(self.MODEL_PY), env, env)
#         GPTClass = env["GPTNeoXForCausalLM"]

#         model_cfg = {
#             "vocab_size": cfg["vocab_size"],
#             "hidden_size": cfg["hidden_size"],
#             "num_layers": cfg["num_layers"],
#             "num_heads": cfg["num_heads"],
#             "context_length": max_seq,
#             "ff_mult": cfg["ff_mult"],
#             "rotary_pct": cfg["rotary_pct"],
#             "max_seq_len": cfg["max_seq_len"],
#             "dropout": cfg["dropout"],
#             "tie_word_embeddings": cfg["tie_word_embeddings"],
#         }
#         model = GPTClass(model_cfg).to(device)
        
#         def neox_init_weights(model: nn.Module, num_layers: int, hidden_size: int):

#             small_std = math.sqrt(2.0 / (5.0 * float(hidden_size)))


#             wang_std = 2.0 / (float(num_layers) * math.sqrt(float(hidden_size)))

#             for name, module in model.named_modules():
#                 if isinstance(module, nn.Linear):
#                     if "dense_4h_to_h" in name or name.endswith("dense_4h_to_h"):
#                         nn.init.normal_(module.weight, mean=0.0, std=wang_std)
#                         if module.bias is not None:
#                             nn.init.zeros_(module.bias)
#                     else:
#                         nn.init.normal_(module.weight, mean=0.0, std=small_std)
#                         if module.bias is not None:
#                             nn.init.zeros_(module.bias)

#         neox_init_weights(model, cfg["num_layers"], cfg["hidden_size"])
        
#         if cfg["dtype"] == "float16":
#             model.half()


#         if getattr(model, "sin_cache", None) is not None:
#             model.sin_cache = model.sin_cache.to(device).to(next(model.parameters()).dtype)
#             model.cos_cache = model.cos_cache.to(device).to(next(model.parameters()).dtype)

#         model.eval()

#         os.makedirs(os.path.dirname(model_weights_out) or ".", exist_ok=True)
#         torch.save(model.state_dict(), model_weights_out)

#         state_check = torch.load(model_weights_out, map_location="cpu")

#         self.save_config(cfg, model_config_out)

#         model_code = dedent(self.MODEL_PY).lstrip("\n").replace("\x00", "")
#         self.save_code(model_code, model_py_out)
#         total_params = self.count_learnable_parameters(model)
#         return {
#             "vocab_size": cfg["vocab_size"],
#             "n_layers": cfg["num_layers"],
#             "layer_pattern": layer_pattern,
#             "total_learnable_parameters": int(total_params),
#             "weights_path": model_weights_out,
#             "config_path": model_config_out,
#             "model_py_path": model_py_out,
#         }