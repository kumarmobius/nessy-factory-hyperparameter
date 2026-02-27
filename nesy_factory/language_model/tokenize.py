# language_model/tokenizer.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from .base import BaseTokenizer
from .registry import register

@register("bpe_tokenizer")
class ByteLevelBPETokenizer(BaseTokenizer):
    """
    Byte-level BPE tokenizer built with Hugging Face `tokenizers`.

    Usage:
        tok = ByteLevelBPETokenizer()
        report = tok.run(
            text_file="data/train.txt",
            vocab_size=32000,
            min_frequency=2,
            special_tokens="[PAD],[UNK],[BOS],[EOS]",
            add_bos_eos=True
        )
        tok.save("artifacts/tokenizer.json")

        # later...
        tok2 = ByteLevelBPETokenizer()
        tok2.load("artifacts/tokenizer.json")
        ids = tok2.encode("hello")
        text = tok2.decode(ids)
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)
        self.tokenizer = None

    # ---------- BaseTokenizer API ----------

    def run(
        self,
        text_file: str,
        vocab_size: int = 32000,
        min_frequency: int = 2,
        special_tokens: Optional[str] = "[PAD],[UNK],[BOS],[EOS]",
        add_bos_eos: bool = True,
    ) -> Dict[str, Any]:
        # Lazy import so package import doesn't hard-require tokenizers
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder

        specials = [s.strip() for s in (special_tokens or "").split(",") if s.strip()]
        if "[UNK]" not in specials:
            specials.append("[UNK]")

        # Initialize tokenizer
        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = ByteLevel()
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=int(min_frequency),
            special_tokens=specials,
            initial_alphabet=ByteLevel.alphabet(),
        )

        # Train from newline-delimited file (one sample per line)
        def iter_lines(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line

        tok.train_from_iterator(iter_lines(text_file), trainer=trainer)

        # Optionally add BOS/EOS post-processor
        if add_bos_eos:
            bos, eos = "[BOS]", "[EOS]"
            bid, eid = tok.token_to_id(bos), tok.token_to_id(eos)
            if bid is not None and eid is not None:
                from tokenizers.processors import TemplateProcessing
                tok.post_processor = TemplateProcessing(
                    single=f"{bos} $0 {eos}",
                    special_tokens=[(bos, bid), (eos, eid)],
                )

        self.tokenizer = tok

        # Return a small training report (you said you don’t want JSON files; this just returns a dict in Python)
        return {
            "text_file": text_file,
            "target_vocab_size": int(vocab_size),
            "actual_vocab_size": int(tok.get_vocab_size()),
            "min_frequency": int(min_frequency),
            "special_tokens": specials,
            "added_bos_eos": bool(add_bos_eos),
        }

    def encode(self, text: str) -> Any:
        self._ensure_loaded()
        return self.tokenizer.encode(text).ids

    def decode(self, ids: Any) -> str:
        self._ensure_loaded()
        # Accept list[int] or tokenizers EncodedInput type
        arr = ids.ids if hasattr(ids, "ids") else ids
        return self.tokenizer.decode(arr)

    def save(self, path: str) -> None:
        self._ensure_loaded()
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tokenizer.save(path)

    def load(self, path: str) -> None:
        from tokenizers import Tokenizer
        self.tokenizer = Tokenizer.from_file(path)

    # ---------- helpers ----------
    def _ensure_loaded(self) -> None:
        if self.tokenizer is None:
            raise RuntimeError(
                f"{self.name}: tokenizer not initialized. "
                "Call .run(...) to train or .load(path) to load a saved tokenizer."
            )
