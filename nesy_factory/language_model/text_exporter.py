
from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
from datetime import datetime

from .base import BaseExporter
from .registry import register

def _lazy_imports():
    # Lazy so importing the package doesn’t hard-require these deps
    from datasets import load_dataset, Value  # type: ignore
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(x, **kwargs):  # type: ignore
            return x
    return load_dataset, Value, tqdm


@register("text_exporter")
class TextExporter(BaseExporter):
    """
    Exports a Hugging Face dataset split to a newline-delimited text file.
    Mirrors your YAML tool:
      - auto text-field detection
      - supports streaming
      - supports max_rows (0/None = unlimited)
      - returns a metadata dict similar to metadata_json
    """

    # ---------- helpers ----------
    @staticmethod
    def _detect_text_fields(dataset, preferred_fields: Optional[List[str]] = None) -> List[str]:
        if preferred_fields is None:
            preferred_fields = ["text","content","sentence","document","article","body","message"]

        features = getattr(dataset, "features", None)
        if features is None:
            # streaming datasets often don’t expose features
            return preferred_fields

        from datasets import Value  # local import ok
        fields: List[str] = [n for n, feat in features.items()
                             if isinstance(feat, Value) and feat.dtype in ("string", "large_string")]

        if not fields:
            for fn in preferred_fields:
                if fn in features:
                    fields.append(fn)
        if not fields and features:
            # fallback to first feature
            fields = [next(iter(features.keys()))]
        return fields

    @staticmethod
    def _process_example(ex: dict, fields: List[str], sep: str = " ") -> Optional[str]:
        parts: List[str] = []
        for f in fields:
            v = ex.get(f)
            if v is None:
                continue
            parts.append(" ".join(str(i) for i in v if i is not None) if isinstance(v, list) else str(v))
        if not parts:
            return None
        out = sep.join(s.strip() for s in parts if s and s.strip())
        out = out.replace("\r\n", "\n").replace("\r", "\n")
        return out.strip() or None





    # ---------- main API ----------
    def run(
        self,
        dataset: str,
        split: str = "train",
        text_fields: Optional[str] = None,   # comma-separated or None for auto
        max_rows: Optional[int] = None,      # None/0 => unlimited
        streaming: bool = False,
        output_file: str = "exported.txt",
    ) -> Dict[str, Any]:
        load_dataset, Value, tqdm = _lazy_imports()

        start_time = datetime.now()

        # parse text_fields param
        fields_list: Optional[List[str]] = None
        if text_fields:
            fields_list = [s.strip() for s in text_fields.split(",") if s.strip()]

        # load dataset
        ds = load_dataset(dataset, split=split, streaming=streaming)

        # auto-detect text fields
        if not fields_list:
            fields_list = self._detect_text_fields(ds)

        # write out
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        exported = 0
        iterator = ds if streaming else tqdm(ds, desc="Exporting", unit="rows")

        with open(output_file, "w", encoding="utf-8") as f:
            for ex in iterator:
                line = self._process_example(ex, fields_list)
                if line:
                    f.write(line + "\n")
                    exported += 1
                    if max_rows and max_rows > 0 and exported >= max_rows:
                        break

        end_time = datetime.now()

        # metadata (matches your YAML’s metadata_json content/intent)
        meta: Dict[str, Any] = {
            "dataset_name": dataset,
            "config_name": None,
            "split_name": split,
            "text_fields": fields_list,
            "separator": " ",
            "streaming_mode": bool(streaming),
            "max_rows": int(max_rows) if max_rows else None,
            "exported_rows": int(exported),
            "output_file": output_file,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "status": "success",
        }
        return meta
