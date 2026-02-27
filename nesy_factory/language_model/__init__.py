# LanguageModel/__init__.py

# Explicit exports from base & registry (avoid wildcard pollution)
from .base import (
    BaseComponent, BaseExporter, BaseTokenizer, BaseModel,
    BaseModelBuilder, BaseTrainer,
)
from .registry import (
    register, get_component, list_components, create, is_registered,
)

# Import modules with @register side effects so components are actually registered.
# Guard each import so missing optional deps don't break the package import.
try:
    from .text_exporter import TextExporter  # registers "text_exporter"
except Exception:
    TextExporter = None  # optional

try:
    from .tokenizer import ByteLevelBPETokenizer  # registers "bpe_tokenizer"
except Exception:
    ByteLevelBPETokenizer = None

try:
    from .gemma import Gemma3Builder  # registers "gemma3_builder"
except Exception:
    Gemma3Builder = None

try:
    from .trainer import GemmaTrainer  # registers "gemma_trainer"
except Exception:
    GemmaTrainer = None

__all__ = [
    # base
    "BaseComponent","BaseExporter","BaseTokenizer","BaseModel",
    "BaseModelBuilder","BaseTrainer",
    "register","get_component","list_components","create","is_registered",
    "TextExporter","ByteLevelBPETokenizer","Gemma3Builder","GemmaTrainer",
]
