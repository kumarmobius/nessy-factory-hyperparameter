# lm_factory/registry.py
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple, Type, Callable
import inspect

# Global catalog: name -> class (not instance)
FACTORY: Dict[str, Type[Any]] = {}


class RegistryError(Exception):
    """Raised for invalid registrations or lookups."""


def _validate_class(obj: Any) -> None:
    """Ensure obj is a class and looks like a component (has run())."""
    if not inspect.isclass(obj):
        raise RegistryError("Only classes can be registered.")
    if not hasattr(obj, "run") or not callable(getattr(obj, "run")):
        raise RegistryError(f"{obj.__name__} must define a run() method.")


def register(name: Optional[str] = None) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to register a component class under a string key.

    Usage:
        @register("text_exporter")
        class TextExporter(BaseExporter):
            ...
    """
    def _decorator(cls: Type[Any]) -> Type[Any]:
        _validate_class(cls)
        key = (name or cls.__name__).strip()
        if not key:
            raise RegistryError("Registration name must be a non-empty string.")
        if key in FACTORY:
            raise RegistryError(f"Component '{key}' is already registered.")
        FACTORY[key] = cls
        return cls
    return _decorator


def is_registered(name: str) -> bool:
    return name in FACTORY


def unregister(name: str) -> None:
    """Remove a component by name (mostly for tests/dev)."""
    FACTORY.pop(name, None)


def get_component(name: str, **kwargs) -> Any:
    """
    Instantiate a registered component by name with kwargs.

    Example:
        tok = get_component("bpe_tokenizer", vocab_size=32000)
    """
    key = name.strip()
    if key not in FACTORY:
        hint = _nearest(key, FACTORY.keys())
        raise RegistryError(
            f"Component '{key}' is not registered."
            + (f" Did you mean: {hint}?" if hint else "")
        )
    cls = FACTORY[key]
    try:
        return cls(**kwargs)
    except TypeError as e:
        # Provide a helpful constructor signature in the error
        sig = inspect.signature(cls)
        raise RegistryError(
            f"Could not construct '{key}' with arguments {kwargs}.\n"
            f"Constructor signature: {cls.__name__}{sig}\n"
            f"Original error: {e}"
        ) from e


def list_components(prefix: Optional[str] = None) -> Iterable[str]:
    """List all registered names, optionally filtered by prefix."""
    names = sorted(FACTORY.keys())
    if prefix:
        p = prefix.lower()
        names = [n for n in names if n.lower().startswith(p)]
    return names


def create(name: str, **kwargs) -> Any:
    """Alias for get_component for users who prefer 'create' wording."""
    return get_component(name, **kwargs)


# -------- small helpers --------

def _nearest(target: str, candidates: Iterable[str]) -> Optional[str]:
    """Return a simple nearest key suggestion (fast, dependency-free)."""
    target_l = target.lower()
    best: Tuple[int, Optional[str]] = (10**9, None)
    for cand in candidates:
        d = _cheap_distance(target_l, cand.lower())
        if d < best[0]:
            best = (d, cand)
    return best[1]


def _cheap_distance(a: str, b: str) -> int:
    """Tiny edit-distance-ish heuristic (no external deps)."""
    L = max(len(a), len(b))
    pad_a, pad_b = a.ljust(L), b.ljust(L)
    mis = sum(1 for x, y in zip(pad_a, pad_b) if x != y)
    return abs(len(a) - len(b)) * 2 + mis
