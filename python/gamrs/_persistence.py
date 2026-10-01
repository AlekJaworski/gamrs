"""Disk-level persistence helpers shared by :class:`gamrs.Gam` (and the
:class:`gamrs.GamPredictor` wrapper that routes through it).

File layout written by :func:`save_gam` and consumed by
:func:`load_gam`:

    ┌──────────────┬──────────────────┬─────────────────────────┐
    │ u32 LE       │ JSON meta header │ native PyFittedGam bytes│
    │ meta_len (4B)│ (UTF-8)          │ (length-framed binary)  │
    └──────────────┴──────────────────┴─────────────────────────┘

The native body is whatever the Rust core's
:meth:`gamrs._gamrs_native.FittedGam.serialize` emits — i.e. gamrs's
own ``MAGIC | VERSION | LEN | JSON`` frame. The Python-side header
adds the wrapper-side metadata (family, link, predictors, design,
``_k_used``) so :meth:`Gam.load` can rebuild the wrapper without
the caller re-passing every constructor kwarg.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:  # pragma: no cover — typing only
    from ._fitter import Gam

SCHEMA_TAG = "gamrs.Gam/1"


def _gather_meta(gam: "Gam") -> dict[str, Any]:
    """Snapshot the wrapper-side fields that survive a save/load."""
    return {
        "schema": SCHEMA_TAG,
        "family": gam.family,
        "link": gam.link,
        "method": gam.method,
        "design": gam.design,
        "target": gam.target,
        "predictors": list(gam.predictors) if gam.predictors is not None else None,
        "_effective_predictors": list(gam._effective_predictors or []),
        "_original_predictors": list(gam._original_predictors or []),
        "_k_used": gam._k_used,
        "df": gam.df,
        "tweedie_p": gam.tweedie_p,
        "negbin_theta": gam.negbin_theta,
        "r": gam.r,
        "aliased_": list(gam.aliased_),
    }


def save_gam(gam: "Gam", path: Union[str, "os.PathLike[str]"]) -> None:
    """Write a fitted :class:`Gam` to disk."""
    if gam._fitted is None:
        raise RuntimeError(
            "Gam must be fitted before saving — call .fit() first."
        )
    meta_bytes = json.dumps(_gather_meta(gam)).encode("utf-8")
    native_bytes = bytes(gam._fitted.serialize())
    with open(path, "wb") as fh:
        fh.write(len(meta_bytes).to_bytes(4, "little"))
        fh.write(meta_bytes)
        fh.write(native_bytes)


def load_gam(cls: type, path: Union[str, "os.PathLike[str]"]) -> "Gam":
    """Read a Gam back. ``cls`` is the :class:`Gam` class (passed in to
    avoid a circular import)."""
    with open(path, "rb") as fh:
        head = fh.read(4)
        if len(head) != 4:
            raise ValueError(f"gamrs.Gam.load: file {path!r} is too short")
        meta_len = int.from_bytes(head, "little")
        meta_bytes = fh.read(meta_len)
        if len(meta_bytes) != meta_len:
            raise ValueError(f"gamrs.Gam.load: truncated metadata in {path!r}")
        meta = json.loads(meta_bytes.decode("utf-8"))
        native_bytes = fh.read()
    if not isinstance(meta, dict) or meta.get("schema") != SCHEMA_TAG:
        got = meta.get("schema") if isinstance(meta, dict) else type(meta).__name__
        raise ValueError(
            f"gamrs.Gam.load: file {path!r} does not carry a {SCHEMA_TAG} "
            f"schema header (got {got!r})"
        )
    gam = cls.deserialize(native_bytes)
    # Restore wrapper-side metadata from the JSON header.
    gam.family = meta.get("family", "gaussian")
    gam.link = meta.get("link", "identity")
    gam.method = meta.get("method", "REML")
    gam.design = meta.get("design", "cr")
    gam.target = meta.get("target", "y")
    gam.predictors = meta.get("predictors")
    gam._effective_predictors = meta.get("_effective_predictors") or None
    gam._original_predictors = meta.get("_original_predictors") or None
    gam._k_used = meta.get("_k_used")
    gam.df = meta.get("df")
    gam.tweedie_p = meta.get("tweedie_p")
    gam.negbin_theta = meta.get("negbin_theta")
    gam.r = meta.get("r")
    gam.aliased_ = meta.get("aliased_", gam.aliased_)
    # Lazy import to avoid a circular load.
    from ._coerce import FAMILY_TO_GAMRS
    gam._gamrs_family = FAMILY_TO_GAMRS.get(gam.family, (gam.family, gam.link))[0]
    return gam
