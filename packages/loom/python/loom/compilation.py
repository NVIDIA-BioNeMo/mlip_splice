# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable source records for the same compiler build and GPU target.

Records use checksummed JSON, not pickle. Checksums detect corruption, not
malicious changes. Only replay records from a trusted source: recipes contain
GPU programs. No compatibility across compiler builds or GPU targets is promised.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class CompilerIdentity:
    """Content identity of native/Python compiler code, JAX, and GPU target."""

    build_sha256: str
    native_sha256: str
    jax_version: str
    jaxlib_version: str
    device_kind: str
    compute_capability: str


def compiler_identity() -> CompilerIdentity:
    """Identify the installed compiler and its default CUDA target by content."""
    import algebraic_real
    import algebraic_real._native
    import jax

    import loom._native

    files = {}
    for name, root in (
        ("loom", Path(__file__).parent),
        ("algebraic_real", Path(algebraic_real.__file__).parent),
    ):
        for source in sorted(root.rglob("*.py")):
            files[f"{name}/{source.relative_to(root)}"] = _sha256(source.read_bytes())
    native = _sha256(Path(loom._native.__file__).read_bytes())
    files["loom/_native"] = native
    files["algebraic_real/_native"] = _sha256(Path(algebraic_real._native.__file__).read_bytes())
    device = jax.devices("gpu")[0]
    return CompilerIdentity(
        _sha256(_json(files).encode()),
        native,
        jax.__version__,
        importlib.metadata.version("jaxlib"),
        device.device_kind,
        str(device.compute_capability),
    )


@dataclass(frozen=True)
class SourceCompilation:
    """One source compilation, before assembly, with detached diagnostic views.

    ``slots``, ``profile``, and ``diagnostics`` return fresh JSON values. Editing
    these views cannot change the frozen record. ``recipe`` is the executable
    byte payload; input and output ports are rebuilt from the matched call.
    """

    identity: CompilerIdentity
    signature: str
    xir: str
    recipe: bytes
    _details_json: str

    @classmethod
    def _from_result(cls, identity, signature, xir, result, diagnostics):
        recipe, slots, profile = result
        return cls(
            identity,
            signature,
            xir,
            bytes(recipe),
            _json(dict(slots=slots, profile=profile, diagnostics=diagnostics)),
        )

    @property
    def slots(self):
        return json.loads(self._details_json)["slots"]

    @property
    def profile(self):
        return json.loads(self._details_json)["profile"]

    @property
    def diagnostics(self):
        return json.loads(self._details_json)["diagnostics"]

    def to_bytes(self) -> bytes:
        """Serialize all fields with a schema version and content checksum."""
        payload = dict(
            version=1,
            identity=asdict(self.identity),
            signature=self.signature,
            xir=self.xir,
            recipe=base64.b64encode(self.recipe).decode("ascii"),
            details=json.loads(self._details_json),
        )
        return _json(dict(payload=payload, sha256=_sha256(_json(payload).encode()))).encode()

    @classmethod
    def from_bytes(cls, raw: bytes) -> SourceCompilation:
        """Read a record without importing or executing serialized Python code."""
        try:
            envelope = json.loads(raw)
            payload = envelope["payload"]
            if payload["version"] != 1:
                raise ValueError("unsupported source compilation version")
            if envelope["sha256"] != _sha256(_json(payload).encode()):
                raise ValueError("source compilation checksum mismatch")
            identity = CompilerIdentity(**payload["identity"])
            if not all(isinstance(value, str) for value in asdict(identity).values()):
                raise ValueError("invalid compiler identity")
            if not isinstance(payload["xir"], str) or not isinstance(payload["signature"], str):
                raise ValueError("invalid source signature or XIR")
            details = payload["details"]
            if not (
                isinstance(details["slots"], list)
                and isinstance(details["profile"], dict)
                and isinstance(details["diagnostics"], dict)
            ):
                raise ValueError("invalid source details")
            return cls(
                identity,
                payload["signature"],
                payload["xir"],
                base64.b64decode(payload["recipe"], validate=True),
                _json(details),
            )
        except (KeyError, TypeError, UnicodeError) as error:
            raise ValueError("invalid source compilation record") from error

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    @classmethod
    def load(cls, path: str | Path) -> SourceCompilation:
        return cls.from_bytes(Path(path).read_bytes())


@dataclass(frozen=True)
class RecipeLoadStarted:
    """Emitted immediately before native recipe registration."""

    recipe_sha256: str
    report_path: str | None


@dataclass(frozen=True)
class RecipeLoadFinished:
    """Emitted after native recipe registration, before the JAX custom call.

    Assembly and tuning are lazy: they run when the custom call needs the
    recipe. ``report_path`` names their eventual report, not a file that must
    already exist. ``elapsed_s`` measures registration, not lazy realization."""

    recipe_sha256: str
    report_path: str | None
    elapsed_s: float
    started_unix_ns: int
    workspace_bytes: int
    runtime_identity: bytes
