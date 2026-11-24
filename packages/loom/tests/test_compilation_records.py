# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture persistence, strict matching, and context isolation without a GPU."""

import contextvars
import dataclasses
import json

import loom
import pytest
from loom._src.runtime import instrumentation as events
from loom.compilation import (
    CompilerIdentity,
    RecipeLoadFinished,
    RecipeLoadStarted,
    SourceCompilation,
)


class Module:
    def text(self):
        return "xir program"


@pytest.fixture
def identity(monkeypatch):
    value = CompilerIdentity("build", "native", "jax", "jaxlib", "gpu", "12.0")
    monkeypatch.setattr(events, "compiler_identity", lambda: value)
    return value


def compile_source(*args, **kwargs):
    kwargs["diagnostics"]["arms"] = ["candidate"]
    return b"recipe", [("index", "source", "target", "")], {"source_ms": 1.0}


def run(compiler=compile_source, *, width=32, **kwargs):
    return events.compile_sources(compiler, "loom", Module(), "xir program", width, **kwargs)


def capture():
    recorded = []
    with loom.observe(recorded.append):
        run(search_budget_ms=20)
    return recorded[0]


def test_record_roundtrip_and_detached_views(identity, tmp_path):
    artifact = capture()
    artifact.save(tmp_path / "record.json")
    assert SourceCompilation.load(tmp_path / "record.json") == artifact
    artifact.profile["source_ms"] = 999
    artifact.slots.clear()
    artifact.diagnostics["arms"].clear()
    assert artifact.profile == {"source_ms": 1.0}
    assert artifact.slots
    assert artifact.diagnostics == {"arms": ["candidate"]}
    with pytest.raises(dataclasses.FrozenInstanceError):
        artifact.recipe = b"changed"


def test_corruption_and_version_rejected(identity):
    record = json.loads(capture().to_bytes())
    record["payload"]["recipe"] = "changed"
    with pytest.raises(ValueError, match="checksum"):
        SourceCompilation.from_bytes(json.dumps(record).encode())
    record["payload"]["version"] = 2
    with pytest.raises(ValueError, match="version"):
        SourceCompilation.from_bytes(json.dumps(record).encode())
    with pytest.raises(ValueError):
        SourceCompilation.from_bytes(b"not JSON")


def test_replay_never_calls_compiler_and_restores_diagnostics(identity):
    artifact = capture()
    observed = []
    diagnostics = {}

    def forbidden(*args, **kwargs):
        pytest.fail("source compiler ran during replay")

    with loom.replay([artifact]), loom.observe(observed.append):
        result = run(forbidden, diagnostics=diagnostics, search_budget_ms=20, dump_ptx=True)
    assert result == (artifact.recipe, artifact.slots, artifact.profile)
    assert observed == [artifact]
    assert diagnostics == artifact.diagnostics


@pytest.mark.parametrize("change", [dict(width=64, search_budget_ms=20), dict(search_budget_ms=21)])
def test_semantic_mismatch_rejected(identity, change):
    with pytest.raises(loom.ReplayError, match="no capture"):
        with loom.replay([capture()]):
            run(**change)
    assert not events.active()


def test_incompatible_build_and_target_rejected(identity):
    artifact = capture()
    for field in (
        "build_sha256",
        "native_sha256",
        "device_kind",
        "compute_capability",
        "jax_version",
        "jaxlib_version",
    ):
        incompatible = dataclasses.replace(
            artifact, identity=dataclasses.replace(identity, **{field: "other"})
        )
        with pytest.raises(loom.ReplayError, match="same compiler build and GPU target"):
            with loom.replay([incompatible]):
                pytest.fail("entered incompatible replay")


def test_missing_unused_and_repeated_records(identity):
    artifact = capture()
    with pytest.raises(loom.ReplayError, match="unused"):
        with loom.replay([artifact]):
            pass
    with pytest.raises(loom.ReplayError, match="no capture"):
        with loom.replay([]):
            run()
    with loom.replay([artifact, artifact]):
        run(search_budget_ms=20)
        run(search_budget_ms=20)
    with pytest.raises(loom.ReplayError, match="no capture"):
        with loom.replay([artifact]):
            run(search_budget_ms=20)
            run(search_budget_ms=20)


def test_abort_nested_observers_and_context_isolation(identity):
    class Stop(Exception):
        pass

    def abort(event):
        raise Stop

    outer, inner = [], []
    with loom.observe(outer.append):
        with loom.observe(inner.append):
            run()
        assert not contextvars.Context().run(events.active)
        with pytest.raises(Stop), loom.observe(abort):
            run()
        run()
    assert len(outer) == 3
    assert len(inner) == 1
    assert not events.active()


def test_observation_bypasses_aot_cache(identity, monkeypatch):
    from loom._src.runtime import recipes

    monkeypatch.setattr(recipes, "_device_fingerprint", lambda: pytest.fail("AOT cache used"))
    recorded = []
    with loom.aot(cache={}, workload_identity="test"), loom.observe(recorded.append):
        for _ in range(2):
            result = recipes.compile_recipe(compile_source, "loom", Module(), "xir program", 32)
            assert result[-1] is None
    assert len(recorded) == 2


def test_recipe_registration_events():
    recorded = []

    def loader(recipe, **kwargs):
        assert recipe == b"recipe"
        assert kwargs["tune_report_to"] == "report.json"
        return b"identity", 64

    with loom.observe(recorded.append):
        assert events.load_recipe(loader, b"recipe", tune_report_to="report.json") == (
            b"identity",
            64,
        )
    start, finish = recorded
    assert isinstance(start, RecipeLoadStarted)
    assert isinstance(finish, RecipeLoadFinished)
    assert start.recipe_sha256 == finish.recipe_sha256
    assert finish.report_path == "report.json"
    assert finish.elapsed_s >= 0
    assert finish.workspace_bytes == 64
    assert finish.runtime_identity == b"identity"
