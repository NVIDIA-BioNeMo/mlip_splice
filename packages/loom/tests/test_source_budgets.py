# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from inspect import Parameter, signature

import pytest
from loom import Compile, _native
from loom._src.runtime import recipes


@pytest.fixture(autouse=True)
def source_only_device(monkeypatch):
    monkeypatch.setattr(recipes, "_device_fingerprint", lambda: ("source-test", 100))
    token = recipes._aot.set(None)
    try:
        yield
    finally:
        recipes._aot.reset(token)


def test_both_budgets_reach_source_compilation_without_aot():
    seen = []

    def compile_sources(*args, **kwargs):
        seen.append({name: kwargs[name] for name in ("search_budget_ms", "shortlist_budget_ms")})
        return b"shortlist"

    result = recipes.compile_recipe(
        compile_sources,
        "program",
        compile_options=Compile(search_budget_ms=20.0, shortlist_budget_ms=70.0),
    )
    assert seen == [{"search_budget_ms": 20.0, "shortlist_budget_ms": 70.0}]
    assert result == (b"shortlist", None)


def test_ambient_budgets_resolve_before_source_compilation():
    seen = []

    def compile_sources(*args, **kwargs):
        seen.append({name: kwargs[name] for name in ("search_budget_ms", "shortlist_budget_ms")})
        return b"shortlist"

    with recipes.aot(
        Compile(search_budget_ms={0: 20.0, 2: 30.0}, shortlist_budget_ms={0: 70.0, 1: 90.0})
    ):
        for depth in range(4):
            recipes.compile_recipe(
                compile_sources, "program", compile_options=Compile(transpose_depth=depth)
            )
        recipes.compile_recipe(
            compile_sources,
            "program",
            compile_options=Compile(
                search_budget_ms=5.0, shortlist_budget_ms=6.0, transpose_depth=2
            ),
        )
    assert seen == [
        {"search_budget_ms": 20.0, "shortlist_budget_ms": 70.0},
        {"search_budget_ms": 20.0, "shortlist_budget_ms": 90.0},
        {"search_budget_ms": 30.0, "shortlist_budget_ms": 90.0},
        {"search_budget_ms": 30.0, "shortlist_budget_ms": 90.0},
        {"search_budget_ms": 5.0, "shortlist_budget_ms": 6.0},
    ]


def test_shortlist_budget_is_part_of_the_recipe_cache_key():
    seen = []

    def compile_sources(*args, **kwargs):
        seen.append({name: kwargs[name] for name in ("search_budget_ms", "shortlist_budget_ms")})
        return b"shortlist"

    cache = {}
    with recipes.aot(Compile(search_budget_ms=20.0), cache=cache, workload_identity="source-test"):
        _, capture = recipes.compile_recipe(
            compile_sources, "program", compile_options=Compile(shortlist_budget_ms=70.0)
        )
        cache[capture.key] = (b"cached-shortlist", ())
        assert recipes.compile_recipe(
            compile_sources, "program", compile_options=Compile(shortlist_budget_ms=70.0)
        ) == (
            b"cached-shortlist",
            None,
        )
        _, other = recipes.compile_recipe(
            compile_sources, "program", compile_options=Compile(shortlist_budget_ms=90.0)
        )
        assert other.key != capture.key
        _, unbounded = recipes.compile_recipe(compile_sources, "program")
        assert unbounded.key not in {capture.key, other.key}
    assert len(seen) == 3


def test_kiln_receives_the_frozen_recipe_without_a_second_budget(monkeypatch):
    seen = []

    class Handoff(Exception):
        pass

    def load_recipe(recipe, **kwargs):
        seen.append((recipe, kwargs))
        raise Handoff

    monkeypatch.setattr(recipes._engine, "load_recipe", load_recipe)
    with recipes.aot(Compile(shortlist_budget_ms=90.0)), pytest.raises(Handoff):
        recipes.run_recipe(b"frozen-shortlist", [], [])
    assert seen == [
        (b"frozen-shortlist", {"tune_report_to": None, "scheduling_time_limit_ms": None})
    ]


def test_source_compilation_uses_native_defaults():
    def compile_sources(program, **kwargs):
        assert program == "program"
        assert kwargs == _native.compile_defaults()
        return b"all-sources"

    assert recipes.compile_recipe(compile_sources, "program") == (b"all-sources", None)


def test_inspectable_recipe_compilation_resolves_ambient_budgets(monkeypatch):
    from types import SimpleNamespace

    from loom._src import pipeline
    from loom._src.primitive.options import Compile

    seen = []

    class SourceBoundary(Exception):
        pass

    def compile_sources(*args, **kwargs):
        seen.append({name: kwargs[name] for name in ("search_budget_ms", "shortlist_budget_ms")})
        raise SourceBoundary

    monkeypatch.setattr(pipeline, "unmarshal_domains", lambda wire: ({}, {}))
    monkeypatch.setattr(pipeline._engine, "compile_loom_to_bytes", compile_sources)
    lowered = SimpleNamespace(
        _wire=SimpleNamespace(layout=SimpleNamespace(domain_sizes={}, operand_domains=[])),
        xir=SimpleNamespace(text=lambda: "program"),
        hidden_width=32,
        inputs=[],
        outputs=[],
    )
    with recipes.aot(Compile(search_budget_ms=20.0, shortlist_budget_ms=70.0)):
        with pytest.raises(SourceBoundary):
            pipeline.compile(lowered)
        with pytest.raises(SourceBoundary):
            pipeline.compile(lowered, compile_options=Compile(shortlist_budget_ms=90.0))
    assert [call["search_budget_ms"] for call in seen] == [20.0, 20.0]
    assert [call["shortlist_budget_ms"] for call in seen] == [70.0, 90.0]


def test_native_search_budget_keyword_matches_the_public_name():
    budget = signature(_native.compile_loom_to_bytes).parameters["search_budget_ms"]
    assert budget.kind is Parameter.POSITIONAL_OR_KEYWORD
    assert budget.default is None
