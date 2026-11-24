# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Field-wise configuration, source boundaries, and runtime policy without a GPU."""

import copy
import pickle
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from loom import Compile, _native
from loom._src import pipeline
from loom._src.runtime import recipes

FIELDS = {
    "search_budget_ms": 11.0,
    "shortlist_budget_ms": 21.0,
    "search_memory_budget_bytes": 100,
    "max_workspace_bytes": 200,
    "max_kernel_statements": 300,
    "max_algebraic_work": 400,
    "max_algebraic_expression_work": 500,
    "scheduling_time_limit_ms": 31.0,
}
NULLABLE = set(FIELDS) - {"max_algebraic_work", "max_algebraic_expression_work"}


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch):
    monkeypatch.setattr(recipes, "_device_fingerprint", lambda: ("source-test", 100))
    token = recipes._aot.set(None)
    try:
        yield
    finally:
        recipes._aot.reset(token)


@pytest.mark.parametrize("name,value", FIELDS.items())
def test_field_priority_and_same_valued_explicit_default(name, value):
    defaults = Compile().resolve()
    with recipes.aot(Compile(**{name: value})):
        assert getattr(recipes.resolve_compile_options(), name) == value
        explicit = Compile(**{name: getattr(defaults, name)})
        assert getattr(recipes.resolve_compile_options(explicit), name) == getattr(defaults, name)
        with recipes.aot(Compile(**{name: value + 1})):
            assert getattr(recipes.resolve_compile_options(), name) == value + 1
            assert getattr(recipes.resolve_compile_options(explicit), name) == getattr(
                defaults, name
            )
        assert getattr(recipes.resolve_compile_options(), name) == value
    assert recipes.resolve_compile_options() == defaults


@pytest.mark.parametrize("name", sorted(NULLABLE))
def test_explicit_none_blocks_inheritance(name):
    with recipes.aot(Compile(**{name: FIELDS[name]})):
        assert getattr(recipes.resolve_compile_options(Compile(**{name: None})), name) is None
        with recipes.aot(Compile(**{name: None})):
            assert getattr(recipes.resolve_compile_options(), name) is None


def test_nested_contexts_merge_fields_and_restore_after_error():
    outer = Compile(search_memory_budget_bytes=64, max_algebraic_work=16)
    inner = Compile(max_workspace_bytes=0, scheduling_time_limit_ms=0)
    with recipes.aot(outer):
        with pytest.raises(RuntimeError), recipes.aot(inner):
            resolved = recipes.resolve_compile_options(Compile(max_algebraic_expression_work=8))
            assert resolved.search_memory_budget_bytes == 64
            assert resolved.max_algebraic_work == 16
            assert resolved.max_workspace_bytes == 0
            assert resolved.max_algebraic_expression_work == 8
            assert resolved.scheduling_time_limit_ms == 0
            raise RuntimeError
        assert recipes.resolve_compile_options() == outer.resolve()
    assert recipes.resolve_compile_options() == Compile().resolve()


@pytest.mark.parametrize("name", ["search_budget_ms", "shortlist_budget_ms"])
def test_schedule_layers_use_the_program_depth_and_explicit_none(name):
    outer = Compile(**{name: {0: 100, 2: 200}}, transpose_depth=99)
    inner = Compile(**{name: {1: None, 3: 300}}, transpose_depth=50)
    call = Compile(**{name: {2: 20, 4: None}})
    with recipes.aot(outer), recipes.aot(inner):
        for depth, expected in enumerate([100, None, 20, 20, None, None]):
            resolved = recipes.resolve_compile_options(replace(call, transpose_depth=depth))
            assert getattr(resolved, name) == expected
        for mapping in ({}, {9: 7}):
            assert getattr(recipes.resolve_compile_options(Compile(**{name: mapping})), name) == 100
    assert getattr(Compile(**{name: {1: None}}).resolve(), name) == getattr(
        Compile().resolve(), name
    )


@pytest.mark.parametrize(
    "config", [Compile(), Compile(**FIELDS), Compile(shortlist_budget_ms={2: None, 0: 100})]
)
def test_config_is_hashable_pickleable_and_preserved_by_transpose(config):
    for restored in (copy.deepcopy(config), pickle.loads(pickle.dumps(config))):
        assert restored == config
        assert hash(restored) == hash(config)
        assert restored.resolve() == config.resolve()
    advanced = config.advance_transpose()
    assert advanced.transpose_depth == config.transpose_depth + 1
    assert replace(advanced, transpose_depth=config.transpose_depth) == config
    assert config.transpose_depth == 0


@pytest.mark.parametrize("name", list(FIELDS)[2:7])
@pytest.mark.parametrize("value", [True, 1.5, -1, 1 << 64])
def test_integer_limits_reject_invalid_values(name, value):
    with pytest.raises((TypeError, ValueError), match=name):
        Compile(**{name: value})


@pytest.mark.parametrize("name", ["max_algebraic_work", "max_algebraic_expression_work"])
@pytest.mark.parametrize("value", [None, 0])
def test_algebraic_limits_are_positive_and_not_nullable(name, value):
    with pytest.raises((TypeError, ValueError), match=name):
        Compile(**{name: value})


@pytest.mark.parametrize(
    "value",
    [
        True,
        -1,
        -float.fromhex("0x0.0000000000001p-1022"),
        float("inf"),
        float("nan"),
        float(1 << 64) * 1000,
    ],
)
def test_runtime_limit_rejects_invalid_durations(value):
    with pytest.raises((TypeError, ValueError), match="scheduling_time_limit_ms"):
        Compile(scheduling_time_limit_ms=value)


@pytest.mark.parametrize("value", [None, 0, 0.5, 1000])
def test_runtime_limit_accepts_unlimited_zero_and_positive(value):
    assert Compile(scheduling_time_limit_ms=value).resolve().scheduling_time_limit_ms == value


def test_aot_requires_config_and_keyword_cache():
    with pytest.raises(TypeError, match="Compile"), recipes.aot({}):
        pass
    with pytest.raises(TypeError):
        with recipes.aot(shortlist_budget_ms=100):
            pass
    with pytest.raises(ValueError, match="workload_identity"), recipes.aot(Compile(), cache={}):
        pass


@pytest.mark.parametrize("name,value", FIELDS.items())
def test_each_setting_distinguishes_cached_realizations(name, value):
    calls = []

    def compile_sources(*args, **kwargs):
        calls.append(kwargs)
        return b"source"

    cache = {}
    with recipes.aot(cache=cache, workload_identity="config-test"):
        _, first = recipes.compile_recipe(compile_sources, "program")
        cache[first.key] = (b"cached", ())
        config = Compile(**{name: value})
        _, second = recipes.compile_recipe(compile_sources, "program", compile_options=config)
        assert second.key != first.key
        cache[second.key] = (b"configured", ())
        assert recipes.compile_recipe(compile_sources, "program", compile_options=config) == (
            b"configured",
            None,
        )
        assert recipes.compile_recipe(
            compile_sources, "program", compile_options=Compile().resolve()
        ) == (b"cached", None)
    assert len(calls) == 2
    assert "scheduling_time_limit_ms" not in calls[1]
    if name == "scheduling_time_limit_ms":
        assert calls[0] == calls[1]
    else:
        assert calls[1][name] == value


def test_explicit_unlimited_shortlist_differs_from_default_cache_entry():
    def compile_sources(*args, **kwargs):
        return b"source"

    with recipes.aot():
        _, default = recipes.compile_recipe(compile_sources, "program")
        _, unlimited = recipes.compile_recipe(
            compile_sources, "program", compile_options=Compile(shortlist_budget_ms=None)
        )
        assert default.key != unlimited.key


class Boundary(Exception):
    pass


@pytest.mark.parametrize(
    "explicit,expected",
    [
        (Compile(), 31),
        (Compile(scheduling_time_limit_ms=None), None),
        (Compile(scheduling_time_limit_ms=0), 0),
    ],
)
def test_runtime_policy_reaches_kiln_without_source_limits(monkeypatch, explicit, expected):
    seen = []

    def load_recipe(recipe, **kwargs):
        seen.append((recipe, kwargs))
        raise Boundary

    monkeypatch.setattr(recipes._engine, "load_recipe", load_recipe)
    with recipes.aot(Compile(**FIELDS)), pytest.raises(Boundary):
        recipes.run_recipe(b"shortlist", [], [], compile_options=explicit)
    assert seen == [(b"shortlist", {"tune_report_to": None, "scheduling_time_limit_ms": expected})]


def test_pipeline_source_receives_all_limits_but_not_runtime_policy(monkeypatch):
    seen = []

    def compile_sources(*args, **kwargs):
        seen.append(kwargs)
        raise Boundary

    monkeypatch.setattr(pipeline, "unmarshal_domains", lambda wire: ({}, {}))
    monkeypatch.setattr(pipeline._engine, "compile_loom_to_bytes", compile_sources)
    lowered = SimpleNamespace(
        _wire=SimpleNamespace(layout=SimpleNamespace(domain_sizes={}, operand_domains=[])),
        xir=SimpleNamespace(text=lambda: "program"),
        hidden_width=32,
        inputs=[],
        outputs=[],
    )
    with recipes.aot(Compile(**FIELDS)), pytest.raises(Boundary):
        pipeline.compile(lowered, compile_options=Compile(max_workspace_bytes=None))
    expected = replace(Compile(**FIELDS), max_workspace_bytes=None).resolve().source_kwargs()
    assert {name: seen[0][name] for name in _native.compile_defaults()} == expected
    assert "scheduling_time_limit_ms" not in seen[0]


def test_pipeline_execute_uses_runtime_config(monkeypatch):
    monkeypatch.setattr(pipeline, "_check_inputs", lambda lowered, inputs: [])
    monkeypatch.setattr(pipeline, "unmarshal_domains", lambda wire: ({}, {}))
    monkeypatch.setattr(pipeline, "resolve_index_bindings", lambda *args: [])
    seen = []

    def load_recipe(recipe, **kwargs):
        seen.append(kwargs)
        raise Boundary

    monkeypatch.setattr(recipes._engine, "load_recipe", load_recipe)
    lowered = SimpleNamespace(_wire=None, outputs=[])
    recipe = SimpleNamespace(inputs=[])
    with recipes.aot(Compile(scheduling_time_limit_ms=31)):
        with pytest.raises(Boundary):
            pipeline.execute(lowered, recipe, [])
        with pytest.raises(Boundary):
            pipeline.execute(
                lowered, recipe, [], compile_options=Compile(scheduling_time_limit_ms=0)
            )
    assert [call["scheduling_time_limit_ms"] for call in seen] == [31, 0]


def test_resolved_defaults_have_no_inherit_markers():
    values = asdict(Compile().resolve())
    assert all(value is not ... for value in values.values())
    assert Compile().resolve().source_kwargs() == _native.compile_defaults()
    assert values["shortlist_budget_ms"] == 600_000
    with pytest.raises(ValueError, match="resolve"):
        Compile().source_kwargs()


def test_primitive_lowering_forwards_one_config_to_both_stages(monkeypatch):
    from loom._src.primitive import cuda

    config = Compile(**FIELDS)
    source_calls = []
    runtime_calls = []

    def run_recipe(recipe, *args, **kwargs):
        runtime_calls.append(kwargs)
        return []

    monkeypatch.setattr(cuda, "_to_xir", lambda _: SimpleNamespace(text=lambda: "program"))
    monkeypatch.setattr(cuda, "Recipe", SimpleNamespace)
    monkeypatch.setattr(cuda, "marshal_index_arrays", lambda *args: [])
    monkeypatch.setattr(cuda, "record", lambda _: None)
    monkeypatch.setattr(cuda, "run_recipe", run_recipe)
    # A held recipe avoids constructing a native executable from the stub payload.
    held = SimpleNamespace()

    def compile_held(*args, **kwargs):
        source_calls.append(kwargs)
        return held, [], {}

    monkeypatch.setattr(cuda._engine, "compile_loom_to_bytes", compile_held)
    with recipes.aot(Compile(max_workspace_bytes=999)):
        assert (
            cuda.loom_engine_call(
                [],
                [],
                jaxpr=None,
                u=1,
                num_axes=0,
                domain_sizes={},
                operand_domains=[],
                index_maps={},
                preimages={},
                options=config,
                input_names=[],
                output_names=[],
            )
            == []
        )
    assert {
        name: source_calls[0][name] for name in _native.compile_defaults()
    } == config.resolve().source_kwargs()
    assert "scheduling_time_limit_ms" not in source_calls[0]
    assert runtime_calls[0]["compile_options"] == config
