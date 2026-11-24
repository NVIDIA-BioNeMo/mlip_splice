# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import loom._native as native

MODULE_EXPORTS = {
    "Dtype",
    "Port",
    "Recipe",
    "SerializedRecipe",
    "XirBuilder",
    "XirModule",
    "add_lib_dir",
    "compile_defaults",
    "compile_loom_to_bytes",
    "decode",
    "lib_dirs",
    "load_recipe",
    "replay_stats",
    "reset_replay_stats",
    "run_recipe_capsule",
    "tuned_recipe_bytes",
}

CLASS_EXPORTS = {
    "Dtype": {"name", "nbytes"},
    "Port": {"dtype", "name", "shape"},
    "Recipe": {"aliases", "inputs", "outputs", "workspace_nbytes"},
    "SerializedRecipe": set(),
    "XirBuilder": {
        "begin_func",
        "binary",
        "broadcast",
        "compare",
        "concatenate",
        "constant",
        "convert",
        "dot",
        "end_func",
        "exact_constant",
        "clamp",
        "finish",
        "gather",
        "pad",
        "param",
        "reduce",
        "repeat_while",
        "reshape",
        "scatter",
        "select",
        "slice",
        "sort",
        "transpose",
        "unary",
    },
    "XirModule": {"entry_name", "function_names", "operation_names", "text"},
}


def public_names(value):
    return {name for name in dir(value) if not name.startswith("__")}


def test_native_module_has_only_the_curated_surface():
    assert public_names(native) == MODULE_EXPORTS


def test_native_classes_have_only_the_curated_surface():
    for name, expected in CLASS_EXPORTS.items():
        assert public_names(getattr(native, name)) == expected


def test_a_recipe_operand_survives_a_pickle_round_trip():
    """A tuned recipe is worth keeping across processes.

    That is what an ahead-of-time cache written to disk is for, and pickling is
    how it is written, so every value the cache holds has to state itself as
    its own constructor arguments.
    """
    import pickle

    dtype = native.Dtype("f32", 4)
    port = native.Port("x", dtype, (4, 8))
    assert pickle.loads(pickle.dumps(dtype)) == dtype
    assert pickle.loads(pickle.dumps(port)) == port


def test_compile_defaults_match_native_defaults():
    from dataclasses import asdict

    from loom import Compile

    defaults = asdict(Compile().resolve())
    assert defaults.pop("transpose_depth") == 0
    assert defaults.pop("scheduling_time_limit_ms") is None
    assert defaults == native.compile_defaults()
    assert defaults["shortlist_budget_ms"] == 600_000


def test_a_compile_option_carries_the_shortlist_budget():
    """The budget travels with the call, so `aot` is not required to set one.

    An ambient context is convenient for a whole training run; a single call
    that wants a bounded search should not have to enter one.
    """
    from loom import Compile

    assert Compile(shortlist_budget_ms=250.0).shortlist_budget_ms == 250.0
    for bad in (0.0, -1.0, float("inf")):
        try:
            Compile(shortlist_budget_ms=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} is not a budget")
