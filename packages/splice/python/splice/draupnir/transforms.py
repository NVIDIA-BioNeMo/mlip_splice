# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chain transforms.

A chain is data: op instances with their slot bindings. A transform consumes that data and returns
another executable of the same program (`record`, `profile` — still a `Chain`, so transforms
compose and jit/grad apply) or an analysis of it (`render`).
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import equinox as eqx

from .chain import Chain


def record(chain: Chain) -> Chain:
    """A chain whose final context also carries each step's outputs, as `step<i>_<op>_<slot>`.

    Record names are never overwritten, so they hold each step's value even where the bound name
    is later replaced. Under `jit`, records left unread are dead-code-eliminated.
    """
    specs = tuple(
        spec._replace(
            outs=tuple((slot, (*names, f"step{i}_{spec.op}_{slot}")) for slot, names in spec.outs)
        )
        for i, spec in enumerate(chain.specs)
    )
    return Chain(ops=chain.ops, specs=specs)


class _Tagged(eqx.Module):
    """Wraps an op so every call runs between `start` and `end` (identity pytree passes)."""

    op: Any
    start: Callable = eqx.field(static=True)
    end: Callable = eqx.field(static=True)

    def __call__(self, **inputs):
        return self.end(self.op(**self.start(inputs)))


def profile(chain: Chain, tag: Callable[[str], tuple[Callable, Callable]]) -> Chain:
    """A chain emitting one profiling range per step, named `step<i>_<op>`.

    `tag(name) -> (start, end)` supplies the range primitives; both are identity pytree passes.
    `start` receives the step's input kwargs, `end` its result.
    """
    ops = tuple(
        _Tagged(op, *tag(f"step{i}_{spec.op}"))
        for i, (op, spec) in enumerate(zip(chain.ops, chain.specs, strict=True))
    )
    return Chain(ops=ops, specs=chain.specs)


class Graph(NamedTuple):
    """The static dataflow graph: steps as nodes, names as edges, chain inputs as sources."""

    nodes: tuple[tuple[int, str], ...]  # (step index, op name)
    edges: tuple[tuple[int | str, int, str], ...]  # (producer step | input name, consumer, name)
    sources: frozenset[str]

    def to_dot(self) -> str:
        lines = ["digraph draupnir {"]
        for name in sorted(self.sources):
            lines.append(f'  "in:{name}" [label="{name}", shape=ellipse];')
        for i, op in self.nodes:
            lines.append(f'  "step{i}" [label="{i}: {op}", shape=box];')
        for src, dst, name in self.edges:
            src_id = f"in:{src}" if isinstance(src, str) else f"step{src}"
            lines.append(f'  "{src_id}" -> "step{dst}" [label="{name}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"


def render(chain: Chain) -> Graph:
    """Derive the chain's static dataflow graph: one edge per input read."""
    writer: dict[str, int] = {}
    nodes: list[tuple[int, str]] = []
    edges: list[tuple[int | str, int, str]] = []
    for i, spec in enumerate(chain.specs):
        nodes.append((i, spec.op))
        for _, _, names in spec.ins:
            for name in names:
                edges.append((writer.get(name, name), i, name))
        for _, names in spec.outs:
            for name in names:
                writer[name] = i
    return Graph(nodes=tuple(nodes), edges=tuple(edges), sources=chain.inputs)
