# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neighbour lists for MD, and the seam every backend fits behind.

The model wants one thing from a neighbour list: a padded, fixed-length pair of
edge-index arrays and a mask saying how much of them is real. JAX recompiles per
shape, so the length has to be constant across steps even though the edge count
is not. :class:`NeighborBackend` states that contract and nothing else, so a
different search — a batched one, say, or one that lives inside a jit — can be
dropped in without the driver noticing.

:class:`CellNeighborList` is the one we ship: a uniform-grid search in CUDA that
writes the compact, symmetric edge list directly, with no dense intermediate.

Lengths are nm throughout, matching :class:`splice.md.state.MDState`.
"""

from typing import Protocol, runtime_checkable

import cupy as cp
import numpy as np

from splice.md import _ops

# Mirrors `enum cell_status` in _ops_src/md/cell_neighbor.hpp.
_STATUS_HALF_EDGES = 0
_STATUS_EDGE_OVERFLOW = 1
_STATUS_PAGE_OVERFLOW = 2
_STATUS_MAX_ROW = 3
_STATUS_WORDS = 4


@runtime_checkable
class NeighborBackend(Protocol):
    """What the MD driver requires of a neighbour search.

    Implementations own their own buffers and are free to be anything —
    cell list, cluster tiles, a jit-able JAX search — provided they can

    * ``update()`` a list for the state's current positions, and
    * ``edges()`` it as ``(senders, receivers, mask)``, all of a length that
      does not change between calls.

    ``senders`` and ``receivers`` are ``int32`` and index atoms in system order;
    ``mask`` is boolean and true exactly on the real edges. Every pair inside
    the cutoff appears in both directions.
    """

    natom: int
    cutoff: float
    edge_budget: int

    def update(self) -> None: ...

    def edges(self) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]: ...


class CellNeighborList:
    """Uniform-grid neighbour search producing a compact symmetric edge list.

    Atoms are sorted into cells at least ``cutoff + skin`` wide; each atom walks
    the half neighbour shell, and a count pass plus an exclusive scan give every
    atom the offset it writes at. There is no dense ``[natom, pagesize]``
    intermediate to size, overflow or compact afterwards, and no per-cell
    capacity either — cells are ranges over an atom list sorted by cell. The
    sort is stable and ownership goes to the lower atom index, so two builds of
    the same configuration agree array-for-array.

    Parameters
    ----------
    state : MDState
    cutoff : float
        Interaction cutoff, nm. Pairs are emitted strictly inside it.
    skin : float
        Verlet skin, nm. The grid and the candidate search widen to
        ``cutoff + skin``, and the emitted list is still filtered to ``cutoff``,
        so results do not depend on it. Reusing a list across steps is not
        implemented — a skin currently costs a little and saves nothing, which
        is why the default is ``0.0``; see the note in ``update``.
    cell_width : float, optional
        Target cell width, nm. Defaults to ``cutoff + skin``. Wider cells scan
        fewer, fuller cells; narrower ones scan more, emptier ones, and measured
        slower.
    edge_budget : int, optional
        Fixed padded length of the edge arrays. Sized from the first build times
        ``headroom`` when not given.
    page : int
        Scratch row length: the most half-list neighbours one atom may have.
        Liquid water at a 0.6 nm cutoff peaks near 100. Too small is detected
        and corrected on the spot — the kernel reports the length it needed —
        so this is a starting guess, not a constraint.
    headroom : float
        Slack over the first build's edge count when sizing ``edge_budget``.
    """

    def __init__(
        self, state, cutoff, skin=0.0, cell_width=None, edge_budget=None, headroom=1.15, page=128
    ):
        self.state = state
        self.cutoff = float(cutoff)
        self.skin = float(skin)
        self.natom = state.natom
        self._headroom = float(headroom)

        width = float(cell_width) if cell_width else self.cutoff + self.skin
        box = cp.asnumpy(state.box).astype(float)
        if not np.isfinite(cp.asnumpy(state.x)).all():
            raise ValueError(
                "positions contain NaN or infinity; the trajectory has diverged, "
                "so the neighbour list cannot be built"
            )

        self._page = int(page)
        self._box = (float(box[0]), float(box[1]), float(box[2]))
        self._ws = _ops.neighbor.workspace(self.cutoff, self.skin, width)
        # Raises with the reason rather than returning a grid that cannot work.
        _ops.neighbor.setup(self._ws, *self._box, self.natom, self._page)
        self.geometry = _ops.neighbor.geometry(self._ws)

        n = self.natom
        # natom + 1: the scan writes the half-list total one past the last atom.
        self._counts = cp.zeros(n + 1, cp.int32)
        self._offsets = cp.zeros(n + 1, cp.int32)
        self._status = cp.zeros(_STATUS_WORDS, cp.int32)

        self.edge_budget = int(edge_budget) if edge_budget else 0
        # Only an unpinned budget is re-sized after the first build; a caller who
        # named a budget gets exactly that one.
        self._auto_budget = not self.edge_budget
        self._sized = False
        self._senders = None
        self._receivers = None
        if self.edge_budget:
            self._allocate(self.edge_budget)
        else:
            # Enough to complete one build and learn the real count; the
            # permanent budget is set from it. Liquid water sits near 86
            # directed edges per atom.
            self._allocate(int(n * 128))
        self._nedge = 0

    # ---------------------------------------------------------------- setup

    @property
    def live_edges(self):
        """Edges actually found by the last build, both directions.

        Distinct from ``edge_budget``, which is the padded capacity. Watching
        the ratio is the cheapest early warning that a trajectory is going
        wrong: it sits flat while the dynamics are healthy and climbs before
        anything overflows.
        """
        return 2 * self._nedge

    def _allocate(self, budget):
        self.edge_budget = int(budget)
        self._senders = cp.zeros(self.edge_budget, cp.int32)
        self._receivers = cp.zeros(self.edge_budget, cp.int32)
        # Preallocated so `edges()` allocates nothing per step.
        self._slots = cp.arange(self.edge_budget, dtype=cp.int32)
        self._mask = cp.zeros(self.edge_budget, cp.bool_)
        self._bind()

    def _bind(self):
        _ops.neighbor.setinterface(
            self._ws,
            self.natom,
            self.state.addr("box"),
            self.state.addr("x"),
            int(self._counts.data.ptr),
            int(self._offsets.data.ptr),
            int(self._senders.data.ptr),
            int(self._receivers.data.ptr),
            int(self._status.data.ptr),
            self.edge_budget,
        )

    # ---------------------------------------------------------------- build

    def update(self):
        """Wrap positions into the box and rebuild the list.

        Rebuilt every step. A Verlet skin would let the candidate list be reused
        instead, but for a message-passing potential the model does full work on
        every edge it is handed, so a wider list costs more per step than the
        rebuild it saves: measured 1.35x slower at a 0.1 nm skin. The pieces to
        try it again are in place — the grid and the candidate search already
        widen to ``cutoff + skin`` — and what is missing is the staleness test
        and the reuse flag.
        """
        self.state.wrap()
        _ops.neighbor.process(self._ws)
        cp.cuda.Stream.null.synchronize()

        status = cp.asnumpy(self._status)

        # Both capacities are guesses on the first build and exact afterwards,
        # because the kernel reports what it needed. Grow and redo rather than
        # hand back a list that is quietly missing edges.
        if status[_STATUS_PAGE_OVERFLOW]:
            self._page = int(int(status[_STATUS_MAX_ROW]) * self._headroom) + 8
            _ops.neighbor.setup(self._ws, *self._box, self.natom, self._page)
            self._bind()
            _ops.neighbor.process(self._ws)
            cp.cuda.Stream.null.synchronize()
            status = cp.asnumpy(self._status)
            if status[_STATUS_PAGE_OVERFLOW]:
                raise RuntimeError(
                    f"neighbour rows still overflow at page={self._page}; the system "
                    f"densified mid-rebuild"
                )
        self._nedge = int(status[_STATUS_HALF_EDGES])

        # Size the permanent budget from what the first build actually needed.
        # The initial n*128 is a guess big enough to complete one build; liquid
        # water sits near 86 directed edges per atom and a solvated protein near
        # 90, so keeping 128 leaves ~40% of every edge tensor padding -- which
        # costs real time (~9.5% of a step, measured at 35k atoms) and real
        # memory. Shrink once, then leave it alone; the overflow path below still
        # grows it if the system ever densifies.
        if self._auto_budget and not self._sized:
            self._sized = True
            want = int(self._nedge * 2 * self._headroom)
            if want and want < self.edge_budget:
                print(
                    f"[nl] edge budget {self.edge_budget} -> {want} "
                    f"({want / self.natom:.0f}/atom, from {2 * self._nedge} live edges); "
                    f"recompiles once",
                    flush=True,
                )
                self._allocate(want)
                _ops.neighbor.process(self._ws)
                cp.cuda.Stream.null.synchronize()
                status = cp.asnumpy(self._status)
                self._nedge = int(status[_STATUS_HALF_EDGES])

        if status[_STATUS_EDGE_OVERFLOW]:
            grown = int(self._nedge * 2 * self._headroom)
            # Loud, because this is never free: the padded edge axis changes, so
            # the model recompiles, and a run that trips this every step spends
            # all its time compiling while printing nothing. Repeated growth
            # also means the neighbour count itself is climbing -- which for a
            # fixed box is a diverging simulation, not a sizing problem.
            print(
                f"[nl] edge budget OVERFLOW at {self.edge_budget} "
                f"({self.edge_budget / self.natom:.0f}/atom); growing to {grown} "
                f"({grown / self.natom:.0f}/atom) for {2 * self._nedge} live edges "
                f"-- forces a recompile",
                flush=True,
            )
            self._allocate(grown)
            _ops.neighbor.process(self._ws)
            cp.cuda.Stream.null.synchronize()
            status = cp.asnumpy(self._status)
            if status[_STATUS_EDGE_OVERFLOW]:
                raise RuntimeError(f"edge budget still too small after growing to {grown}")
            self._nedge = int(status[_STATUS_HALF_EDGES])

    def edges(self):
        """Padded ``(senders, receivers, mask)`` at a fixed length, for the model.

        The kernel writes the forward half at ``[0, nedge)`` and its mirror at
        ``[nedge, 2*nedge)``; everything past that is masked off. The arrays are
        the same objects every call, so the model sees one shape and compiles once.
        """
        if self._nedge == 0:
            self.update()
        if self.edge_budget == 0 or self._senders is None:
            raise RuntimeError("neighbour list has no edge buffers")
        cp.less(self._slots, 2 * self._nedge, out=self._mask)
        # CuPy's writes must land before JAX reads the same memory over DLPack.
        cp.cuda.Stream.null.synchronize()
        return self._senders, self._receivers, self._mask

    def pairs(self):
        """The real edges only, unpadded: ``(senders, receivers)``, both int32."""
        live = 2 * self._nedge
        return self._senders[:live], self._receivers[:live]

    @property
    def nedge(self):
        """Directed edges in the last build."""
        return 2 * self._nedge

    def __del__(self):
        ws = getattr(self, "_ws", None)
        if ws is not None:
            _ops.neighbor.release(ws)
            self._ws = None
