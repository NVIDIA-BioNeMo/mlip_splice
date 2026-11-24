# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The neighbour list against a brute-force reference.

The interesting property is not "roughly the right number of neighbours" — a
list can be wrong by a percent and still look plausible — but the exact edge
SET. These compare against every pair under the minimum-image convention.
"""

import cupy as cp
import numpy as np
import pytest
from splice.md.neighbors import CellNeighborList, NeighborBackend
from splice.md.state import MDState

CUTOFF = 0.6  # nm


def water_box(n_atoms, seed=0):
    """Rattled water on a shuffled lattice at liquid density, in nm.

    Shuffled, not taken in meshgrid order: filling the lattice in order packs a
    slab and leaves the rest of the cell empty, which inflates the neighbour
    count by about a fifth and makes the test easier than it should be.
    """
    rng = np.random.default_rng(seed)
    n_mol = max(1, n_atoms // 3)
    side = (n_mol * 3 / 100.3) ** (1 / 3)  # atoms/nm^3
    n_side = int(np.ceil(n_mol ** (1 / 3)))
    step = side / n_side
    grid = np.stack(np.meshgrid(*[np.arange(n_side)] * 3, indexing="ij"), -1).reshape(-1, 3)
    centres = rng.permutation(grid)[:n_mol] * step + step / 2
    pos = np.concatenate(
        [centres, centres + np.array([0.096, 0.0, 0.0]), centres + np.array([-0.024, 0.093, 0.0])]
    )
    pos = np.mod(pos + rng.normal(0.0, 0.025, pos.shape), side)
    masses = np.concatenate([np.full(n_mol, 16.0), np.full(2 * n_mol, 1.0)])
    return pos.astype(np.float32), float(side), masses.astype(np.float32)


def brute_force(pos, side):
    """Every unordered pair inside the cutoff, by minimum image."""
    x = cp.asarray(pos)
    L = cp.float32(side)
    rc2 = np.float32(CUTOFF * CUTOFF)
    n = pos.shape[0]
    pairs = set()
    for s in range(0, n, 2048):
        e = min(s + 2048, n)
        d = x[s:e, None, :] - x[None, :, :]
        d -= L * cp.round(d / L)
        ii, jj = cp.nonzero((d * d).sum(-1) < rc2)
        for a, b in zip(cp.asnumpy(ii) + s, cp.asnumpy(jj), strict=True):
            if a != b:
                pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    return pairs


def unordered(senders, receivers, nedge):
    s = cp.asnumpy(senders)[:nedge]
    r = cp.asnumpy(receivers)[:nedge]
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(s, r, strict=True)}


def build(n_atoms, **kw):
    pos, side, masses = water_box(n_atoms)
    state = MDState(pos, np.array([side] * 3, np.float32), masses)
    nl = CellNeighborList(state, CUTOFF, **kw)
    nl.update()
    return nl, pos, side


@pytest.mark.parametrize("n_atoms", [3000, 12000])
def test_edge_set_is_exact(n_atoms):
    nl, pos, side = build(n_atoms)
    s, r, _ = nl.edges()
    got = unordered(s, r, nl.nedge)
    ref = brute_force(pos, side)
    assert got == ref, (
        f"{len(got - ref)} edges beyond the cutoff, {len(ref - got)} inside it missing"
    )


def test_list_is_symmetric_and_has_no_self_edges():
    nl, _, _ = build(3000)
    s, r, _ = nl.edges()
    n = nl.nedge
    sn, rn = cp.asnumpy(s)[:n], cp.asnumpy(r)[:n]
    assert not (sn == rn).any()
    assert set(zip(sn.tolist(), rn.tolist(), strict=True)) == set(
        zip(rn.tolist(), sn.tolist(), strict=True)
    )


def test_rebuild_is_reproducible():
    """Same positions, same arrays — element for element.

    Cells are filled by a stable sort and ownership within a cell goes to the
    lower atom index, so nothing here depends on the order atomics happened to
    fire in. Without that the edge SET would still be right but the force
    summation order would wander between runs.
    """
    nl, _, _ = build(3000)
    s1, r1, _ = nl.edges()
    first = (cp.asnumpy(s1).copy(), cp.asnumpy(r1).copy(), nl.nedge)
    nl.update()
    s2, r2, _ = nl.edges()
    assert nl.nedge == first[2]
    assert (cp.asnumpy(s2) == first[0]).all()
    assert (cp.asnumpy(r2) == first[1]).all()


def test_skin_does_not_change_the_edge_set():
    """A skin widens the search, never what is emitted."""
    a, _, _ = build(3000)
    b, _, _ = build(3000, skin=0.1)
    assert unordered(*a.edges()[:2], a.nedge) == unordered(*b.edges()[:2], b.nedge)


def test_undersized_scratch_rows_are_corrected_not_ignored():
    """`page` is a guess; the kernel reports what it needed and the list resizes.

    The original had no bounds check at all here: an atom with more neighbours
    than its row could hold wrote into the next atom's row, which corrupts the
    list of a completely unrelated atom and reports success.
    """
    tight, pos, side = build(3000, page=8)
    assert unordered(*tight.edges()[:2], tight.nedge) == brute_force(pos, side)
    assert tight.geometry["page"] == 128 or tight._page > 8


def test_a_thin_axis_collapses_to_one_cell_and_stays_exact():
    """Below three cells an axis would have its half shell wrap onto itself, and
    every pair across it would be counted twice. Instead the axis collapses to a
    single cell and is minimum-imaged directly, which is exact for any box the
    minimum image convention itself allows."""
    pos, side, masses = water_box(3000)
    state = MDState(pos, np.array([side] * 3, np.float32), masses)
    nl = CellNeighborList(state, CUTOFF, cell_width=side / 2.5)
    nl.update()
    assert nl.geometry["cells"] == (1, 1, 1)
    assert unordered(*nl.edges()[:2], nl.nedge) == brute_force(pos, side)


def test_a_box_below_twice_the_cutoff_is_refused():
    """There minimum image does not hold, so no neighbour list is well defined."""
    pos, _, masses = water_box(3000)
    tiny = np.array([CUTOFF * 1.5] * 3, np.float32)
    state = MDState(np.mod(pos, tiny), tiny, masses)
    with pytest.raises(RuntimeError, match="minimum image"):
        CellNeighborList(state, CUTOFF)


def test_cell_list_satisfies_the_backend_protocol():
    nl, _, _ = build(3000)
    assert isinstance(nl, NeighborBackend)
