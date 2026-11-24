// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A cell-list neighbour search that emits a COMPACT, SYMMETRIC pair list.
//
// Atoms are sorted into a uniform grid of cells at least as wide as the reach;
// each atom then walks the half neighbour shell once, keeping the partners
// inside the cutoff in a per-atom row. An exclusive scan over the row lengths
// gives every atom the offset its edges start at, and a scatter pass — one
// thread per (atom, slot) — moves them into the final arrays and mirrors them.
//
// The rows are scratch, not output: they are read once, immediately, by the
// scatter. Walking a second time instead of buffering them was measured 35-57%
// slower at 300k-1M atoms, which is where the walk dominates everything else.
//
// Three properties follow from that shape, and they are the reason for it:
//
//   * cells are CSR ranges over an atom list sorted by cell, not fixed-width
//     rows, so there is no per-cell capacity to exceed;
//   * the sort is stable and ownership within a cell goes to the lower atom
//     index, so the output is a pure function of the configuration — two builds
//     of the same positions agree element for element.
//
// The list is emitted in both directions: the forward half occupies
// `[0, nedge)` and its mirror `[nedge, 2*nedge)`.
//
// Lengths are nm. Positions must be wrapped into `[0, box)` before `process`.

#ifndef SPLICE_MD_CELL_NEIGHBOR_HPP
#define SPLICE_MD_CELL_NEIGHBOR_HPP

#include "workspace.hpp"

// Upper bound on the half neighbour shell. With cells at or above the reach the
// shell is 13; the bound only bites if someone asks for cells much smaller than
// the cutoff, and it is checked rather than assumed.
#define CELLNEIGH_MAX 1024

namespace splice_md {

    // Slots in the device status block. It is one small array so a build reports
    // everything it noticed in a single copy back to the host, rather than one
    // synchronisation per condition.
    enum cell_status {
        CELL_STATUS_HALF_EDGES = 0,     // half-list edges the last build found
        CELL_STATUS_EDGE_OVERFLOW = 1,  // 1 if the edge list ran out of room
        CELL_STATUS_PAGE_OVERFLOW = 2,  // 1 if some atom's scratch row ran out
        CELL_STATUS_MAX_ROW = 3,        // longest row, so the caller can resize exactly
        CELL_STATUS_WORDS = 4,
    };

    class cell_neigh_workspace : public workspace {
    public:
        // `cell_width` is a target; the real width is the box divided by a whole
        // number of cells, and is never below `cutoff + skin`. Pass 0 for the
        // default, which is `cutoff + skin` itself.
        cell_neigh_workspace(float cutoff, float skin = 0.0f, float cell_width = 0.0f);
        ~cell_neigh_workspace();

        // Sizes the grid and allocates everything that depends on it. The box is
        // passed on the HOST: reading it back from the device here would put a
        // synchronisation in every step of the MD loop, and under NVT it never
        // changes. Returns the cell count, or -1 with `last_error` set.
        // `page` is the scratch row length: the most half-list neighbours an
        // atom may have. Overflow is counted and reported, never silent, and
        // `CELL_STATUS_MAX_ROW` says exactly what it should have been.
        int Setup(const float *box_host, int natom, int page);

        // `counts` and `offsets` hold natom+1 entries — the scan writes the
        // half-list total one past the last atom. `senders` and `receivers` are
        // the padded edge arrays the model reads, `edge_capacity` long each.
        void SetInterface(int natom, float *box, float *pos, int *counts, int *offsets,
                          int *senders, int *receivers, int *status, int edge_capacity);

        void process(cudaStream_t stream = 0) override;

        // Empty unless something went wrong; cleared by the next `process`.
        const char *last_error() const { return error_[0] ? error_ : nullptr; }

        int ncell(int d) const { return cell_dim[d]; }
        float cell_size(int d) const { return cw_real[d]; }
        int shell_size() const { return nshell; }
        int cell_total() const { return ncell_total; }
        int page_size() const { return page; }

        float cutoff, skin, cell_width;

    private:
        void Bin(cudaStream_t stream);
        void Walk(cudaStream_t stream);
        void Scatter(cudaStream_t stream);
        bool check(const char *where);
        void free_all();

        // Caller-owned device memory, handed over once.
        float *box, *pos;
        int *counts, *offsets, *senders, *receivers, *status;
        int natom, edge_capacity, natom_alloc;

        // Ours.
        int cell_dim[3], ncell_total;
        float cw_real[3];
        float minimg[3];                   // box length on single-cell axes, else 0
        int nshell;                        // half-shell cells, home excluded
        int *shell_x, *shell_y, *shell_z;
        int *cell_of, *cell_of_sorted;     // per atom, before and after the sort
        int *atom_seq, *sorted_atom;       // identity, and atoms in cell order
        int *cell_count, *cell_start;      // ncell+1 each: the CSR ranges
        float *cell_xyz;                   // positions gathered into cell order
        int *page_list;                    // natom x page scratch rows
        int page;
        void *scan_temp;
        size_t scan_bytes;
        int cell_bits;                     // radix bits the cell ids need

        char error_[256];
    };
}

#endif
