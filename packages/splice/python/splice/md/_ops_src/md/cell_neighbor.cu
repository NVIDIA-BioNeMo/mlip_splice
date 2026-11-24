// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cell_neighbor.hpp"

#include <cub/cub.cuh>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

using namespace splice_md;

// One block per cell; the block strides over the cell's atoms, so the block
// size is a launch-tuning choice rather than a function of how full a cell is.
// (Sizing the block from a cell capacity, as the original did, silently fails
// to launch at all once a cell can hold more than 1024 atoms.)
#define CELL_BLOCK 128

// ---------------------------------------------------------------- binning

__global__ void cell_assign_kernel(int natom, int nx, int ny, int nz,
                                   float cx, float cy, float cz,
                                   const float *__restrict__ pos,
                                   int *__restrict__ cell_of, int *__restrict__ atom_seq,
                                   int *__restrict__ cell_count)
{
    int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= natom) return;

    float px = pos[atom * 3 + 0], py = pos[atom * 3 + 1], pz = pos[atom * 3 + 2];

    // Positions arrive wrapped into [0, box), but floating-point wrapping can
    // land exactly on the upper edge and a NaN would index arbitrarily. Clamp
    // rather than trust: the result is a raw array index.
    int ix = min(max((int) floorf(px / cx), 0), nx - 1);
    int iy = min(max((int) floorf(py / cy), 0), ny - 1);
    int iz = min(max((int) floorf(pz / cz), 0), nz - 1);

    int icell = ix + iy * nx + iz * nx * ny;
    cell_of[atom] = icell;
    atom_seq[atom] = atom;
    atomicAdd(&cell_count[icell], 1);
}

// Positions in cell order, so the shell walk reads each cell contiguously.
__global__ void cell_gather_kernel(int natom, const int *__restrict__ sorted_atom,
                                   const float *__restrict__ pos, float *__restrict__ cell_xyz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= natom) return;
    int atom = sorted_atom[i];
    cell_xyz[i * 3 + 0] = pos[atom * 3 + 0];
    cell_xyz[i * 3 + 1] = pos[atom * 3 + 1];
    cell_xyz[i * 3 + 2] = pos[atom * 3 + 2];
}

// ------------------------------------------------------- the shell walk
//
// One block per cell, striding over the cell's atoms. Each atom writes the
// partners it owns into its own scratch row and records how many there were.

// `minimg` is the box length on axes the grid could not divide into three or
// more cells, and 0 on the others. A single-cell axis has no shell offset to
// carry its periodic shift, so the pair distance is minimum-imaged along it
// directly; a multi-cell axis gets its shift from the cell offset instead.
__device__ __forceinline__ void min_image(float &dx, float &dy, float &dz,
                                          float mx, float my, float mz)
{
    if (mx > 0.0f) dx -= mx * rintf(dx / mx);
    if (my > 0.0f) dy -= my * rintf(dy / my);
    if (mz > 0.0f) dz -= mz * rintf(dz / mz);
}

__global__ void cell_walk_kernel(
    float cutoff_sq, int nx, int ny, int nz, const float *__restrict__ box,
    float mgx, float mgy, float mgz,
    const int *__restrict__ shell_x, const int *__restrict__ shell_y,
    const int *__restrict__ shell_z, int nshell,
    const int *__restrict__ cell_start, const int *__restrict__ sorted_atom,
    const float *__restrict__ cell_xyz,
    int *__restrict__ counts, int *__restrict__ page_list, int page,
    int *__restrict__ status)
{
    int cell_a = blockIdx.x;
    int a0 = cell_start[cell_a], a1 = cell_start[cell_a + 1];
    float bx = box[0], by = box[1], bz = box[2];
    int nxny = nx * ny;

    int caz = cell_a / nxny;
    int cay = (cell_a - caz * nxny) / nx;
    int cax = cell_a - caz * nxny - cay * nx;

    for (int ia = a0 + threadIdx.x; ia < a1; ia += blockDim.x) {
        int atom = sorted_atom[ia];
        float px = cell_xyz[ia * 3 + 0], py = cell_xyz[ia * 3 + 1], pz = cell_xyz[ia * 3 + 2];
        int *row = page_list + (long) atom * page;
        int n = 0;

        // The home cell. Ownership goes to the lower GLOBAL ATOM INDEX; the sort
        // is stable, so within a cell `sorted_atom` ascends and "later in the
        // cell" and "higher index" are the same thing.
        //
        // The distance test here is not optional. The original emitted every
        // same-cell partner unconditionally, which admits pairs out to the cell
        // diagonal — measured at 0.15 and 0.45 extra edges per atom for 3k and
        // 12.5k water boxes, every one of them beyond the cutoff.
        for (int ib = ia + 1; ib < a1; ib++) {
            float dx = cell_xyz[ib * 3 + 0] - px;
            float dy = cell_xyz[ib * 3 + 1] - py;
            float dz = cell_xyz[ib * 3 + 2] - pz;
            min_image(dx, dy, dz, mgx, mgy, mgz);
            if (dx * dx + dy * dy + dz * dz >= cutoff_sq) continue;
            if (n < page) row[n] = sorted_atom[ib];
            n++;
        }

        for (int c = 0; c < nshell; c++) {
            float xp = 0.0f, yp = 0.0f, zp = 0.0f;
            int cbx = cax + shell_x[c];
            if (cbx < 0) { cbx += nx; xp = -bx; } else if (cbx >= nx) { cbx -= nx; xp = bx; }
            int cby = cay + shell_y[c];
            if (cby < 0) { cby += ny; yp = -by; } else if (cby >= ny) { cby -= ny; yp = by; }
            int cbz = caz + shell_z[c];
            if (cbz < 0) { cbz += nz; zp = -bz; } else if (cbz >= nz) { cbz -= nz; zp = bz; }

            int cell_b = cbx + cby * nx + cbz * nxny;
            int b0 = cell_start[cell_b], b1 = cell_start[cell_b + 1];
            for (int ib = b0; ib < b1; ib++) {
                float dx = cell_xyz[ib * 3 + 0] - px + xp;
                float dy = cell_xyz[ib * 3 + 1] - py + yp;
                float dz = cell_xyz[ib * 3 + 2] - pz + zp;
                min_image(dx, dy, dz, mgx, mgy, mgz);
                if (dx * dx + dy * dy + dz * dz >= cutoff_sq) continue;
                if (n < page) row[n] = sorted_atom[ib];
                n++;
            }
        }
        // The count is the TRUE count even when the row overflowed, so a caller
        // that resizes knows exactly what to resize to.
        counts[atom] = n;
        if (n > page) atomicMax(&status[CELL_STATUS_PAGE_OVERFLOW], 1);
        atomicMax(&status[CELL_STATUS_MAX_ROW], n);
    }
}

// One thread per (atom, slot). Thread-per-atom instead is 12x slower at 300k:
// the rows differ in length, so a thread-per-atom launch serialises the whole
// warp on its longest member.
__global__ void cell_scatter_kernel(
    int natom, int page, const int *__restrict__ counts, const int *__restrict__ offsets,
    const int *__restrict__ page_list, int *__restrict__ out_s, int *__restrict__ out_r,
    int edge_capacity, int *__restrict__ status)
{
    long t = (long) blockIdx.x * blockDim.x + threadIdx.x;
    long atom = t / page;
    int k = (int) (t - atom * page);
    if (atom >= natom || k >= counts[atom]) return;

    int at = offsets[atom] + k;
    int j = page_list[atom * (long) page + k];
    int nedge = offsets[natom];
    // The mirror sits at `nedge + at`, so that is the index the capacity has to
    // cover — not `at`.
    if (nedge + at < edge_capacity) {
        out_s[at] = (int) atom;      out_r[at] = j;
        out_s[nedge + at] = j;       out_r[nedge + at] = (int) atom;
    } else {
        atomicMax(&status[CELL_STATUS_EDGE_OVERFLOW], 1);
    }
}

// cub's exclusive sum over natom+1 counts puts the total at offsets[natom], so
// long as the extra count is zero.
__global__ void cell_clear_tail(int *counts, int natom) { counts[natom] = 0; }

// ---------------------------------------------------------------- host side

cell_neigh_workspace::cell_neigh_workspace(float cutoff, float skin, float cell_width)
    : cutoff(cutoff), skin(skin)
{
    this->cell_width = (cell_width > 0.0f) ? cell_width : (cutoff + skin);
    cell_dim[0] = cell_dim[1] = cell_dim[2] = 0;
    ncell_total = 0; nshell = 0; natom = 0; natom_alloc = 0; edge_capacity = 0;
    shell_x = shell_y = shell_z = nullptr;
    cell_of = cell_of_sorted = atom_seq = sorted_atom = nullptr;
    cell_count = cell_start = nullptr;
    cell_xyz = nullptr; page_list = nullptr; page = 0;
    scan_temp = nullptr; scan_bytes = 0; cell_bits = 32;
    minimg[0] = minimg[1] = minimg[2] = 0.0f;
    box = pos = nullptr;
    counts = offsets = senders = receivers = status = nullptr;
    error_[0] = '\0';
}

void cell_neigh_workspace::free_all()
{
    cudaFree(shell_x); cudaFree(shell_y); cudaFree(shell_z);
    cudaFree(cell_of); cudaFree(cell_of_sorted); cudaFree(atom_seq); cudaFree(sorted_atom);
    cudaFree(cell_count); cudaFree(cell_start); cudaFree(cell_xyz);
    cudaFree(page_list); cudaFree(scan_temp);
    shell_x = shell_y = shell_z = nullptr;
    cell_of = cell_of_sorted = atom_seq = sorted_atom = nullptr;
    cell_count = cell_start = nullptr;
    cell_xyz = nullptr; page_list = nullptr; scan_temp = nullptr;
}

cell_neigh_workspace::~cell_neigh_workspace() { free_all(); }

bool cell_neigh_workspace::check(const char *where)
{
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) return true;
    snprintf(error_, sizeof(error_), "%s: %s", where, cudaGetErrorString(e));
    return false;
}

int cell_neigh_workspace::Setup(const float *box_host, int natom, int page)
{
    error_[0] = '\0';
    float reach = cutoff + skin;
    float width = fmaxf(cell_width, reach);

    // A half shell of radius one needs three cells on an axis: with two, the +1
    // and -1 offsets land on the same cell and every pair across it comes out
    // twice. Rather than dedupe afterwards, a thin axis collapses to a SINGLE
    // cell and is minimum-imaged directly — correct for any box the minimum
    // image convention itself allows, which is the real constraint.
    int n[3];
    for (int d = 0; d < 3; d++) {
        n[d] = (int) floorf(box_host[d] / width);
        if (n[d] < 3) {
            if (box_host[d] < 2.0f * (cutoff + skin)) {
                snprintf(error_, sizeof(error_),
                         "box axis %d is %.4f nm, under twice the reach %.4f nm "
                         "(cutoff %.4f + skin %.4f). No neighbour list can be built: "
                         "the minimum image convention itself does not hold",
                         d, box_host[d], cutoff + skin, cutoff, skin);
                return -1;
            }
            n[d] = 1;
        }
    }

    bool same = (n[0] == cell_dim[0] && n[1] == cell_dim[1] && n[2] == cell_dim[2]
                 && natom <= natom_alloc && page <= this->page);
    for (int d = 0; d < 3; d++) {
        cell_dim[d] = n[d];
        cw_real[d] = box_host[d] / n[d];
        minimg[d] = (n[d] == 1) ? box_host[d] : 0.0f;
    }
    ncell_total = cell_dim[0] * cell_dim[1] * cell_dim[2];
    this->natom = natom;
    if (same) return ncell_total;
    natom_alloc = natom;
    this->page = page > 0 ? page : 128;

    // The half shell: cells at a non-negative offset in (z, y, x) order,
    // excluding the home cell, whose nearest corner is within reach.
    // A single-cell axis carries no offset: its periodicity is minimum-imaged.
    int mx = cell_dim[0] == 1 ? 0 : (int) ceilf(reach / cw_real[0]);
    int my = cell_dim[1] == 1 ? 0 : (int) ceilf(reach / cw_real[1]);
    int mz = cell_dim[2] == 1 ? 0 : (int) ceilf(reach / cw_real[2]);
    int hx[CELLNEIGH_MAX], hy[CELLNEIGH_MAX], hz[CELLNEIGH_MAX];
    nshell = 0;
    for (int ix = -mx; ix <= mx; ix++)
      for (int iy = -my; iy <= my; iy++)
        for (int iz = -mz; iz <= mz; iz++) {
            if (iz < 0) continue;
            if (iz == 0 && iy < 0) continue;
            if (iz == 0 && iy == 0 && ix <= 0) continue;
            float dx = (abs(ix) > 1 ? abs(ix) - 1 : 0) * cw_real[0];
            float dy = (abs(iy) > 1 ? abs(iy) - 1 : 0) * cw_real[1];
            float dz = (abs(iz) > 1 ? abs(iz) - 1 : 0) * cw_real[2];
            if (dx * dx + dy * dy + dz * dz > reach * reach) continue;
            if (nshell >= CELLNEIGH_MAX) {
                snprintf(error_, sizeof(error_),
                         "the half neighbour shell needs more than %d cells; cell width "
                         "%.4f nm is far below the reach %.4f nm", CELLNEIGH_MAX,
                         cw_real[0], reach);
                return -1;
            }
            hx[nshell] = ix; hy[nshell] = iy; hz[nshell] = iz; nshell++;
        }

    free_all();
    cudaMalloc(&shell_x, nshell * sizeof(int));
    cudaMalloc(&shell_y, nshell * sizeof(int));
    cudaMalloc(&shell_z, nshell * sizeof(int));
    cudaMemcpy(shell_x, hx, nshell * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(shell_y, hy, nshell * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(shell_z, hz, nshell * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&cell_of, (size_t) natom * sizeof(int));
    cudaMalloc(&cell_of_sorted, (size_t) natom * sizeof(int));
    cudaMalloc(&atom_seq, (size_t) natom * sizeof(int));
    cudaMalloc(&sorted_atom, (size_t) natom * sizeof(int));
    cudaMalloc(&cell_count, (size_t) (ncell_total + 1) * sizeof(int));
    cudaMalloc(&cell_start, (size_t) (ncell_total + 1) * sizeof(int));
    cudaMalloc(&cell_xyz, (size_t) natom * 3 * sizeof(float));
    cudaMalloc(&page_list, (size_t) natom * this->page * sizeof(int));

    cell_bits = 1;
    while ((1 << cell_bits) < ncell_total) cell_bits++;

    // One temp buffer for both the scans and the sort.
    size_t a = 0, b = 0, c = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, a, (int *) nullptr, (int *) nullptr, natom + 1);
    cub::DeviceScan::ExclusiveSum(nullptr, b, (int *) nullptr, (int *) nullptr, ncell_total + 1);
    cub::DeviceRadixSort::SortPairs(nullptr, c, (int *) nullptr, (int *) nullptr,
                                    (int *) nullptr, (int *) nullptr, natom, 0, cell_bits);
    scan_bytes = a > b ? a : b;
    if (c > scan_bytes) scan_bytes = c;
    cudaMalloc(&scan_temp, scan_bytes);

    if (!check("cell neighbour list setup")) return -1;
    return ncell_total;
}

void cell_neigh_workspace::SetInterface(int natom, float *box, float *pos, int *counts,
                                        int *offsets, int *senders, int *receivers,
                                        int *status, int edge_capacity)
{
    this->natom = natom;
    this->box = box; this->pos = pos;
    this->counts = counts; this->offsets = offsets;
    this->senders = senders; this->receivers = receivers;
    this->status = status; this->edge_capacity = edge_capacity;
}

void cell_neigh_workspace::Bin(cudaStream_t stream)
{
    cudaMemsetAsync(cell_count, 0, (size_t) (ncell_total + 1) * sizeof(int), stream);
    cudaMemsetAsync(status, 0, CELL_STATUS_WORDS * sizeof(int), stream);

    int threads = 256, blocks = (natom + threads - 1) / threads;
    cell_assign_kernel<<<blocks, threads, 0, stream>>>(
        natom, cell_dim[0], cell_dim[1], cell_dim[2], cw_real[0], cw_real[1], cw_real[2],
        pos, cell_of, atom_seq, cell_count);

    cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, cell_count, cell_start,
                                  ncell_total + 1, stream);
    // Stable, and fed in atom order, so each cell's slice comes out ascending in
    // atom index. That is what makes the edge order reproducible.
    cub::DeviceRadixSort::SortPairs(scan_temp, scan_bytes, cell_of, cell_of_sorted,
                                    atom_seq, sorted_atom, natom, 0, cell_bits, stream);
    cell_gather_kernel<<<blocks, threads, 0, stream>>>(natom, sorted_atom, pos, cell_xyz);
}

void cell_neigh_workspace::Walk(cudaStream_t stream)
{
    cell_clear_tail<<<1, 1, 0, stream>>>(counts, natom);
    // Filtered at the cutoff, never at cutoff + skin: the skin widens the grid
    // so a future build can be reused across steps, but what is emitted today
    // must not depend on it.
    cell_walk_kernel<<<ncell_total, CELL_BLOCK, 0, stream>>>(
        cutoff * cutoff, cell_dim[0], cell_dim[1], cell_dim[2], box,
        minimg[0], minimg[1], minimg[2],
        shell_x, shell_y, shell_z, nshell, cell_start, sorted_atom, cell_xyz,
        counts, page_list, page, status);
    cub::DeviceScan::ExclusiveSum(scan_temp, scan_bytes, counts, offsets, natom + 1, stream);
}

void cell_neigh_workspace::Scatter(cudaStream_t stream)
{
    long slots = (long) natom * page;
    int threads = 256;
    long blocks = (slots + threads - 1) / threads;
    cell_scatter_kernel<<<(int) blocks, threads, 0, stream>>>(
        natom, page, counts, offsets, page_list, senders, receivers,
        edge_capacity, status);
    // The half-list total, so the caller can size and mask without recomputing.
    cudaMemcpyAsync(status + CELL_STATUS_HALF_EDGES, offsets + natom, sizeof(int),
                    cudaMemcpyDeviceToDevice, stream);
}

void cell_neigh_workspace::process(cudaStream_t stream)
{
    error_[0] = '\0';
    if (ncell_total == 0) {
        snprintf(error_, sizeof(error_), "process() before a successful Setup()");
        return;
    }
    Bin(stream);      if (!check("cell binning")) return;
    Walk(stream);     if (!check("neighbour shell walk")) return;
    Scatter(stream);  check("edge scatter");
}
