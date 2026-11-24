// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "verlet.hpp"
#include "vec3.hpp"

using namespace splice_md;

__global__ void cuda_update_v(float dt, int natoms, float *masses, float *v, float *f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    float mass = masses[idx];
    float dtm = dt / mass;
    int shift = idx * 3;

    float3 vi = v3::from_array(v + shift);
    float3 fi = v3::from_array(f + shift);
    v3::scale_sadd(vi, fi, dtm);
    v3::to_array(vi, v + shift);
}

__global__ void cuda_update_x(float dt, int natoms, float *box, float *x, float *v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    float3 pbc = v3::from_array(box);
    int shift = idx * 3;

    // `x` advances by THIS call's increment. The original accumulated the
    // step's displacement in a scratch array and then added the running total,
    // which moves the atom by 1.5*v*dt over a full step -- a drift out of phase
    // with the kick, and no longer symplectic. It sampled configurations at
    // 1.5x the reported temperature.
    float3 vi = v3::from_array(v + shift);
    float3 step = v3::scale_by(vi, dt);

    float3 xi = v3::from_array(x + shift);
    v3::sadd(xi, step);
    v3::pbc_unwrap(xi, pbc);
    v3::to_array(xi, x + shift);
}

verlet_workspace::verlet_workspace() {
    v_correction = nullptr;
    v_scale = nullptr;
}

verlet_workspace::~verlet_workspace() {}

void verlet_workspace::SetInterface(int natoms, float dt, float *box, float *masses,
                                    float *x, float *v, float *f) {
    this->natoms = natoms;
    this->dt = dt;

    this->box = box;
    this->masses = masses;

    this->x = x;
    this->v = v;
    this->f = f;

}

void verlet_workspace::set_vscale(workspace *w) {
    this->v_scale = w;
}

void verlet_workspace::set_vcorrection(workspace *w) {
    this->v_correction = w;
}

void verlet_workspace::process(cudaStream_t stream) {
    int nthreads = 96;
    int nblocks = (natoms + nthreads - 1) / nthreads;

    cuda_update_v<<<nblocks, nthreads, 0, stream>>>(dt, natoms, masses, v, f);

    if (v_correction) v_correction->process();
    cuda_update_x<<<nblocks, nthreads, 0, stream>>>(dt * 0.5f, natoms, box, x, v);

    if (v_scale) v_scale->process();
    cuda_update_x<<<nblocks, nthreads, 0, stream>>>(dt * 0.5f, natoms, box, x, v);
}
