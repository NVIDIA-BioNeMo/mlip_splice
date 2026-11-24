// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "langevin.hpp"
#include "vec3.hpp"
#include <cuda_runtime.h>
#include <cstdio>
using namespace splice_md;

const float boltzmann_constant = 1.98720426e-3; // kcal/mol/K

langevin_workspace::langevin_workspace() {    
    alpha = nullptr;
    beta = nullptr;
    random_numbers = nullptr;
    rand = nullptr;
}

langevin_workspace::~langevin_workspace() {
    if (alpha != nullptr) cudaFree(alpha);
    if (beta != nullptr) cudaFree(beta);
    if (random_numbers != nullptr) cudaFree(random_numbers);
    if (rand != nullptr) curandDestroyGenerator(rand);
}

void langevin_workspace::SetInterface(float *pos, float *vel, float *frc) {
    this->pos = pos;
    this->vel = vel;
    this->frc = frc;
}

__global__ void cuda_compute_prefactors(float *alpha, float *beta, float dt, float temperature, float friction, int natoms, float *masses) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    float mass = masses[idx];
    float a = exp(- friction * dt);

    float bsq = (1.0f - a * a) * (boltzmann_constant * temperature / mass) ;
    float b = sqrtf(bsq);

    alpha[idx] = a;
    beta[idx] = b;

    //if (idx == 0) printf("alpha: %f, beta: %f, mass: %f\n", alpha[idx], beta[idx], mass);
}

void langevin_workspace::setup(float dt, float temperature, float friction, int natoms, float *masses) {
    this->dt = dt;
    this->natoms = natoms;
    this->temperature = temperature;
    this->friction = friction;
    this->masses = masses;
    
    if(alpha != nullptr) cudaFree(alpha);
    if(beta != nullptr) cudaFree(beta);
    
    cudaMalloc(&alpha, natoms * sizeof(float));
    cudaMalloc(&beta, natoms * sizeof(float));

    int nthreads = 128;
    int nblocks = (natoms + nthreads - 1) / nthreads;
    cuda_compute_prefactors<<<nblocks, nthreads>>>(alpha, beta, dt, temperature, friction, natoms, masses);

    if (random_numbers != nullptr) cudaFree(random_numbers);
    cudaMalloc(&random_numbers, natoms * sizeof(float) * 4); // to keep the n even
    curandCreateGenerator(&rand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rand, 0);
}

__global__ void cuda_langevin_update(float *pos, float *vel, float *frc, float *alpha, float *beta, float *masses, int natoms, float dt, float *random_numbers) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= natoms) return;

    float a = alpha[idx];
    float b = beta[idx];

    float3 v = v3::from_array(vel + idx * 3);
    v3::scale(v, a);

    float3 r = v3::from_array(random_numbers + idx * 3);
    v3::scale(r, b);
    v3::sadd(v, r);

    v3::to_array(v, vel + idx * 3); 
    
    //if (idx == 100) printf("a: %f, b: %f, r: %f %f %f\n", a, b, r.x, r.y, r.z);
}

void langevin_workspace::process(cudaStream_t stream) {
    curandGenerateNormal(rand, random_numbers, natoms * 4, 0.0f, 1.0f);

    int nthreads = 128;
    int nblocks = (natoms + nthreads - 1) / nthreads;
    cuda_langevin_update<<<nblocks, nthreads>>>(pos, vel, frc, alpha, beta, masses, natoms, dt, random_numbers);
}
