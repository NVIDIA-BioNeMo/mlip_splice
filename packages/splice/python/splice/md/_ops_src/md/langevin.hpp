// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Langevin thermostat: an Ornstein-Uhlenbeck velocity update applied as the
// velocity-scaling stage of the Verlet integrator.

#ifndef SPLICE_MD_LANGEVIN_HPP
#define SPLICE_MD_LANGEVIN_HPP

#include "workspace.hpp"
#include <curand.h>

namespace splice_md {

class langevin_workspace : public workspace {
    public:
        langevin_workspace();
        ~langevin_workspace();
        
        void SetInterface(float *pos, float *vel, float *frc);
        void setup(float dt, float temperature, float friction, int natoms, float *masses);
        virtual void process(cudaStream_t stream = 0);

        float dt;
        float temperature;
        float friction;
        
        int natoms;
        float *pos;
        float *vel;
        float *frc;

        float *alpha;
        float *beta;
        float *masses;

        float *random_numbers;
        curandGenerator_t rand;
};

}
#endif
