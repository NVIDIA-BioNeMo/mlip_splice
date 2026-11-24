// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Velocity-Verlet integration on the GPU, with an optional velocity-scaling
// stage (the thermostat) run between the two half kicks.

#ifndef SPLICE_MD_VERLET_HPP
#define SPLICE_MD_VERLET_HPP

#include "workspace.hpp"

namespace splice_md {

class verlet_workspace : public workspace {
    public:
        verlet_workspace();
        ~verlet_workspace();

        void SetInterface(int natoms, float dt, float *box, float *masses,
                          float *x, float *v, float *f);
        virtual void process(cudaStream_t stream = 0);

        void set_vscale(workspace *w);
        void set_vcorrection(workspace *w);

        int natoms;
        float dt;

        float *box, *masses;
        float *x, *v, *f;

        workspace *v_correction;
        workspace *v_scale;
};

}

#endif
