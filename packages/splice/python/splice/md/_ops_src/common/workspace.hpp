// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Base class for a GPU work unit: a long-lived object that owns its device
// scratch, is handed raw device pointers once, and is then stepped.
//
// Launch geometry is held here rather than at each call site so it can be
// overridden from the environment while tuning:
//
//     SPLICE_MD_LAUNCH_<NAME>=<blocks-per-SM>x<warps-per-block>

#ifndef SPLICE_MD_WORKSPACE_HPP
#define SPLICE_MD_WORKSPACE_HPP

#define WARP_SIZE 32
#define MAX_LAUNCH_CONFIGS 16

#include <cuda_runtime.h>

#define ENV_VAR_PREFIX (char *)"SPLICE_MD_LAUNCH"

namespace splice_md {

    class workspace {
    public:
        workspace();
        virtual ~workspace() {};

        virtual void process(cudaStream_t stream = 0) = 0;

        struct SLaunchConfig {
            int blocks;
            int threads;
        };

        SLaunchConfig launch_configs[MAX_LAUNCH_CONFIGS];
        void set_launch_configs(int idx, int block_multiplier, int warps);
        void load_launch_configs(int idx, const char *env_var, int default_block_multiplier, int default_warps);
        int SM_COUNT;
    };
}

#endif
