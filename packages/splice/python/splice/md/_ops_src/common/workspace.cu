// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "workspace.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace splice_md;

workspace::workspace() {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    SM_COUNT = deviceProp.multiProcessorCount;
}

void workspace::set_launch_configs(int idx, int block_multiplier, int warps) {
    launch_configs[idx].blocks = SM_COUNT * block_multiplier;
    launch_configs[idx].threads = WARP_SIZE * warps;
}

void workspace::load_launch_configs(int idx, const char *env_var, int default_block_multiplier, int default_warps) {
    char ekey[256];
    snprintf(ekey, sizeof(ekey), "%s_%s", ENV_VAR_PREFIX, env_var);

    char *value = std::getenv(ekey);
    int block_multiplier = 0, warps = 0;

    if (value) {
        // strtok mutates its argument, and the environment block is not ours to edit.
        char scratch[64];
        snprintf(scratch, sizeof(scratch), "%s", value);
        char *tok = strtok(scratch, "x");
        if (tok) block_multiplier = atoi(tok);
        tok = strtok(nullptr, "x");
        if (tok) warps = atoi(tok);
        printf("[splice_md.md] launch config %s: %dx%d\n", env_var, block_multiplier, warps);
    }

    if (block_multiplier == 0 || warps == 0) {
        block_multiplier = default_block_multiplier;
        warps = default_warps;
    }

    set_launch_configs(idx, block_multiplier, warps);
}
