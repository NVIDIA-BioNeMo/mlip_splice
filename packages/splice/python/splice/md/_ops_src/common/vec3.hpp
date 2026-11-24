// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// float3 helpers shared by the MD kernels.

#pragma once
#include <cuda_runtime.h>

namespace v3 {

    __forceinline__ __device__ float3 zero() {
        return {0.0f, 0.0f, 0.0f};
    }

    __forceinline__ __device__ float3 one() {
        return {1.0f, 1.0f, 1.0f};
    }

    __forceinline__ __device__ float3 of(float v) {
        return {v, v, v};
    }

    __forceinline__ __device__ float3 of(int v) {
        return {(float)v, (float)v, (float)v};
    }

    __forceinline__ __device__ float length(float3 v) {
        return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    __forceinline__ __device__ void scale(float3 &v, float s) {
        v.x *= s;
        v.y *= s;
        v.z *= s;
    }

    __forceinline__ __device__ float3 scale_by(float3 v, float s) {
        return {v.x * s, v.y * s, v.z * s};
    }
    
    __forceinline__ __device__ void normalize(float3 &v) {
        scale(v, 1.0f / length(v));
    }

    __forceinline__ __device__ float3 abs(float3 v) {
        return {fabs(v.x), fabs(v.y), fabs(v.z)};
    }

    __forceinline__ __device__ float3 smaller(float3 a, float3 b) {
        return {min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)};
    }
    
    __forceinline__ __device__ float3 add(float3 a, float3 b) {
        return {a.x + b.x, a.y + b.y, a.z + b.z};
    }
    
    __forceinline__ __device__ float3 sub(float3 a, float3 b) {
        return {a.x - b.x, a.y - b.y, a.z - b.z};
    }
    
    __forceinline__ __device__ float3 by(float3 a, float3 b) {
        return {a.x * b.x, a.y * b.y, a.z * b.z};
    }

    __forceinline__ __device__ float dot(float3 a, float3 b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    __forceinline__ __device__ float3 inverse(float3 v) {
        return {1.0f / v.x, 1.0f / v.y, 1.0f / v.z};
    }

    __forceinline__ __device__ float3 neg(float3 v) {
        return {-v.x, -v.y, -v.z};
    }

    __forceinline__ __device__ float3 diagonal(float3 a, float3 b, float3 c) {
        return {a.x, b.y, c.z};
    }

    __forceinline__ __device__ void sadd(float3 &a, float3 b) {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
    }

    __forceinline__ __device__ void ssub(float3 &a, float3 b) {
        a.x -= b.x;
        a.y -= b.y;
        a.z -= b.z;
    }

    __forceinline__ __device__ void scale_sadd(float3 &a, float3 b, float c) {
        a.x += b.x * c;
        a.y += b.y * c;
        a.z += b.z * c;
    }

    __forceinline__ __device__ void pbc_wrap(float3 &v, float3 pbc, float3 invpbc) {
        v.x -= pbc.x * nearbyint(v.x * invpbc.x);
        v.y -= pbc.y * nearbyint(v.y * invpbc.y);
        v.z -= pbc.z * nearbyint(v.z * invpbc.z);
    }
    
    __forceinline__ __device__ float3 cross(float3 a, float3 b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }
    
    __forceinline__ __device__ float3 mxv(float3 m1, float3 m2, float3 m3, float3 v) {
        return {
            m1.x * v.x + m1.y * v.y + m1.z * v.z,
            m2.x * v.x + m2.y * v.y + m2.z * v.z,
            m3.x * v.x + m3.y * v.y + m3.z * v.z
        };
    }

    __forceinline__ __device__ float3 mtxv(float3 m1, float3 m2, float3 m3, float3 v) {
        return {
            m1.x * v.x + m2.x * v.y + m3.x * v.z,
            m1.y * v.x + m2.y * v.y + m3.y * v.z,
            m1.z * v.x + m2.z * v.y + m3.z * v.z
        };
    }
    
    __forceinline__ __device__ float3 from_array(float *a) {
        return {a[0], a[1], a[2]};
    }
    
    __forceinline__ __device__ void to_array(float3 v, float *a) {
        a[0] = v.x;
        a[1] = v.y;
        a[2] = v.z;
    }

    __forceinline__ __device__ void to_array3(float3 v1, float3 v2, float3 v3, float *a) {
        to_array(v1, a);
        to_array(v2, a + 3);
        to_array(v3, a + 6);
    }
    
    __forceinline__ __device__ void pbc_unwrap(float3 &r, float3 pbc) {
        if (r.x < 0.0f) r.x += pbc.x;
        if (r.y < 0.0f) r.y += pbc.y;
        if (r.z < 0.0f) r.z += pbc.z;
        if (r.x >= pbc.x) r.x -= pbc.x;
        if (r.y >= pbc.y) r.y -= pbc.y;
        if (r.z >= pbc.z) r.z -= pbc.z;
    }
    
    __forceinline__ __device__ void cuda_v2r(float3 a, float3 b, float3 &r1, float3 &r2, float3 &r3)
    {
        r1 = add(a, b);
        normalize(r1);
        
        r3 = cross(r1, b);
        normalize(r3);
    
        r2 = cross(r3, r1);
    }
}