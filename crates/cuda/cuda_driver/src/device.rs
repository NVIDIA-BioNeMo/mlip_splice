// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::os::raw::c_int;

use crate::context::init;
use crate::{
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE, CUdevice, CudaDriver, Result, check,
};

/// A CUDA device ordinal. Cheap, `Copy`, multi-GPU-ready: every query targets
/// this specific device rather than implicitly the current one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Device(pub(crate) CUdevice);

impl Device {
    /// The device at `ordinal` (0-based). Initialises the driver if needed.
    pub fn get(ordinal: i32) -> Result<Device> {
        init()?;
        let drv = CudaDriver::get();
        let mut dev: CUdevice = 0;
        check(
            unsafe { (drv.cuDeviceGet)(&mut dev, ordinal) },
            "cuDeviceGet",
        )?;
        Ok(Device(dev))
    }

    /// Number of CUDA devices visible to the driver.
    pub fn count() -> Result<i32> {
        init()?;
        let drv = CudaDriver::get();
        let mut n: c_int = 0;
        check(
            unsafe { (drv.cuDeviceGetCount)(&mut n) },
            "cuDeviceGetCount",
        )?;
        Ok(n)
    }

    /// The device backing the current context.
    pub fn current() -> Device {
        let drv = CudaDriver::get();
        let mut dev: CUdevice = 0;
        unsafe { (drv.cuCtxGetDevice)(&mut dev) };
        Device(dev)
    }

    /// Wrap a raw device ordinal obtained elsewhere.
    pub fn from_raw(raw: CUdevice) -> Device {
        Device(raw)
    }

    pub fn raw(&self) -> CUdevice {
        self.0
    }

    pub fn attribute(&self, attr: c_int) -> Result<i32> {
        let drv = CudaDriver::get();
        let mut v: c_int = 0;
        check(
            unsafe { (drv.cuDeviceGetAttribute)(&mut v, attr, self.0) },
            "cuDeviceGetAttribute",
        )?;
        Ok(v)
    }

    pub fn compute_capability(&self) -> Result<(i32, i32)> {
        Ok((
            self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
            self.attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?,
        ))
    }

    /// Compute capability as `10 * major + minor` (e.g. 90 for sm_90).
    pub fn sm_arch(&self) -> Result<i32> {
        let (major, minor) = self.compute_capability()?;
        Ok(10 * major + minor)
    }

    pub fn sm_count(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    }

    pub fn max_threads_per_sm(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    }

    pub fn warp_size(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE)
    }

    pub fn max_blocks_per_sm(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR)
    }

    pub fn max_smem_optin(&self) -> Result<i32> {
        self.attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
    }
}
