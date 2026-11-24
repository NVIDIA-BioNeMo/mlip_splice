// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw `dlopen` bindings to the libcuda function table — the unsafe substrate
//! under the crate's safe layer. Reach for it (via `cuda_driver::ffi`) only for
//! operations the safe API doesn't cover.

use std::ffi::{c_char, c_void};
use std::os::raw::{c_int, c_uchar, c_uint, c_ulonglong, c_ushort};
use std::sync::OnceLock;

use dlbind::load_sym;
use libloading::Library;

pub type CUdevice = c_int;
pub type CUcontext = *mut c_void;
pub type CUdeviceptr = c_ulonglong;
pub type CUresult = c_uint;
pub type CUmodule = *mut c_void;
pub type CUfunction = *mut c_void;
pub type CUkernel = *mut c_void;
pub type CUlibrary = *mut c_void;
pub type CUstream = *mut c_void;
pub type CUevent = *mut c_void;
pub type CUgraph = *mut c_void;
pub type CUgraphNode = *mut c_void;
pub type CUgraphExec = *mut c_void;

pub const CUDA_SUCCESS: CUresult = 0;

pub const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
pub const CU_DEVICE_ATTRIBUTE_WARP_SIZE: c_int = 10;
pub const CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR: c_int = 106;
pub const CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: c_int = 39;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

pub const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
pub const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;
pub const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;

#[repr(C)]
#[allow(non_snake_case)]
pub struct CUDA_KERNEL_NODE_PARAMS {
    pub func: CUfunction,
    pub gridDimX: c_uint,
    pub gridDimY: c_uint,
    pub gridDimZ: c_uint,
    pub blockDimX: c_uint,
    pub blockDimY: c_uint,
    pub blockDimZ: c_uint,
    pub sharedMemBytes: c_uint,
    pub kernelParams: *mut *mut c_void,
    pub extra: *mut *mut c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}

#[repr(C)]
#[allow(non_snake_case)]
pub struct CUDA_MEMSET_NODE_PARAMS {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: c_uint,
    pub elementSize: c_uint,
    pub width: usize,
    pub height: usize,
}

pub type CUmemGenericAllocationHandle = c_ulonglong;

pub type CUmemAllocationType = c_uint;
pub const CU_MEM_ALLOCATION_TYPE_PINNED: CUmemAllocationType = 0x1;

pub type CUmemAllocationHandleType = c_uint;
pub const CU_MEM_HANDLE_TYPE_NONE: CUmemAllocationHandleType = 0x0;

pub type CUmemLocationType = c_uint;
pub const CU_MEM_LOCATION_TYPE_DEVICE: CUmemLocationType = 0x1;

#[allow(non_camel_case_types)]
pub type CUmemAccess_flags = c_uint;
pub const CU_MEM_ACCESS_FLAGS_PROT_READWRITE: CUmemAccess_flags = 0x3;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemLocation {
    pub type_: CUmemLocationType,
    pub id: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub struct CUmemAllocationPropAllocFlags {
    pub compressionType: c_uchar,
    pub gpuDirectRDMACapable: c_uchar,
    pub usage: c_ushort,
    pub reserved: [c_uchar; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_snake_case)]
pub struct CUmemAllocationProp {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut c_void,
    pub allocFlags: CUmemAllocationPropAllocFlags,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}

#[allow(non_snake_case)]
pub struct CudaDriver {
    _lib: Library,

    pub cuInit: unsafe extern "C" fn(c_uint) -> CUresult,
    pub cuDeviceGet: unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult,
    pub cuDeviceGetCount: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult,
    pub cuCtxCreate_v2: unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult,
    pub cuCtxSetCurrent: unsafe extern "C" fn(CUcontext) -> CUresult,
    pub cuCtxGetCurrent: unsafe extern "C" fn(*mut CUcontext) -> CUresult,
    pub cuCtxGetDevice: unsafe extern "C" fn(*mut c_int) -> CUresult,
    pub cuStreamGetCtx: unsafe extern "C" fn(CUstream, *mut CUcontext) -> CUresult,
    pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,

    pub cuLibraryLoadData: unsafe extern "C" fn(
        *mut CUlibrary,
        *const c_void,
        *const c_void,
        *const c_void,
        c_uint,
        *const c_void,
        *const c_void,
        c_uint,
    ) -> CUresult,
    pub cuLibraryUnload: unsafe extern "C" fn(CUlibrary) -> CUresult,
    pub cuLibraryGetKernel:
        unsafe extern "C" fn(*mut CUkernel, CUlibrary, *const c_char) -> CUresult,
    pub cuKernelGetFunction: unsafe extern "C" fn(*mut CUfunction, CUkernel) -> CUresult,
    pub cuKernelGetAttribute:
        unsafe extern "C" fn(*mut c_int, c_int, CUkernel, CUdevice) -> CUresult,

    pub cuModuleLoadData: unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult,
    pub cuModuleGetFunction:
        unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult,

    pub cuLaunchKernel: unsafe extern "C" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,

    pub cuEventCreate: unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult,
    pub cuEventRecord: unsafe extern "C" fn(CUevent, CUstream) -> CUresult,
    pub cuEventSynchronize: unsafe extern "C" fn(CUevent) -> CUresult,
    pub cuEventElapsedTime: unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult,
    pub cuEventDestroy: unsafe extern "C" fn(CUevent) -> CUresult,

    pub cuGraphCreate: unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult,
    pub cuGraphDestroy: unsafe extern "C" fn(CUgraph) -> CUresult,
    pub cuGraphAddKernelNode_v2: unsafe extern "C" fn(
        *mut CUgraphNode,
        CUgraph,
        *const CUgraphNode,
        usize,
        *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult,
    pub cuGraphAddMemsetNode: unsafe extern "C" fn(
        *mut CUgraphNode,
        CUgraph,
        *const CUgraphNode,
        usize,
        *const CUDA_MEMSET_NODE_PARAMS,
        CUcontext,
    ) -> CUresult,
    pub cuGraphInstantiateWithFlags:
        unsafe extern "C" fn(*mut CUgraphExec, CUgraph, c_ulonglong) -> CUresult,
    pub cuGraphLaunch: unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult,
    pub cuGraphExecDestroy: unsafe extern "C" fn(CUgraphExec) -> CUresult,
    pub cuGraphExecKernelNodeSetParams_v2:
        unsafe extern "C" fn(CUgraphExec, CUgraphNode, *const CUDA_KERNEL_NODE_PARAMS) -> CUresult,
    pub cuGraphExecMemsetNodeSetParams: unsafe extern "C" fn(
        CUgraphExec,
        CUgraphNode,
        *const CUDA_MEMSET_NODE_PARAMS,
        CUcontext,
    ) -> CUresult,
    pub cuStreamCreate: unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult,
    pub cuStreamDestroy_v2: unsafe extern "C" fn(CUstream) -> CUresult,
    pub cuStreamIsCapturing: unsafe extern "C" fn(CUstream, *mut c_uint) -> CUresult,

    pub cuFuncGetName: unsafe extern "C" fn(*mut *const c_char, CUfunction) -> CUresult,
    pub cuFuncGetModule: unsafe extern "C" fn(*mut CUmodule, CUfunction) -> CUresult,
    pub cuKernelGetLibrary: unsafe extern "C" fn(*mut CUlibrary, CUkernel) -> CUresult,

    pub cuMemGetAddressRange_v2:
        unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUdeviceptr) -> CUresult,

    pub cuMemAlloc_v2: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub cuMemFree_v2: unsafe extern "C" fn(CUdeviceptr) -> CUresult,

    pub cuMemGetAllocationGranularity:
        unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, c_uint) -> CUresult,
    pub cuMemAddressReserve:
        unsafe extern "C" fn(*mut CUdeviceptr, usize, usize, CUdeviceptr, c_ulonglong) -> CUresult,
    pub cuMemCreate: unsafe extern "C" fn(
        *mut CUmemGenericAllocationHandle,
        usize,
        *const CUmemAllocationProp,
        c_ulonglong,
    ) -> CUresult,
    pub cuMemMap: unsafe extern "C" fn(
        CUdeviceptr,
        usize,
        usize,
        CUmemGenericAllocationHandle,
        c_ulonglong,
    ) -> CUresult,
    pub cuMemSetAccess:
        unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult,
    pub cuMemUnmap: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,
    pub cuMemRelease: unsafe extern "C" fn(CUmemGenericAllocationHandle) -> CUresult,
    pub cuMemAddressFree: unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult,

    pub cuFuncSetAttribute: unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult,
    pub cuDeviceGetAttribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult,
    pub cuDriverGetVersion: unsafe extern "C" fn(*mut c_int) -> CUresult,

    pub cuMemsetD8Async: unsafe extern "C" fn(CUdeviceptr, c_uchar, usize, CUstream) -> CUresult,

    pub cuMemcpyHtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoHAsync_v2:
        unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuMemcpyDtoDAsync_v2:
        unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult,
    pub cuStreamSynchronize: unsafe extern "C" fn(CUstream) -> CUresult,

    pub cuMemcpyHtoD_v2: unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    pub cuMemcpyDtoH_v2: unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    pub cuMemcpyDtoD_v2: unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,

    pub cuGetErrorString: unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult,
}

unsafe impl Send for CudaDriver {}
unsafe impl Sync for CudaDriver {}

impl CudaDriver {
    #[allow(non_snake_case)]
    fn load() -> Option<Self> {
        let lib = dlfind::open_named("libcuda", &["1"])?;

        let cuInit = load_sym!(&lib, "cuInit", unsafe extern "C" fn(c_uint) -> CUresult);
        let cuDeviceGet = load_sym!(
            &lib,
            "cuDeviceGet",
            unsafe extern "C" fn(*mut CUdevice, c_int) -> CUresult
        );
        let cuDeviceGetCount = load_sym!(
            &lib,
            "cuDeviceGetCount",
            unsafe extern "C" fn(*mut c_int) -> CUresult
        );
        let cuDevicePrimaryCtxRetain = load_sym!(
            &lib,
            "cuDevicePrimaryCtxRetain",
            unsafe extern "C" fn(*mut CUcontext, CUdevice) -> CUresult
        );
        let cuCtxCreate_v2 = load_sym!(
            &lib,
            "cuCtxCreate_v2",
            unsafe extern "C" fn(*mut CUcontext, c_uint, CUdevice) -> CUresult
        );
        let cuCtxSetCurrent = load_sym!(
            &lib,
            "cuCtxSetCurrent",
            unsafe extern "C" fn(CUcontext) -> CUresult
        );
        let cuCtxGetCurrent = load_sym!(
            &lib,
            "cuCtxGetCurrent",
            unsafe extern "C" fn(*mut CUcontext) -> CUresult
        );
        let cuCtxGetDevice = load_sym!(
            &lib,
            "cuCtxGetDevice",
            unsafe extern "C" fn(*mut c_int) -> CUresult
        );
        let cuStreamGetCtx = load_sym!(
            &lib,
            "cuStreamGetCtx",
            unsafe extern "C" fn(CUstream, *mut CUcontext) -> CUresult
        );
        let cuCtxSynchronize =
            load_sym!(&lib, "cuCtxSynchronize", unsafe extern "C" fn() -> CUresult);

        let cuLibraryLoadData = load_sym!(
            &lib,
            "cuLibraryLoadData",
            unsafe extern "C" fn(
                *mut CUlibrary,
                *const c_void,
                *const c_void,
                *const c_void,
                c_uint,
                *const c_void,
                *const c_void,
                c_uint,
            ) -> CUresult
        );
        let cuLibraryUnload = load_sym!(
            &lib,
            "cuLibraryUnload",
            unsafe extern "C" fn(CUlibrary) -> CUresult
        );
        let cuLibraryGetKernel = load_sym!(
            &lib,
            "cuLibraryGetKernel",
            unsafe extern "C" fn(*mut CUkernel, CUlibrary, *const c_char) -> CUresult
        );
        let cuKernelGetFunction = load_sym!(
            &lib,
            "cuKernelGetFunction",
            unsafe extern "C" fn(*mut CUfunction, CUkernel) -> CUresult
        );
        let cuKernelGetAttribute = load_sym!(
            &lib,
            "cuKernelGetAttribute",
            unsafe extern "C" fn(*mut c_int, c_int, CUkernel, CUdevice) -> CUresult
        );
        let cuModuleLoadData = load_sym!(
            &lib,
            "cuModuleLoadData",
            unsafe extern "C" fn(*mut CUmodule, *const c_void) -> CUresult
        );
        let cuModuleGetFunction = load_sym!(
            &lib,
            "cuModuleGetFunction",
            unsafe extern "C" fn(*mut CUfunction, CUmodule, *const c_char) -> CUresult
        );

        let cuLaunchKernel = load_sym!(
            &lib,
            "cuLaunchKernel",
            unsafe extern "C" fn(
                CUfunction,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                c_uint,
                CUstream,
                *mut *mut c_void,
                *mut *mut c_void,
            ) -> CUresult
        );

        let cuEventCreate = load_sym!(
            &lib,
            "cuEventCreate",
            unsafe extern "C" fn(*mut CUevent, c_uint) -> CUresult
        );
        let cuEventRecord = load_sym!(
            &lib,
            "cuEventRecord",
            unsafe extern "C" fn(CUevent, CUstream) -> CUresult
        );
        let cuEventSynchronize = load_sym!(
            &lib,
            "cuEventSynchronize",
            unsafe extern "C" fn(CUevent) -> CUresult
        );
        let cuEventElapsedTime = load_sym!(
            &lib,
            "cuEventElapsedTime",
            unsafe extern "C" fn(*mut f32, CUevent, CUevent) -> CUresult
        );
        let cuEventDestroy = load_sym!(
            &lib,
            "cuEventDestroy",
            unsafe extern "C" fn(CUevent) -> CUresult
        );

        let cuGraphCreate = load_sym!(
            &lib,
            "cuGraphCreate",
            unsafe extern "C" fn(*mut CUgraph, c_uint) -> CUresult
        );
        let cuGraphDestroy = load_sym!(
            &lib,
            "cuGraphDestroy",
            unsafe extern "C" fn(CUgraph) -> CUresult
        );
        let cuGraphAddKernelNode_v2 = load_sym!(
            &lib,
            "cuGraphAddKernelNode_v2",
            unsafe extern "C" fn(
                *mut CUgraphNode,
                CUgraph,
                *const CUgraphNode,
                usize,
                *const CUDA_KERNEL_NODE_PARAMS,
            ) -> CUresult
        );
        let cuGraphAddMemsetNode = load_sym!(
            &lib,
            "cuGraphAddMemsetNode",
            unsafe extern "C" fn(
                *mut CUgraphNode,
                CUgraph,
                *const CUgraphNode,
                usize,
                *const CUDA_MEMSET_NODE_PARAMS,
                CUcontext,
            ) -> CUresult
        );

        // `cuGraphInstantiate` in the ABI is the legacy 5-arg form; cuda.h
        // redirects the name to `cuGraphInstantiateWithFlags` via a macro that
        // dlsym bypasses. Binding "cuGraphInstantiate" as 3-arg calls the 5-arg
        // function with two garbage trailing args, causing driver writes to a
        // garbage log-buffer pointer and segfaults on some GPUs.
        let cuGraphInstantiateWithFlags = load_sym!(
            &lib,
            "cuGraphInstantiateWithFlags",
            unsafe extern "C" fn(*mut CUgraphExec, CUgraph, c_ulonglong) -> CUresult
        );
        let cuGraphLaunch = load_sym!(
            &lib,
            "cuGraphLaunch",
            unsafe extern "C" fn(CUgraphExec, CUstream) -> CUresult
        );
        let cuGraphExecDestroy = load_sym!(
            &lib,
            "cuGraphExecDestroy",
            unsafe extern "C" fn(CUgraphExec) -> CUresult
        );
        let cuGraphExecKernelNodeSetParams_v2 = load_sym!(
            &lib,
            "cuGraphExecKernelNodeSetParams_v2",
            unsafe extern "C" fn(
                CUgraphExec,
                CUgraphNode,
                *const CUDA_KERNEL_NODE_PARAMS,
            ) -> CUresult
        );
        let cuGraphExecMemsetNodeSetParams = load_sym!(
            &lib,
            "cuGraphExecMemsetNodeSetParams",
            unsafe extern "C" fn(
                CUgraphExec,
                CUgraphNode,
                *const CUDA_MEMSET_NODE_PARAMS,
                CUcontext,
            ) -> CUresult
        );

        let cuStreamCreate = load_sym!(
            &lib,
            "cuStreamCreate",
            unsafe extern "C" fn(*mut CUstream, c_uint) -> CUresult
        );
        let cuStreamDestroy_v2 = load_sym!(
            &lib,
            "cuStreamDestroy_v2",
            unsafe extern "C" fn(CUstream) -> CUresult
        );
        let cuStreamIsCapturing = load_sym!(
            &lib,
            "cuStreamIsCapturing",
            unsafe extern "C" fn(CUstream, *mut c_uint) -> CUresult
        );
        let cuFuncGetName = load_sym!(
            &lib,
            "cuFuncGetName",
            unsafe extern "C" fn(*mut *const c_char, CUfunction) -> CUresult
        );
        let cuFuncGetModule = load_sym!(
            &lib,
            "cuFuncGetModule",
            unsafe extern "C" fn(*mut CUmodule, CUfunction) -> CUresult
        );
        let cuKernelGetLibrary = load_sym!(
            &lib,
            "cuKernelGetLibrary",
            unsafe extern "C" fn(*mut CUlibrary, CUkernel) -> CUresult
        );
        let cuMemGetAddressRange_v2 = load_sym!(
            &lib,
            "cuMemGetAddressRange_v2",
            unsafe extern "C" fn(*mut CUdeviceptr, *mut usize, CUdeviceptr) -> CUresult
        );

        let cuMemAlloc_v2 = load_sym!(
            &lib,
            "cuMemAlloc_v2",
            unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult
        );
        let cuMemFree_v2 = load_sym!(
            &lib,
            "cuMemFree_v2",
            unsafe extern "C" fn(CUdeviceptr) -> CUresult
        );
        let cuMemGetAllocationGranularity = load_sym!(
            &lib,
            "cuMemGetAllocationGranularity",
            unsafe extern "C" fn(*mut usize, *const CUmemAllocationProp, c_uint) -> CUresult
        );
        let cuMemAddressReserve = load_sym!(
            &lib,
            "cuMemAddressReserve",
            unsafe extern "C" fn(
                *mut CUdeviceptr,
                usize,
                usize,
                CUdeviceptr,
                c_ulonglong,
            ) -> CUresult
        );
        let cuMemCreate = load_sym!(
            &lib,
            "cuMemCreate",
            unsafe extern "C" fn(
                *mut CUmemGenericAllocationHandle,
                usize,
                *const CUmemAllocationProp,
                c_ulonglong,
            ) -> CUresult
        );
        let cuMemMap = load_sym!(
            &lib,
            "cuMemMap",
            unsafe extern "C" fn(
                CUdeviceptr,
                usize,
                usize,
                CUmemGenericAllocationHandle,
                c_ulonglong,
            ) -> CUresult
        );
        let cuMemSetAccess = load_sym!(
            &lib,
            "cuMemSetAccess",
            unsafe extern "C" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult
        );
        let cuMemUnmap = load_sym!(
            &lib,
            "cuMemUnmap",
            unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult
        );
        let cuMemRelease = load_sym!(
            &lib,
            "cuMemRelease",
            unsafe extern "C" fn(CUmemGenericAllocationHandle) -> CUresult
        );
        let cuMemAddressFree = load_sym!(
            &lib,
            "cuMemAddressFree",
            unsafe extern "C" fn(CUdeviceptr, usize) -> CUresult
        );

        let cuFuncSetAttribute = load_sym!(
            &lib,
            "cuFuncSetAttribute",
            unsafe extern "C" fn(CUfunction, c_int, c_int) -> CUresult
        );

        let cuDeviceGetAttribute = load_sym!(
            &lib,
            "cuDeviceGetAttribute",
            unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CUresult
        );
        let cuDriverGetVersion = load_sym!(
            &lib,
            "cuDriverGetVersion",
            unsafe extern "C" fn(*mut c_int) -> CUresult
        );

        let cuMemsetD8Async = load_sym!(
            &lib,
            "cuMemsetD8Async",
            unsafe extern "C" fn(CUdeviceptr, c_uchar, usize, CUstream) -> CUresult
        );

        let cuMemcpyHtoDAsync_v2 = load_sym!(
            &lib,
            "cuMemcpyHtoDAsync_v2",
            unsafe extern "C" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult
        );
        let cuMemcpyDtoHAsync_v2 = load_sym!(
            &lib,
            "cuMemcpyDtoHAsync_v2",
            unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult
        );
        let cuMemcpyDtoDAsync_v2 = load_sym!(
            &lib,
            "cuMemcpyDtoDAsync_v2",
            unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CUresult
        );
        let cuStreamSynchronize = load_sym!(
            &lib,
            "cuStreamSynchronize",
            unsafe extern "C" fn(CUstream) -> CUresult
        );

        let cuMemcpyHtoD_v2 = load_sym!(
            &lib,
            "cuMemcpyHtoD_v2",
            unsafe extern "C" fn(CUdeviceptr, *const c_void, usize) -> CUresult
        );
        let cuMemcpyDtoH_v2 = load_sym!(
            &lib,
            "cuMemcpyDtoH_v2",
            unsafe extern "C" fn(*mut c_void, CUdeviceptr, usize) -> CUresult
        );
        let cuMemcpyDtoD_v2 = load_sym!(
            &lib,
            "cuMemcpyDtoD_v2",
            unsafe extern "C" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult
        );

        let cuGetErrorString = load_sym!(
            &lib,
            "cuGetErrorString",
            unsafe extern "C" fn(CUresult, *mut *const c_char) -> CUresult
        );

        Some(Self {
            _lib: lib,
            cuInit,
            cuDeviceGet,
            cuDeviceGetCount,
            cuDevicePrimaryCtxRetain,
            cuCtxCreate_v2,
            cuCtxSetCurrent,
            cuCtxGetCurrent,
            cuCtxGetDevice,
            cuStreamGetCtx,
            cuCtxSynchronize,
            cuLibraryLoadData,
            cuLibraryUnload,
            cuLibraryGetKernel,
            cuKernelGetFunction,
            cuKernelGetAttribute,
            cuModuleLoadData,
            cuModuleGetFunction,
            cuLaunchKernel,
            cuEventCreate,
            cuEventRecord,
            cuEventSynchronize,
            cuEventElapsedTime,
            cuEventDestroy,
            cuGraphCreate,
            cuGraphDestroy,
            cuGraphAddKernelNode_v2,
            cuGraphAddMemsetNode,
            cuGraphInstantiateWithFlags,
            cuGraphLaunch,
            cuGraphExecDestroy,
            cuGraphExecKernelNodeSetParams_v2,
            cuGraphExecMemsetNodeSetParams,
            cuStreamCreate,
            cuStreamDestroy_v2,
            cuStreamIsCapturing,
            cuFuncGetName,
            cuFuncGetModule,
            cuKernelGetLibrary,
            cuMemGetAddressRange_v2,
            cuMemAlloc_v2,
            cuMemFree_v2,
            cuMemGetAllocationGranularity,
            cuMemAddressReserve,
            cuMemCreate,
            cuMemMap,
            cuMemSetAccess,
            cuMemUnmap,
            cuMemRelease,
            cuMemAddressFree,
            cuFuncSetAttribute,
            cuDeviceGetAttribute,
            cuDriverGetVersion,
            cuMemsetD8Async,
            cuMemcpyHtoDAsync_v2,
            cuMemcpyDtoHAsync_v2,
            cuMemcpyDtoDAsync_v2,
            cuStreamSynchronize,
            cuMemcpyHtoD_v2,
            cuMemcpyDtoH_v2,
            cuMemcpyDtoD_v2,
            cuGetErrorString,
        })
    }

    pub fn get() -> &'static Self {
        static INST: OnceLock<CudaDriver> = OnceLock::new();
        INST.get_or_init(|| {
            CudaDriver::load().expect("libcuda not found. Tried libcuda.so, libcuda.so.1.")
        })
    }
}
