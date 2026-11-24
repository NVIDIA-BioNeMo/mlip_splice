// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::ffi::c_void;
use std::marker::PhantomData;

use crate::context::with_current_context;
use crate::{CUcontext, CUdeviceptr, CudaDriver, Result, check};

/// An owned device allocation of `len` elements of `T`, freed on drop. Carries
/// the context it was allocated in, so it frees correctly even when dropped on
/// a thread with a different current context (and is multi-GPU-safe).
pub struct DeviceBuffer<T: Copy> {
    ptr: CUdeviceptr,
    len: usize,
    ctx: CUcontext,
    _marker: PhantomData<T>,
}

unsafe impl<T: Copy> Send for DeviceBuffer<T> {}
// `&DeviceBuffer` only reads (device→host copies, pointer/len accessors), so
// sharing it across threads is sound.
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Allocate `len` elements in `ctx`. The caller chooses the context
    /// explicitly (pass `Context::current().raw()` for the current one).
    pub fn alloc(ctx: CUcontext, len: usize) -> Result<Self> {
        let mut ptr: CUdeviceptr = 0;
        if len != 0 {
            let drv = CudaDriver::get();
            with_current_context(ctx, || {
                check(
                    unsafe { (drv.cuMemAlloc_v2)(&mut ptr, len * std::mem::size_of::<T>()) },
                    "cuMemAlloc_v2",
                )
            })?;
        }
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Allocate in `ctx` and copy `host` up (synchronously).
    pub fn from_host(ctx: CUcontext, host: &[T]) -> Result<Self> {
        let buf = Self::alloc(ctx, host.len())?;
        if !host.is_empty() {
            let drv = CudaDriver::get();
            with_current_context(ctx, || {
                check(
                    unsafe {
                        (drv.cuMemcpyHtoD_v2)(
                            buf.ptr,
                            host.as_ptr() as *const c_void,
                            std::mem::size_of_val(host),
                        )
                    },
                    "cuMemcpyHtoD_v2",
                )
            })?;
        }
        Ok(buf)
    }

    pub fn to_host(&self) -> Result<Vec<T>> {
        let mut host: Vec<T> = Vec::with_capacity(self.len);
        if self.len != 0 {
            let drv = CudaDriver::get();
            with_current_context(self.ctx, || {
                unsafe {
                    check(
                        (drv.cuMemcpyDtoH_v2)(
                            host.as_mut_ptr() as *mut c_void,
                            self.ptr,
                            self.len * std::mem::size_of::<T>(),
                        ),
                        "cuMemcpyDtoH_v2",
                    )?;
                    host.set_len(self.len);
                }
                Ok(())
            })?;
        }
        Ok(host)
    }

    pub fn ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    pub fn context(&self) -> CUcontext {
        self.ctx
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn nbytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            let drv = CudaDriver::get();
            let _ = with_current_context(self.ctx, || {
                check(unsafe { (drv.cuMemFree_v2)(self.ptr) }, "cuMemFree_v2")
            });
        }
    }
}

/// A half-open device address range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: CUdeviceptr,
    pub end: CUdeviceptr,
}

impl Span {
    /// `None` when the range runs past the end of the address space: an
    /// unchecked end would wrap to zero and make an impossible buffer look
    /// disjoint from everything.
    pub fn of(start: CUdeviceptr, nbytes: usize) -> Option<Span> {
        let end = u64::try_from(nbytes)
            .ok()
            .and_then(|length| start.checked_add(length))?;
        Some(Span { start, end })
    }

    /// Half-open, and empty on either side means no overlap: a zero-byte buffer
    /// occupies no address, so it can neither be clobbered nor clobber. An empty
    /// extent is a real extent — an empty array — and not a stand-in for an
    /// extent nobody stated.
    pub fn overlaps(self, other: Span) -> bool {
        self.start < self.end
            && other.start < other.end
            && self.start < other.end
            && other.start < self.end
    }
}

#[cfg(test)]
mod tests {
    use super::Span;

    #[test]
    fn a_range_past_the_end_of_the_address_space_is_not_a_span() {
        assert_eq!(
            Span::of(u64::MAX - 4, 4),
            Some(Span {
                start: u64::MAX - 4,
                end: u64::MAX,
            })
        );
        assert_eq!(Span::of(u64::MAX - 3, 4), None);
        assert_eq!(Span::of(u64::MAX, 1), None);
    }

    #[test]
    fn abutting_ranges_do_not_overlap_and_straddling_ones_do() {
        let left = Span::of(4096, 256).unwrap();
        assert!(!left.overlaps(Span::of(4096 + 256, 256).unwrap()));
        assert!(left.overlaps(Span::of(4096 + 255, 256).unwrap()));
        assert!(left.overlaps(left));
    }

    #[test]
    fn an_empty_range_overlaps_nothing_including_itself() {
        let empty = Span::of(4096, 0).unwrap();
        assert!(!empty.overlaps(empty));
        assert!(!empty.overlaps(Span::of(4096, 256).unwrap()));
        assert!(!Span::of(4096, 256).unwrap().overlaps(empty));
    }
}
