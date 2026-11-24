// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Reading one call frame: its stream, its buffers and its attributes.
//!
//! Everything here borrows from the frame XLA handed in and validates what it
//! reads, because a call frame states its own shape and a handler that trusts
//! that statement without checking it reads whatever follows.

use std::ffi::c_void;

use xla_ffi_sys::{
    XLA_FFI_ArgType_BUFFER, XLA_FFI_Array, XLA_FFI_AttrType_ARRAY, XLA_FFI_AttrType_SCALAR,
    XLA_FFI_AttrType_STRING, XLA_FFI_Buffer, XLA_FFI_ByteSpan, XLA_FFI_CallFrame,
    XLA_FFI_RetType_BUFFER, XLA_FFI_Scalar, XLA_FFI_Stream_Get_Args,
    XLA_FFI_Stream_Get_Args_STRUCT_SIZE,
};

use crate::buffer::{AnyBuffer, Buffer};
use crate::dtype::Dtype;
use crate::error::{Error, Result};

#[derive(Copy, Clone)]
pub struct Stream(pub *mut c_void);

unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

pub struct DecodeState<'cf> {
    cf: &'cf XLA_FFI_CallFrame,
    arg_cursor: usize,
    ret_cursor: usize,
}

impl<'cf> DecodeState<'cf> {
    pub fn new(cf: &'cf XLA_FFI_CallFrame) -> Self {
        Self {
            cf,
            arg_cursor: 0,
            ret_cursor: 0,
        }
    }

    pub fn call_frame(&self) -> &'cf XLA_FFI_CallFrame {
        self.cf
    }

    pub fn stream(&self) -> Result<Stream> {
        let mut args = XLA_FFI_Stream_Get_Args {
            struct_size: XLA_FFI_Stream_Get_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            ctx: self.cf.ctx,
            stream: std::ptr::null_mut(),
        };
        let getter = unsafe { (*self.cf.api).XLA_FFI_Stream_Get }
            .ok_or_else(|| Error::internal("XLA_FFI_Stream_Get is null"))?;
        let err = unsafe { getter(&mut args) };
        if err.is_null() {
            Ok(Stream(args.stream))
        } else {
            Err(Error::internal("XLA_FFI_Stream_Get failed"))
        }
    }

    fn n_args(&self) -> usize {
        self.cf.args.size as usize
    }

    fn n_rets(&self) -> usize {
        self.cf.rets.size as usize
    }

    fn raw_arg(&self, i: usize) -> Result<&'cf XLA_FFI_Buffer> {
        if i >= self.n_args() {
            return Err(Error::invalid_argument(format!(
                "expected at least {} arg(s), got {}",
                i + 1,
                self.n_args()
            )));
        }
        let ty = unsafe { *self.cf.args.types.add(i) };
        if ty != XLA_FFI_ArgType_BUFFER {
            return Err(Error::invalid_argument(format!(
                "arg {i}: expected BUFFER, got type tag {ty}"
            )));
        }
        Ok(unsafe { &*(*self.cf.args.args.add(i) as *const XLA_FFI_Buffer) })
    }

    fn raw_ret(&self, i: usize) -> Result<&'cf XLA_FFI_Buffer> {
        if i >= self.n_rets() {
            return Err(Error::invalid_argument(format!(
                "expected at least {} ret(s), got {}",
                i + 1,
                self.n_rets()
            )));
        }
        let ty = unsafe { *self.cf.rets.types.add(i) };
        if ty != XLA_FFI_RetType_BUFFER {
            return Err(Error::invalid_argument(format!(
                "ret {i}: expected BUFFER, got type tag {ty}"
            )));
        }
        Ok(unsafe { &*(*self.cf.rets.rets.add(i) as *const XLA_FFI_Buffer) })
    }

    pub fn next_arg<T: Dtype>(&mut self) -> Result<Buffer<'cf, T>> {
        let raw = self.raw_arg(self.arg_cursor)?;
        self.arg_cursor += 1;
        Buffer::from_raw(raw)
    }

    pub fn next_arg_any(&mut self) -> Result<AnyBuffer<'cf>> {
        let raw = self.raw_arg(self.arg_cursor)?;
        self.arg_cursor += 1;
        Ok(AnyBuffer::from_raw(raw))
    }

    pub fn next_ret<T: Dtype>(&mut self) -> Result<Buffer<'cf, T>> {
        let raw = self.raw_ret(self.ret_cursor)?;
        self.ret_cursor += 1;
        Buffer::from_raw(raw)
    }

    pub fn next_ret_any(&mut self) -> Result<AnyBuffer<'cf>> {
        let raw = self.raw_ret(self.ret_cursor)?;
        self.ret_cursor += 1;
        Ok(AnyBuffer::from_raw(raw))
    }

    pub fn remaining_args(&mut self) -> crate::RemainingArgs<'cf> {
        let start = self.arg_cursor;
        self.arg_cursor = self.n_args();
        crate::RemainingArgs::new(self.cf, start, self.n_args())
    }

    pub fn remaining_rets(&mut self) -> crate::RemainingRets<'cf> {
        let start = self.ret_cursor;
        self.ret_cursor = self.n_rets();
        crate::RemainingRets::new(self.cf, start, self.n_rets())
    }

    fn find_attr(&self, name: &str) -> Result<usize> {
        let n = self.cf.attrs.size as usize;
        for i in 0..n {
            let span = unsafe { *self.cf.attrs.names.add(i) };
            let s = unsafe { std::slice::from_raw_parts((*span).ptr as *const u8, (*span).len) };
            if s == name.as_bytes() {
                return Ok(i);
            }
        }
        Err(Error::invalid_argument(format!(
            "missing required attr `{name}`"
        )))
    }

    pub fn attr_array<T: Dtype>(&self, name: &str) -> Result<&'cf [T]> {
        let i = self.find_attr(name)?;
        let ty = unsafe { *self.cf.attrs.types.add(i) };
        if ty != XLA_FFI_AttrType_ARRAY {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected ARRAY, got type tag {ty}"
            )));
        }
        let arr = unsafe { &*(*self.cf.attrs.attrs.add(i) as *const XLA_FFI_Array) };
        if arr.dtype != T::TAG {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected array of {} (tag {}), got tag {}",
                T::NAME,
                T::TAG,
                arr.dtype
            )));
        }
        if arr.size == 0 {
            return Ok(&[]);
        }
        Ok(unsafe { std::slice::from_raw_parts(arr.data as *const T, arr.size) })
    }

    pub fn attr_scalar<T: Dtype>(&self, name: &str) -> Result<T> {
        let i = self.find_attr(name)?;
        let ty = unsafe { *self.cf.attrs.types.add(i) };
        if ty != XLA_FFI_AttrType_SCALAR {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected SCALAR, got type tag {ty}"
            )));
        }
        let sc = unsafe { &*(*self.cf.attrs.attrs.add(i) as *const XLA_FFI_Scalar) };
        if sc.dtype != T::TAG {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected scalar of {} (tag {}), got tag {}",
                T::NAME,
                T::TAG,
                sc.dtype
            )));
        }
        // SAFETY: XLA owns the scalar payload; we read by value.
        Ok(unsafe { *(sc.value as *const T) })
    }

    pub fn attr_string(&self, name: &str) -> Result<&'cf str> {
        let i = self.find_attr(name)?;
        let ty = unsafe { *self.cf.attrs.types.add(i) };
        if ty != XLA_FFI_AttrType_STRING {
            return Err(Error::invalid_argument(format!(
                "attr `{name}`: expected STRING, got type tag {ty}"
            )));
        }
        let span = unsafe { *(self.cf.attrs.attrs.add(i) as *const *const XLA_FFI_ByteSpan) };
        let bytes = unsafe { std::slice::from_raw_parts((*span).ptr as *const u8, (*span).len) };
        std::str::from_utf8(bytes)
            .map_err(|e| Error::invalid_argument(format!("attr `{name}`: invalid UTF-8: {e}")))
    }
}
