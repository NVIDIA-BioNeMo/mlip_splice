// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State XLA owns on a handler's behalf, for the lifetime of one call site.
//!
//! An instantiate handler builds a value and hands it to XLA; every execute of
//! that call site borrows it back; destroying the executable destroys it through
//! the deleter registered with its type. Nothing here is a process-global map
//! and nothing outlives the executable that asked for it, so a handler needs no
//! immortal handle and no address travelling through the compiled program.
//!
//! Type ids are per FFI API instance, not per process: the same extension loaded
//! by two runtimes is registered once in each. A [`StateType`] is therefore a
//! registry keyed by the API pointer, and the id it hands back is the one that
//! API assigned.

use std::any::type_name;
use std::ffi::c_void;
use std::marker::PhantomData;
use std::sync::Mutex;

use xla_ffi_sys::{
    XLA_FFI_Api, XLA_FFI_ByteSpan, XLA_FFI_Error, XLA_FFI_Error_Destroy_Args,
    XLA_FFI_Error_Destroy_Args_STRUCT_SIZE, XLA_FFI_ExecutionStage, XLA_FFI_State_Get_Args,
    XLA_FFI_State_Get_Args_STRUCT_SIZE, XLA_FFI_State_Set_Args, XLA_FFI_State_Set_Args_STRUCT_SIZE,
    XLA_FFI_Type_Register_Args, XLA_FFI_Type_Register_Args_STRUCT_SIZE, XLA_FFI_TypeId,
    XLA_FFI_TypeInfo, XLA_FFI_TypeInfo_STRUCT_SIZE,
};

use crate::decode::DecodeState;
use crate::error::{Error, Result};

/// A Rust type XLA has been told about, as the id one API instance assigned it.
///
/// It is a trait because the metadata a handler reports must name its state
/// type without naming that type's parameters: a dispatcher takes whichever
/// registration its handler owns and asks it for an id.
pub trait RegisteredType: Sync {
    /// # Safety
    ///
    /// `api` must be a live XLA FFI API table — the one a call frame carries.
    unsafe fn id(&self, api: *const XLA_FFI_Api) -> Result<XLA_FFI_TypeId>;
}

/// One Rust type registered as XLA-owned handler state.
///
/// The deleter is part of the registration: when an executable dies, XLA drops
/// the exact box an instantiate handler set, through the destructor of the type
/// this stands for.
///
/// `T` must be usable from any thread. XLA instantiates on the thread that
/// compiles and executes on whichever thread runs the program, and it may run
/// one executable concurrently — so the state is shared, never owned by one
/// caller, which is why [`DecodeState::state`] hands back a shared reference.
pub struct StateType<T> {
    name: &'static str,
    ids: Mutex<Vec<(usize, XLA_FFI_TypeId)>>,
    marker: PhantomData<fn() -> T>,
}

impl<T> StateType<T> {
    /// A registration under the name XLA will know this type by. The name is
    /// the identity XLA matches on, so two builds that mean the same state must
    /// spell it the same way.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            ids: Mutex::new(Vec::new()),
            marker: PhantomData,
        }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }
}

impl<T: Send + Sync + 'static> RegisteredType for StateType<T> {
    unsafe fn id(&self, api: *const XLA_FFI_Api) -> Result<XLA_FFI_TypeId> {
        if api.is_null() {
            return Err(Error::internal("XLA FFI call frame carries no API table"));
        }
        let key = api as usize;
        let mut ids = self
            .ids
            .lock()
            .map_err(|_| Error::internal("XLA FFI state type registry lock poisoned"))?;
        if let Some((_, id)) = ids.iter().find(|(held, _)| *held == key) {
            return Ok(*id);
        }

        unsafe extern "C" fn delete<T>(object: *mut c_void) {
            if !object.is_null() {
                drop(unsafe { Box::from_raw(object.cast::<T>()) });
            }
        }

        let info = XLA_FFI_TypeInfo {
            struct_size: XLA_FFI_TypeInfo_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            deleter: Some(delete::<T>),
        };
        let mut id = XLA_FFI_TypeId { type_id: 0 };
        let mut args = XLA_FFI_Type_Register_Args {
            struct_size: XLA_FFI_Type_Register_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            name: XLA_FFI_ByteSpan {
                ptr: self.name.as_ptr().cast(),
                len: self.name.len(),
            },
            type_id: &mut id,
            type_info: &info,
        };
        let register = unsafe { (*api).XLA_FFI_Type_Register }
            .ok_or_else(|| Error::internal("XLA_FFI_Type_Register is null"))?;
        let error = unsafe { register(&mut args) };
        if !error.is_null() {
            return Err(unsafe { taken(api, error, &format!("registering {}", self.name)) });
        }
        if id.type_id == 0 {
            return Err(Error::internal(format!(
                "XLA registered {} with the unknown type id",
                self.name
            )));
        }
        ids.push((key, id));
        Ok(id)
    }
}

/// The message an XLA error carries, and the error released either way.
///
/// XLA owns the object it handed back, so reading it and destroying it are one
/// action: a handler that returned without destroying would leak one error per
/// failed call.
unsafe fn taken(api: *const XLA_FFI_Api, error: *mut XLA_FFI_Error, doing: &str) -> Error {
    let message = match unsafe { (*api).XLA_FFI_Error_GetMessage } {
        Some(get) => {
            let mut args = xla_ffi_sys::XLA_FFI_Error_GetMessage_Args {
                struct_size: xla_ffi_sys::XLA_FFI_Error_GetMessage_Args_STRUCT_SIZE as usize,
                extension_start: std::ptr::null_mut(),
                error,
                message: std::ptr::null(),
            };
            unsafe { get(&mut args) };
            if args.message.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(args.message) }
                    .to_string_lossy()
                    .into_owned()
            }
        }
        None => String::new(),
    };
    if let Some(destroy) = unsafe { (*api).XLA_FFI_Error_Destroy } {
        let mut args = XLA_FFI_Error_Destroy_Args {
            struct_size: XLA_FFI_Error_Destroy_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            error,
        };
        unsafe { destroy(&mut args) };
    }
    if message.is_empty() {
        Error::internal(format!("XLA refused {doing}"))
    } else {
        Error::internal(format!("XLA refused {doing}: {message}"))
    }
}

impl<'cf> DecodeState<'cf> {
    /// Hand `state` to XLA as this call site's state for `stage`.
    ///
    /// Ownership moves: XLA holds the box and drops it through the type's
    /// deleter when the executable dies. A refusal gives it straight back, so a
    /// failed set drops the value here rather than leaking it.
    pub fn set_state<T: Send + Sync + 'static>(
        &self,
        stage: XLA_FFI_ExecutionStage,
        state_type: &StateType<T>,
        state: T,
    ) -> Result<()> {
        let cf = self.call_frame();
        let mut id = unsafe { state_type.id(cf.api) }?;
        let raw = Box::into_raw(Box::new(state));
        let mut args = XLA_FFI_State_Set_Args {
            struct_size: XLA_FFI_State_Set_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            ctx: cf.ctx,
            stage,
            type_id: &mut id,
            state: raw.cast(),
        };
        let set = unsafe { (*cf.api).XLA_FFI_State_Set }
            .ok_or_else(|| Error::internal("XLA_FFI_State_Set is null"))?;
        let error = unsafe { set(&mut args) };
        if error.is_null() {
            return Ok(());
        }
        drop(unsafe { Box::from_raw(raw) });
        Err(unsafe {
            taken(
                cf.api,
                error,
                &format!("the state {} set", state_type.name()),
            )
        })
    }

    /// This call site's state, as the type it was set under.
    ///
    /// Shared, never exclusive: XLA may execute one call site on several
    /// threads at once, so a handler that needs to change what it holds owns
    /// that synchronization itself. It outlives the call frame — XLA destroys
    /// it with the executable — so reading it does not pin the frame.
    pub fn state<T: Send + Sync + 'static>(
        &self,
        stage: XLA_FFI_ExecutionStage,
        state_type: &StateType<T>,
    ) -> Result<&'cf T> {
        let cf = self.call_frame();
        let mut id = unsafe { state_type.id(cf.api) }?;
        let mut args = XLA_FFI_State_Get_Args {
            struct_size: XLA_FFI_State_Get_Args_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            ctx: cf.ctx,
            stage,
            type_id: &mut id,
            state: std::ptr::null_mut(),
        };
        let get = unsafe { (*cf.api).XLA_FFI_State_Get }
            .ok_or_else(|| Error::internal("XLA_FFI_State_Get is null"))?;
        let error = unsafe { get(&mut args) };
        if !error.is_null() {
            return Err(unsafe {
                taken(
                    cf.api,
                    error,
                    &format!("reading the state {}", state_type.name()),
                )
            });
        }
        if args.state.is_null() {
            return Err(Error::internal(format!(
                "this call site holds no {} state, so its instantiate stage never ran",
                type_name::<T>()
            )));
        }
        Ok(unsafe { &*args.state.cast::<T>() })
    }
}
