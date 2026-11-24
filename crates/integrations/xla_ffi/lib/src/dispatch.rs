// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The C entry point every handler is reached through.
//!
//! One handler answers for one stage. XLA calls a bundle's handlers at the
//! stages the bundle registers them for, asks each of them for its metadata
//! first, and expects a panic never to cross the boundary — so this reports the
//! API version and state type, refuses a frame from a stage this handler is not,
//! and turns both a returned error and an unwind into an XLA error object.

use std::panic;

use xla_ffi_sys::{
    XLA_FFI_API_MAJOR, XLA_FFI_API_MINOR, XLA_FFI_Api_Version, XLA_FFI_Api_Version_STRUCT_SIZE,
    XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_ExecutionStage, XLA_FFI_ExecutionStage_EXECUTE,
    XLA_FFI_ExecutionStage_INITIALIZE, XLA_FFI_ExecutionStage_INSTANTIATE,
    XLA_FFI_ExecutionStage_PREPARE, XLA_FFI_Extension_Metadata, XLA_FFI_Metadata_Extension,
    XLA_FFI_TypeId,
};

use crate::decode::DecodeState;
use crate::error::Error;
use crate::state::RegisteredType;

/// One stage's name, for a message a caller can act on.
fn stage_name(stage: XLA_FFI_ExecutionStage) -> &'static str {
    // The stages are C constants rather than a Rust enum, so they are compared
    // rather than matched: a pattern would bind the name instead of reading it.
    for (held, name) in [
        (XLA_FFI_ExecutionStage_INSTANTIATE, "instantiate"),
        (XLA_FFI_ExecutionStage_PREPARE, "prepare"),
        (XLA_FFI_ExecutionStage_INITIALIZE, "initialize"),
        (XLA_FFI_ExecutionStage_EXECUTE, "execute"),
    ] {
        if held == stage {
            return name;
        }
    }
    "an unknown stage"
}

unsafe fn populate_metadata(cf: &XLA_FFI_CallFrame, state_type_id: XLA_FFI_TypeId) {
    let ext = cf.extension_start as *mut XLA_FFI_Metadata_Extension;
    let md = unsafe { (*ext).metadata };
    if md.is_null() {
        return;
    }
    // NGC JAX 26.03 ships XLA FFI API 0.2. The vendored header can be
    // newer than the runtime, and XLA silently rejects handlers that
    // announce a higher minor version at registration time. The fields
    // used by this wrapper are covered by 0.2, so report that compatible
    // floor while continuing to build against the vendored definitions.
    let minor_version = (XLA_FFI_API_MINOR as i32).min(2);
    unsafe {
        (*md).api_version = XLA_FFI_Api_Version {
            struct_size: XLA_FFI_Api_Version_STRUCT_SIZE as usize,
            extension_start: std::ptr::null_mut(),
            major_version: XLA_FFI_API_MAJOR as i32,
            minor_version,
        };
        (*md).traits = 0;
        (*md).state_type_id = state_type_id;
    }
}

/// The entry point of a stateless execute handler.
///
/// It is what `#[handler]` expands to and what Loom's `run_recipe` is reached
/// through: one handler, registered for execute alone, holding no XLA-owned
/// state and therefore reporting no state type.
pub unsafe fn dispatch<F>(call_frame: *mut XLA_FFI_CallFrame, body: F) -> *mut XLA_FFI_Error
where
    F: FnOnce(&mut DecodeState<'_>) -> crate::Result<()> + panic::UnwindSafe,
{
    unsafe { dispatch_stage(call_frame, XLA_FFI_ExecutionStage_EXECUTE, None, body) }
}

/// The entry point of one stage of a stateful handler bundle.
///
/// `state_type` is the registration this bundle's state lives under. Every
/// handler of one bundle names the same one: XLA reads it from the metadata
/// reply and uses it to route `XLA_FFI_State_Set` and `XLA_FFI_State_Get`, so a
/// bundle whose stages disagreed about it would set state one stage and fail to
/// find it in the next.
pub unsafe fn dispatch_stage<F>(
    call_frame: *mut XLA_FFI_CallFrame,
    stage: XLA_FFI_ExecutionStage,
    state_type: Option<&dyn RegisteredType>,
    body: F,
) -> *mut XLA_FFI_Error
where
    F: FnOnce(&mut DecodeState<'_>) -> crate::Result<()> + panic::UnwindSafe,
{
    if call_frame.is_null() {
        return std::ptr::null_mut();
    }
    let cf = unsafe { &mut *call_frame };
    let api = cf.api;

    if !cf.extension_start.is_null()
        && unsafe { (*cf.extension_start).type_ } == XLA_FFI_Extension_Metadata
    {
        let id = match state_type.map(|held| unsafe { held.id(api) }).transpose() {
            Ok(id) => id.unwrap_or(XLA_FFI_TypeId { type_id: 0 }),
            Err(error) => return unsafe { error.into_xla(api) },
        };
        unsafe { populate_metadata(cf, id) };
        return std::ptr::null_mut();
    }

    // A handler decodes the frame its own stage populates. Instantiate sees
    // attributes and no buffers; execute sees both. Running one stage's body
    // over another's frame would read arguments XLA has not filled in, so a
    // frame from elsewhere is refused rather than half-answered.
    if cf.stage != stage {
        return unsafe {
            Error::invalid_argument(format!(
                "this is the {} handler and XLA called it at {}",
                stage_name(stage),
                stage_name(cf.stage)
            ))
            .into_xla(api)
        };
    }

    let r = panic::catch_unwind(|| {
        let mut state = DecodeState::new(cf);
        body(&mut state)
    });
    match r {
        Ok(Ok(())) => std::ptr::null_mut(),
        Ok(Err(e)) => unsafe { e.into_xla(api) },
        Err(_) => unsafe { Error::internal("panic in FFI handler").into_xla(api) },
    }
}
