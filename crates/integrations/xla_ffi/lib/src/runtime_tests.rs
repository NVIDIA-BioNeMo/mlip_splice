// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The boundary answered against a runtime standing in for XLA.
//!
//! Everything a stateful handler owes XLA is a property of the C ABI and not of
//! a device: which stage it answers, what it reports about itself, where its
//! state lives and who destroys it. So this implements the handful of API
//! entries this crate calls — error objects, the type registry, the state
//! slots — and drives real handlers through them. No GPU is involved.

use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use xla_ffi_sys::{
    XLA_FFI_Api, XLA_FFI_CallFrame, XLA_FFI_Error, XLA_FFI_Error_Create_Args,
    XLA_FFI_Error_Destroy_Args, XLA_FFI_Error_GetMessage_Args, XLA_FFI_ExecutionContext,
    XLA_FFI_ExecutionStage, XLA_FFI_ExecutionStage_EXECUTE, XLA_FFI_ExecutionStage_INSTANTIATE,
    XLA_FFI_ExecutionStage_PREPARE, XLA_FFI_Extension_Base, XLA_FFI_Extension_Metadata,
    XLA_FFI_Metadata, XLA_FFI_Metadata_Extension, XLA_FFI_State_Get_Args, XLA_FFI_State_Set_Args,
    XLA_FFI_Type_Register_Args, XLA_FFI_TypeId,
};

use crate::error::Error;
use crate::state::{RegisteredType, StateType};

// ---- the runtime standing in for XLA ------------------------------------

/// One registered type: the name XLA knows it by and the deleter it destroys
/// state through.
struct Registered {
    name: String,
    deleter: unsafe extern "C" fn(*mut c_void),
}

fn registry() -> &'static Mutex<Vec<Registered>> {
    static HELD: OnceLock<Mutex<Vec<Registered>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(Vec::new()))
}

unsafe extern "C" fn error_create(args: *mut XLA_FFI_Error_Create_Args) -> *mut XLA_FFI_Error {
    let message = unsafe { CStr::from_ptr((*args).message) }.to_owned();
    Box::into_raw(Box::new(message)).cast()
}

unsafe extern "C" fn error_get_message(args: *mut XLA_FFI_Error_GetMessage_Args) {
    let held = unsafe { &*(*args).error.cast::<CString>() };
    unsafe { (*args).message = held.as_ptr() };
}

unsafe extern "C" fn error_destroy(args: *mut XLA_FFI_Error_Destroy_Args) {
    let error = unsafe { (*args).error };
    if !error.is_null() {
        drop(unsafe { Box::from_raw(error.cast::<CString>()) });
    }
}

unsafe extern "C" fn type_register(args: *mut XLA_FFI_Type_Register_Args) -> *mut XLA_FFI_Error {
    let args = unsafe { &mut *args };
    let name = String::from_utf8_lossy(unsafe {
        std::slice::from_raw_parts(args.name.ptr.cast::<u8>(), args.name.len)
    })
    .into_owned();
    let deleter = unsafe { (*args.type_info).deleter }.expect("a state type registers a deleter");
    let mut held = registry().lock().unwrap();
    let at = held.iter().position(|entry| entry.name == name);
    let at = match at {
        Some(at) => at,
        None => {
            held.push(Registered { name, deleter });
            held.len() - 1
        }
    };
    unsafe {
        (*args.type_id).type_id = at as i64 + 1;
    }
    std::ptr::null_mut()
}

/// One executable's state slots, which is what an execution context is here.
///
/// Setting twice is refused exactly as XLA refuses it, and dropping the context
/// destroys whatever is held through the deleter its type registered — the two
/// facts the ownership of handler state rests on.
#[derive(Default)]
struct Context {
    held: Mutex<Vec<(XLA_FFI_ExecutionStage, XLA_FFI_TypeId, *mut c_void)>>,
}

impl Context {
    fn raw(&self) -> *mut XLA_FFI_ExecutionContext {
        (self as *const Context as *mut Context).cast()
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        let registered = registry().lock().unwrap();
        for (_, id, state) in self.held.lock().unwrap().drain(..) {
            let entry = &registered[(id.type_id - 1) as usize];
            unsafe { (entry.deleter)(state) };
        }
    }
}

unsafe extern "C" fn state_set(args: *mut XLA_FFI_State_Set_Args) -> *mut XLA_FFI_Error {
    let args = unsafe { &mut *args };
    let context = unsafe { &*args.ctx.cast::<Context>() };
    let id = unsafe { *args.type_id };
    let mut held = context.held.lock().unwrap();
    if held
        .iter()
        .any(|(stage, at, _)| *stage == args.stage && at.type_id == id.type_id)
    {
        return unsafe {
            error_create(&mut XLA_FFI_Error_Create_Args {
                struct_size: 0,
                extension_start: std::ptr::null_mut(),
                message: c"state is already set".as_ptr(),
                errc: xla_ffi_sys::XLA_FFI_Error_Code_INTERNAL,
            })
        };
    }
    held.push((args.stage, id, args.state));
    std::ptr::null_mut()
}

unsafe extern "C" fn state_get(args: *mut XLA_FFI_State_Get_Args) -> *mut XLA_FFI_Error {
    let args = unsafe { &mut *args };
    let context = unsafe { &*args.ctx.cast::<Context>() };
    let id = unsafe { *args.type_id };
    let held = context.held.lock().unwrap();
    args.state = held
        .iter()
        .find(|(stage, at, _)| *stage == args.stage && at.type_id == id.type_id)
        .map_or(std::ptr::null_mut(), |(_, _, state)| *state);
    std::ptr::null_mut()
}

/// The API table, which is a plain C struct of function pointers: XLA hands one
/// out per runtime and every thread reads it, so sharing it is what it is for.
struct Api(XLA_FFI_Api);

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> &'static XLA_FFI_Api {
    static HELD: OnceLock<Api> = OnceLock::new();
    &HELD
        .get_or_init(|| {
            Api(XLA_FFI_Api {
                XLA_FFI_Error_Create: Some(error_create),
                XLA_FFI_Error_GetMessage: Some(error_get_message),
                XLA_FFI_Error_Destroy: Some(error_destroy),
                XLA_FFI_Type_Register: Some(type_register),
                XLA_FFI_State_Set: Some(state_set),
                XLA_FFI_State_Get: Some(state_get),
                ..XLA_FFI_Api::default()
            })
        })
        .0
}

/// One call frame at one stage, with no arguments and no results: every fact
/// under test here is about the stage, the metadata and the state.
fn frame(context: &Context, stage: XLA_FFI_ExecutionStage) -> XLA_FFI_CallFrame {
    XLA_FFI_CallFrame {
        api: api(),
        ctx: context.raw(),
        stage,
        ..XLA_FFI_CallFrame::default()
    }
}

/// What an XLA error says, released the way a runtime releases it.
fn message(error: *mut XLA_FFI_Error) -> String {
    assert!(!error.is_null(), "this call was expected to fail");
    let mut args = XLA_FFI_Error_GetMessage_Args {
        struct_size: 0,
        extension_start: std::ptr::null_mut(),
        error,
        message: std::ptr::null(),
    };
    unsafe { error_get_message(&mut args) };
    let held = unsafe { CStr::from_ptr(args.message) }
        .to_string_lossy()
        .into_owned();
    unsafe {
        error_destroy(&mut XLA_FFI_Error_Destroy_Args {
            struct_size: 0,
            extension_start: std::ptr::null_mut(),
            error,
        })
    };
    held
}

// ---- the state under test ------------------------------------------------

/// A state that says when it is destroyed, which is the only way to observe
/// that XLA ran the deleter its type registered.
struct Counted {
    label: &'static str,
}

static LIVE: AtomicUsize = AtomicUsize::new(0);

impl Counted {
    fn new(label: &'static str) -> Counted {
        LIVE.fetch_add(1, Ordering::SeqCst);
        Counted { label }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        LIVE.fetch_sub(1, Ordering::SeqCst);
    }
}

static COUNTED: StateType<Counted> = StateType::new("xla_ffi.tests.Counted");

/// What the last execute read back, so a test can prove it reached the state
/// instantiate set rather than one of its own.
static SEEN: Mutex<Option<&'static str>> = Mutex::new(None);

unsafe fn instantiate(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    unsafe {
        crate::dispatch_stage(
            call_frame,
            XLA_FFI_ExecutionStage_INSTANTIATE,
            Some(&COUNTED),
            |state| {
                state.set_state(
                    XLA_FFI_ExecutionStage_INSTANTIATE,
                    &COUNTED,
                    Counted::new("instantiated"),
                )
            },
        )
    }
}

unsafe fn execute(call_frame: *mut XLA_FFI_CallFrame) -> *mut XLA_FFI_Error {
    unsafe {
        crate::dispatch_stage(
            call_frame,
            XLA_FFI_ExecutionStage_EXECUTE,
            Some(&COUNTED),
            |state| {
                let held = state.state(XLA_FFI_ExecutionStage_INSTANTIATE, &COUNTED)?;
                *SEEN.lock().unwrap() = Some(held.label);
                Ok(())
            },
        )
    }
}

// ---- the facts -----------------------------------------------------------

/// The state one instantiate created is the state every execute of that call
/// site reads, and destroying the executable destroys exactly it.
#[test]
fn state_outlives_instantiate_and_dies_with_the_executable() {
    let before = LIVE.load(Ordering::SeqCst);
    {
        let context = Context::default();
        let mut held = frame(&context, XLA_FFI_ExecutionStage_INSTANTIATE);
        assert!(unsafe { instantiate(&mut held) }.is_null());
        assert_eq!(LIVE.load(Ordering::SeqCst), before + 1);

        for _ in 0..3 {
            let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
            assert!(unsafe { execute(&mut held) }.is_null());
        }
        assert_eq!(*SEEN.lock().unwrap(), Some("instantiated"));
        assert_eq!(
            LIVE.load(Ordering::SeqCst),
            before + 1,
            "executing does not create state"
        );
    }
    assert_eq!(
        LIVE.load(Ordering::SeqCst),
        before,
        "destroying the executable runs the registered deleter"
    );
}

/// Two executables are two states: a call site's state is the call site's, and
/// nothing here is a process-global map.
#[test]
fn two_call_sites_hold_their_own_state() {
    let before = LIVE.load(Ordering::SeqCst);
    let first = Context::default();
    let second = Context::default();
    for context in [&first, &second] {
        let mut held = frame(context, XLA_FFI_ExecutionStage_INSTANTIATE);
        assert!(unsafe { instantiate(&mut held) }.is_null());
    }
    assert_eq!(LIVE.load(Ordering::SeqCst), before + 2);
    drop(first);
    assert_eq!(LIVE.load(Ordering::SeqCst), before + 1);
    drop(second);
    assert_eq!(LIVE.load(Ordering::SeqCst), before);
}

/// An execute of a call site whose instantiate never ran says so, rather than
/// reading a null pointer as a state.
#[test]
fn state_that_was_never_set_is_refused() {
    let context = Context::default();
    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    let said = message(unsafe { execute(&mut held) });
    assert!(said.contains("instantiate stage never ran"), "{said}");
}

/// A refused `XLA_FFI_State_Set` gives the value straight back, so a second
/// instantiate of one call site drops what it made instead of leaking it.
#[test]
fn a_refused_set_drops_the_state_it_was_given() {
    let before = LIVE.load(Ordering::SeqCst);
    let context = Context::default();
    let mut held = frame(&context, XLA_FFI_ExecutionStage_INSTANTIATE);
    assert!(unsafe { instantiate(&mut held) }.is_null());
    let said = message(unsafe { instantiate(&mut held) });
    assert!(said.contains("state is already set"), "{said}");
    assert_eq!(
        LIVE.load(Ordering::SeqCst),
        before + 1,
        "the refused state was dropped, not leaked"
    );
    drop(context);
    assert_eq!(LIVE.load(Ordering::SeqCst), before);
}

/// A handler answers one stage. A frame from another one is refused by name,
/// because its arguments are not the ones this body decodes.
#[test]
fn a_frame_from_another_stage_is_refused_by_name() {
    let context = Context::default();
    let mut held = frame(&context, XLA_FFI_ExecutionStage_PREPARE);
    let said = message(unsafe { execute(&mut held) });
    assert_eq!(
        said,
        "this is the execute handler and XLA called it at prepare"
    );

    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    let said = message(unsafe { instantiate(&mut held) });
    assert_eq!(
        said,
        "this is the instantiate handler and XLA called it at execute"
    );
}

/// Metadata is answered before any stage check, and it names the state type —
/// which is how XLA routes this bundle's `State_Set` and `State_Get`.
#[test]
fn metadata_reports_the_api_version_and_the_state_type() {
    let context = Context::default();
    let mut metadata = XLA_FFI_Metadata {
        struct_size: std::mem::size_of::<XLA_FFI_Metadata>(),
        ..XLA_FFI_Metadata::default()
    };
    let mut extension = XLA_FFI_Metadata_Extension {
        extension_base: XLA_FFI_Extension_Base {
            struct_size: std::mem::size_of::<XLA_FFI_Metadata_Extension>(),
            type_: XLA_FFI_Extension_Metadata,
            next: std::ptr::null_mut(),
        },
        metadata: &mut metadata,
    };
    // A stage nobody answers: metadata is asked of a handler before it is ever
    // called, so answering it must not depend on the frame's stage.
    let mut held = frame(&context, XLA_FFI_ExecutionStage_PREPARE);
    held.extension_start = (&mut extension as *mut XLA_FFI_Metadata_Extension).cast();
    assert!(unsafe { execute(&mut held) }.is_null());

    assert_eq!(
        metadata.api_version.major_version,
        xla_ffi_sys::XLA_FFI_API_MAJOR as i32
    );
    assert_eq!(metadata.traits, 0);
    let stated = unsafe { COUNTED.id(api()) }.unwrap();
    assert_ne!(stated.type_id, 0);
    assert_eq!(metadata.state_type_id.type_id, stated.type_id);

    // And the same bundle's other stage reports the same one, or state set in
    // one would not be found in the other.
    metadata.state_type_id = XLA_FFI_TypeId { type_id: 0 };
    assert!(unsafe { instantiate(&mut held) }.is_null());
    assert_eq!(metadata.state_type_id.type_id, stated.type_id);
}

/// A stateless execute handler — Loom's — reports no state type and still
/// answers metadata and its own stage.
#[test]
fn a_stateless_handler_reports_no_state_type() {
    let context = Context::default();
    let mut metadata = XLA_FFI_Metadata {
        struct_size: std::mem::size_of::<XLA_FFI_Metadata>(),
        state_type_id: XLA_FFI_TypeId { type_id: 7 },
        ..XLA_FFI_Metadata::default()
    };
    let mut extension = XLA_FFI_Metadata_Extension {
        extension_base: XLA_FFI_Extension_Base {
            struct_size: std::mem::size_of::<XLA_FFI_Metadata_Extension>(),
            type_: XLA_FFI_Extension_Metadata,
            next: std::ptr::null_mut(),
        },
        metadata: &mut metadata,
    };
    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    held.extension_start = (&mut extension as *mut XLA_FFI_Metadata_Extension).cast();
    assert!(unsafe { crate::dispatch(&mut held, |_| Ok(())) }.is_null());
    assert_eq!(metadata.state_type_id.type_id, 0);

    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    assert!(unsafe { crate::dispatch(&mut held, |_| Ok(())) }.is_null());
}

/// A panic is contained and reported. Unwinding across the C boundary is
/// undefined behaviour, so this is the one thing a handler may never do.
#[test]
fn a_panic_becomes_an_error_rather_than_an_unwind() {
    let context = Context::default();
    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let error = unsafe { crate::dispatch(&mut held, |_| panic!("the handler gave up")) };
    std::panic::set_hook(hook);
    assert_eq!(message(error), "panic in FFI handler");
}

/// An error a handler returns arrives as the message it wrote.
#[test]
fn a_returned_error_arrives_with_its_message() {
    let context = Context::default();
    let mut held = frame(&context, XLA_FFI_ExecutionStage_EXECUTE);
    let error = unsafe {
        crate::dispatch(&mut held, |_| {
            Err(Error::invalid_argument("format 9 is unsupported"))
        })
    };
    assert_eq!(message(error), "format 9 is unsupported");
}

/// A null frame is XLA asking nothing, and a call frame with no API table is a
/// question the registry cannot answer — neither may dereference null.
#[test]
fn a_frame_without_an_api_table_is_refused() {
    assert!(unsafe { crate::dispatch(std::ptr::null_mut(), |_| Ok(())) }.is_null());
    assert!(unsafe { COUNTED.id(std::ptr::null()) }.is_err());
}

/// One name is one id, however often it is asked for: a handler registers its
/// type once per API instance and reuses what that instance assigned.
#[test]
fn one_type_keeps_one_id() {
    static OTHER: StateType<Counted> = StateType::new("xla_ffi.tests.Other");
    let first = unsafe { COUNTED.id(api()) }.unwrap();
    assert_eq!(unsafe { COUNTED.id(api()) }.unwrap().type_id, first.type_id);
    assert_ne!(unsafe { OTHER.id(api()) }.unwrap().type_id, first.type_id);
}

/// Ids come from the API instance, so the registry is keyed by it rather than
/// memoized once for the process.
#[test]
fn a_second_api_instance_gets_its_own_id() {
    static ANOTHER: StateType<Counted> = StateType::new("xla_ffi.tests.Another");
    static NEXT: AtomicI64 = AtomicI64::new(1000);
    unsafe extern "C" fn counting(args: *mut XLA_FFI_Type_Register_Args) -> *mut XLA_FFI_Error {
        unsafe { (*(*args).type_id).type_id = NEXT.fetch_add(1, Ordering::SeqCst) };
        std::ptr::null_mut()
    }
    let other = XLA_FFI_Api {
        XLA_FFI_Type_Register: Some(counting),
        ..XLA_FFI_Api::default()
    };
    let mine = unsafe { ANOTHER.id(api()) }.unwrap();
    let theirs = unsafe { ANOTHER.id(&other) }.unwrap();
    assert_ne!(mine.type_id, theirs.type_id);
    assert_eq!(
        unsafe { ANOTHER.id(&other) }.unwrap().type_id,
        theirs.type_id
    );
}

/// The deleter that runs is the one the state's own type registered.
///
/// XLA routes destruction by type id, so a call site holding two kinds of state
/// destroys each through its own registration. This is the whole of what a
/// handler relies on when its state owns a device resource: dropping the box
/// drops the value, and dropping the value releases what it holds.
#[test]
fn each_state_type_is_destroyed_through_its_own_deleter() {
    struct Other;

    static OTHERS: AtomicUsize = AtomicUsize::new(0);

    impl Drop for Other {
        fn drop(&mut self) {
            OTHERS.fetch_add(1, Ordering::SeqCst);
        }
    }

    static OTHER: StateType<Other> = StateType::new("xla_ffi.tests.OtherState");

    let live = LIVE.load(Ordering::SeqCst);
    let destroyed = OTHERS.load(Ordering::SeqCst);
    {
        let context = Context::default();
        let mut held = frame(&context, XLA_FFI_ExecutionStage_INSTANTIATE);
        assert!(unsafe { instantiate(&mut held) }.is_null());
        let mut held = frame(&context, XLA_FFI_ExecutionStage_INSTANTIATE);
        assert!(
            unsafe {
                crate::dispatch_stage(
                    &mut held,
                    XLA_FFI_ExecutionStage_INSTANTIATE,
                    Some(&OTHER),
                    |state| state.set_state(XLA_FFI_ExecutionStage_INSTANTIATE, &OTHER, Other),
                )
            }
            .is_null()
        );
        assert_eq!(LIVE.load(Ordering::SeqCst), live + 1);
        assert_eq!(OTHERS.load(Ordering::SeqCst), destroyed);
    }
    assert_eq!(LIVE.load(Ordering::SeqCst), live, "one deleter per type");
    assert_eq!(OTHERS.load(Ordering::SeqCst), destroyed + 1);
}
