// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a launch, a module of launches, and the target they render for
//! must satisfy.

use super::*;

#[test]
fn a_launched_kernel_hands_nothing_back() {
    let mut b = Builder::new();
    let one = b.float(Scalar::F32, 1.0);
    let one = b.bind_value_as("one", Ty::scalar(Scalar::F32), one);
    b.returns(vec![one]);
    let function = DeviceFunction::new("k", b.into_function());
    verify(&function).expect("a function may hand back a value");
    assert_eq!(
        verify_launch(&Launch::new(function, 64, None))
            .expect_err("a kernel may not")
            .rule,
        Rule::Abi
    );
}

#[test]
fn a_launch_states_a_block_and_a_cap_a_target_admits() {
    let body = || Builder::new().into_function();
    for (threads, cap) in [(0, None), (2048, None), (64, Some(4)), (64, Some(300))] {
        let launch = Launch::new(DeviceFunction::new("k", body()), threads, cap);
        assert_eq!(
            verify_launch(&launch)
                .expect_err("outside what a target admits")
                .rule,
            Rule::Abi
        );
    }
    verify_launch(&Launch::new(
        DeviceFunction::new("k", body()),
        1024,
        Some(255),
    ))
    .expect("the widest block and the highest cap are admitted");
}

/// Shared memory belongs to the kernel that is launched, so a callee cannot
/// declare any of its own.
#[test]
fn a_callee_declaring_shared_memory_is_refused() {
    let mut b = Builder::new();
    b.shared_array(SharedArr {
        name: "stripe".into(),
        elem: Scalar::F32,
        len: 32,
    });
    let callee = Arc::new(DeviceFunction::new("helper", b.into_function()));
    let mut b = Builder::new();
    b.call(callee, Vec::new());
    assert_eq!(refuses(b.into_function()).rule, Rule::Abi);
}

/// Two launches, for a module verified together.
fn module(a: Launch, b: Launch, sm: i32) -> crate::verify::Result {
    crate::verify_module(&[&a, &b], sm)
}

/// A constant is emitted once under its name, so two launches reading
/// different arrays under one name is a module that cannot be written.
#[test]
fn one_constant_name_holding_two_arrays_is_refused() {
    let reads = |name: &str, data: Vec<u8>, entry: &str| {
        let mut b = Builder::new();
        let out = b.parameter(Param {
            name: "out".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let zero = b.i32(0);
        let held = b.const_load(name, Scalar::F32, data, zero.clone());
        let at = b.offset(out, zero);
        b.store(at, held);
        Launch::new(DeviceFunction::new(entry, b.into_function()), 64, None)
    };
    let refused = module(
        reads("table", vec![0u8; 4], "a"),
        reads("table", vec![0u8; 8], "b"),
        90,
    )
    .expect_err("one name, two arrays");
    assert_eq!(refused.rule, Rule::Memory);
    module(
        reads("table", vec![0u8; 4], "a"),
        reads("table", vec![0u8; 4], "b"),
        90,
    )
    .expect("one name, one array");
}

/// Two entries of one name would declare the entry twice.
#[test]
fn two_launches_entered_under_one_name_are_refused() {
    let entry = |name: &str| Launch::new(DeviceFunction::new(name, Function::new()), 64, None);
    assert_eq!(
        module(entry("k"), entry("k"), 90)
            .expect_err("one entry name, twice")
            .rule,
        Rule::Abi
    );
    module(entry("k"), entry("j"), 90).expect("two names");
}

/// A parameter and a shared array of one name would declare one symbol twice.
#[test]
fn a_parameter_and_a_shared_array_of_one_name_are_refused() {
    let mut b = Builder::new();
    b.parameter(Param {
        name: "held".into(),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    // The builder refuses a repeated declaration, so the arena is written
    // directly to prove the module pass decides it too.
    let mut f = b.into_function();
    f.declare(
        Declaration::Shared(SharedArr {
            name: "held".into(),
            elem: Scalar::F32,
            len: 4,
        }),
        ValTy::Ptr(PtrTy {
            elem: Scalar::F32,
            space: Space::Shared,
        }),
    );
    let launch = Launch::new(DeviceFunction::new("k", f), 64, None);
    assert_eq!(
        crate::verify_module(&[&launch], 90)
            .expect_err("one name, two declarations")
            .rule,
        Rule::Abi
    );
}

/// A name the renderer cannot print is not escaped into one: it would land in
/// the middle of a declaration saying whatever it says.
#[test]
fn a_name_ptx_cannot_be_given_is_refused() {
    for held in ["", "0k", "a b", "a;\n.visible .entry evil(", "a-b"] {
        let launch = Launch::new(DeviceFunction::new(held, Function::new()), 64, None);
        assert_eq!(
            crate::verify_module(&[&launch], 90)
                .expect_err("not a PTX identifier")
                .rule,
            Rule::Abi,
            "{held:?}"
        );
    }
    for held in ["k", "_k", "$k", "k0", "loom_u$1"] {
        let launch = Launch::new(DeviceFunction::new(held, Function::new()), 64, None);
        crate::verify_module(&[&launch], 90).expect("a PTX identifier");
    }
}

/// A reduction of a float width the target has no instruction for cannot be
/// split into anything equivalent, so it is refused for that target rather than
/// lowered into something else.
#[test]
fn an_atomic_add_states_the_target_it_needs() {
    // The two forms of one addition answer to one gate: whether the previous
    // value comes back cannot change which targets have the instruction.
    let reduces = |scalar: Scalar, sm: i32, hands_back: bool| {
        let mut b = Builder::new();
        let total = b.parameter(Param {
            name: "total".into(),
            ty: ParamTy::Ptr(scalar),
        });
        let zero = b.i32(0);
        let one = b.float(scalar, 1.0);
        let at = b.offset(total, zero);
        if hands_back {
            let before = b.fetch_add(at.clone(), one);
            b.store(at, before);
        } else {
            b.red_add(at, one);
        }
        let launch = Launch::new(DeviceFunction::new("k", b.into_function()), 64, None);
        crate::verify_module(&[&launch], sm)
    };
    for hands_back in [false, true] {
        for (scalar, since) in [(Scalar::F64, 60), (Scalar::F16, 70), (Scalar::BF16, 90)] {
            let refused = reduces(scalar, since - 10, hands_back).expect_err("below what it needs");
            assert_eq!(refused.rule, Rule::Memory, "{scalar:?}");
            assert!(refused.detail.contains(&format!("sm_{since}")), "{refused}");
            reduces(scalar, since, hands_back).expect("exactly what it needs");
            reduces(scalar, since + 10, hands_back).expect("past what it needs");
        }
        // A 32-bit float reduction is universal.
        reduces(Scalar::F32, 60, hands_back).expect("f32 needs no target");
    }
}

#[test]
fn unsupported_targets_are_refused_before_rendering() {
    for sm in [-1, 0, 88, 99, 122, i32::MAX] {
        let launch = Launch::new(DeviceFunction::new("k", Function::new()), 64, None);
        let refused = crate::verify_module(&[&launch], sm).expect_err("unknown target");
        assert_eq!(refused.rule, Rule::Abi, "sm_{sm}");
        assert!(refused.detail.contains(&format!("sm_{sm}")), "{refused}");
    }
    for sm in [50, 80, 90, 100, 103, 110, 120, 121] {
        let launch = Launch::new(DeviceFunction::new("k", Function::new()), 64, None);
        crate::verify_module(&[&launch], sm).expect("known target");
    }
}

#[test]
fn a_kernel_predicate_parameter_is_refused() {
    let mut b = Builder::new();
    b.parameter(Param {
        name: "predicate".into(),
        ty: ParamTy::Scalar(Scalar::Pred),
    });
    assert_eq!(
        verify_launch(&Launch::new(
            DeviceFunction::new("k", b.into_function()),
            64,
            None,
        ))
        .expect_err("predicates do not cross the launch ABI")
        .rule,
        Rule::Abi
    );
}

#[test]
fn an_overflowing_shared_allocation_is_refused() {
    let mut b = Builder::new();
    b.shared_array(SharedArr {
        name: "too_large".into(),
        elem: Scalar::F64,
        len: i64::MAX,
    });
    assert_eq!(
        verify_launch(&Launch::new(
            DeviceFunction::new("k", b.into_function()),
            64,
            None,
        ))
        .expect_err("its byte size overflows")
        .rule,
        Rule::Abi
    );
}

#[test]
fn declarations_and_entry_values_have_one_abi() {
    let mut f = Function::new();
    f.declare(
        Declaration::Param(Param {
            name: "x".into(),
            ty: ParamTy::Scalar(Scalar::S32),
        }),
        ValTy::Data(Ty::scalar(Scalar::F32)),
    );
    let refused = verify_launch(&Launch::new(DeviceFunction::new("k", f), 64, None))
        .expect_err("the entry value disagrees with its declaration");
    assert_eq!(refused.rule, Rule::Abi);
}

#[test]
fn constants_do_not_collide_with_function_symbols() {
    let reads = |entry: &str| {
        let mut b = Builder::new();
        let out = b.parameter(Param {
            name: "out".into(),
            ty: ParamTy::Ptr(Scalar::F32),
        });
        let zero = b.i32(0);
        let held = b.const_load("x", Scalar::F32, vec![0; 4], zero.clone());
        let at = b.offset(out, zero);
        b.store(at, held);
        Launch::new(DeviceFunction::new(entry, b.into_function()), 64, None)
    };
    let launch = reads("const_x");
    assert_eq!(
        crate::verify_module(&[&launch], 90)
            .expect_err("the emitted names collide")
            .rule,
        Rule::Abi
    );
}
