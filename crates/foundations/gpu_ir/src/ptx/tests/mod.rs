// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the renderer emits, read back from the text it produced.

mod aliasing;
mod atomics;
mod carrying;
mod control_flow;
mod divisors;
mod expressions;
mod storage;
mod takeover;
mod text_size;
mod warping;

use super::render_ptx_module;
use crate::build::{Builder, Value};
use crate::func::{DeviceFunction, Launch};
use crate::{Carry, CmpOp, Param, ParamTy, Scalar, SharedArr, Trip, Ty};

/// A kernel named `k` over the body `build` writes, launched at 64 threads.
pub(super) fn kernel(build: impl FnOnce(&mut Builder)) -> Launch {
    capped(None, build)
}

pub(super) fn capped(max_registers: Option<u32>, build: impl FnOnce(&mut Builder)) -> Launch {
    let mut b = Builder::new();
    build(&mut b);
    Launch::new(
        DeviceFunction::new("k", b.into_function()),
        64,
        max_registers,
    )
}

pub(super) fn ptr(b: &mut Builder, name: &str, elem: Scalar) -> Value {
    b.parameter(Param {
        name: name.into(),
        ty: ParamTy::Ptr(elem),
    })
}

pub(super) fn render(kernel: &Launch, sm: i32) -> String {
    let launches = std::slice::from_ref(kernel);
    let text = render_ptx_module(launches, sm);
    let size = super::measure_module(launches, sm).unwrap();
    assert_eq!(size.module_bytes, text.len());
    assert_eq!(size.module_bytes, text.capacity());
    assert!(size.peak_buffer_bytes >= size.module_bytes);
    text
}

/// A kernel over one input and one output pointer, both `f32`, which every
/// question below is asked of.
fn io_kernel(build: impl FnOnce(&mut Builder, Value, Value)) -> Launch {
    kernel(|b| {
        let source = ptr(b, "in", Scalar::F32);
        let out = ptr(b, "out", Scalar::F32);
        build(b, source, out);
    })
}

fn lane_below_four(b: &mut Builder) -> Value {
    let tid = b.tid_x();
    let tid = b.cast(Scalar::S32, tid);
    let four = b.i32(4);
    b.lt(tid, four)
}

fn f32_load(b: &mut Builder, source: Value) -> Value {
    let tid = b.tid_x();
    let tid = b.cast(Scalar::S32, tid);
    let at = b.offset(source, tid);
    b.load(at, 1, false)
}

fn f32() -> Ty {
    Ty::scalar(Scalar::F32)
}

/// `out = fma(2, in, 0)` over `lanes` lanes, the shape every instruction
/// selection question is asked of.
fn fma_kernel(lanes: u8) -> Launch {
    kernel(|b| {
        let source = ptr(b, "in", Scalar::F32);
        let out = ptr(b, "out", Scalar::F32);
        let ty = Ty::vec(Scalar::F32, lanes);
        let zero = b.float(Scalar::F32, 0.0);
        let zero = b.splat(lanes, zero);
        let acc = b.bind_value_as("acc", ty, zero);
        let x = b.load(source, lanes, false);
        let x = b.bind_value_as("x", ty, x);
        let two = b.float(Scalar::F32, 2.0);
        let two = b.splat(lanes, two);
        let stepped = b.fma(two, x, acc.clone());
        let acc = b.update(acc, stepped);
        b.store(out, acc);
    })
}

#[test]
fn f32_pairs_pack_on_sm100_and_split_below() {
    let k = fma_kernel(4);
    let blackwell = render(&k, 120);
    assert_eq!(blackwell.matches("fma.rn.f32x2").count(), 2, "{blackwell}");
    assert!(blackwell.contains("ld.global.v4.f32"), "{blackwell}");
    let hopper = render(&k, 90);
    assert!(!hopper.contains("f32x2"), "{hopper}");
    assert_eq!(hopper.matches("fma.rn.f32 ").count(), 4, "{hopper}");
    assert!(hopper.contains("ld.global.v4.f32"), "{hopper}");
}

#[test]
fn scalar_fma_never_packs() {
    let ptx = render(&fma_kernel(1), 120);
    assert!(!ptx.contains("f32x2"), "{ptx}");
    assert_eq!(ptx.matches("fma.rn.f32 ").count(), 1, "{ptx}");
}

/// An accumulator produced into its own storage is one register: the update is
/// the instruction, not a copy after it.
#[test]
fn an_update_writes_the_register_it_reads() {
    let ptx = render(&fma_kernel(1), 120);
    let fma = ptx
        .lines()
        .find(|line| line.contains("fma.rn.f32 "))
        .expect("one fma");
    let operands = fma
        .trim()
        .trim_end_matches(';')
        .split_once(' ')
        .expect("an opcode")
        .1
        .split(", ")
        .map(str::trim)
        .collect::<Vec<_>>();
    assert_eq!(operands[0], operands[3], "{ptx}");
    assert_eq!(ptx.matches("mov.f32").count(), 1, "only the zero: {ptx}");
}

/// A fold reads its table once per iteration, inside the one loop body it
/// renders as.
#[test]
fn a_fold_renders_as_one_loop_body() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let table = [1.0f32, 2.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let start = b.float(Scalar::F32, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: start,
                into: None,
            }],
            |b, index, given| {
                let read = b.const_load("coefficients", Scalar::F32, table.clone(), index);
                (vec![b.add(given[0].clone(), read)], ())
            },
        );
        let (zero, stop, step) = (b.i32(0), b.i32(3), b.i32(1));
        let folded = b.fold_loop(Trip::PerThread, zero, stop, step, body);
        b.store(out, folded.into_iter().next().unwrap());
    });
    let ptx = render(&k, 120);
    assert!(
        ptx.contains(".global .align 4 .b8 const_coefficients[12]"),
        "{ptx}"
    );
    assert_eq!(ptx.matches("ld.global.nc.f32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("setp.lt.s32").count(), 1, "{ptx}");
}

/// A fold whose bounds are loaded reads each of them once, before the loop.
#[test]
fn a_fold_loads_dynamic_bounds_once() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let table = [1i32, 3]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let bound = |b: &mut Builder, index: i64| {
            let at = b.i32(index);
            b.const_load("bounds", Scalar::S32, table.clone(), at)
        };
        let (start, stop) = (bound(b, 0), bound(b, 1));
        let step = b.i32(1);
        let seed = b.float(Scalar::F32, 0.0);
        let (body, ()) = b.build_loop_body(
            Scalar::S32,
            vec![Carry {
                ty: Ty::scalar(Scalar::F32),
                init: seed,
                into: None,
            }],
            |b, _, given| {
                let one = b.float(Scalar::F32, 1.0);
                (vec![b.add(given[0].clone(), one)], ())
            },
        );
        let folded = b.fold_loop(Trip::PerThread, start, stop, step, body);
        b.store(out, folded.into_iter().next().unwrap());
    });
    let ptx = render(&k, 120);
    assert_eq!(ptx.matches("ld.global.nc.b32").count(), 2, "{ptx}");
    assert_eq!(ptx.matches("setp.lt.s32").count(), 1, "{ptx}");
}

/// A scatter fold carries one accumulator per slot and branches per slot. Each
/// update is bound once for the whole iteration, so widening the slot count
/// adds arms rather than loads.
#[test]
fn a_scatter_fold_loads_each_update_once() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let destinations = [0i32, 1, 2]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect::<Vec<_>>();
        let updates = [1.0f32, 2.0, 3.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let ty = Ty::scalar(Scalar::F32);
        let carried = (0..3)
            .map(|_| {
                let init = b.float(Scalar::F32, 0.0);
                Carry {
                    ty,
                    init,
                    into: None,
                }
            })
            .collect::<Vec<_>>();
        let (body, ()) = b.build_loop_body(Scalar::S32, carried, |b, index, given| {
            let slot = b.const_load(
                "destinations",
                Scalar::S32,
                destinations.clone(),
                index.clone(),
            );
            let slot = b.bind_value_as("slot", Ty::scalar(Scalar::S32), slot);
            let update = b.const_load("updates", Scalar::F32, updates.clone(), index);
            let update = b.bind_value_as("update", ty, update);
            let produced = (0..3)
                .map(|slot_index| {
                    let at = b.i32(slot_index as i64);
                    let here = b.eq(slot.clone(), at);
                    let held = given[slot_index].clone();
                    let update = update.clone();
                    b.if_producing(
                        here,
                        &[(ty, Some(held.clone()))],
                        {
                            let held = held.clone();
                            move |b| vec![b.add(held.clone(), update.clone())]
                        },
                        move |_| vec![held.clone()],
                    )
                    .into_iter()
                    .next()
                    .unwrap()
                })
                .collect::<Vec<_>>();
            (produced, ())
        });
        let (zero, stop, step) = (b.i32(0), b.i32(3), b.i32(1));
        let folded = b.place_counted_loop(Trip::PerThread, zero, stop, step, body);
        let total = folded
            .into_iter()
            .reduce(|left, right| b.add(left, right))
            .unwrap();
        b.store(out, total);
    });
    let ptx = render(&k, 120);
    assert_eq!(ptx.matches("ld.global.nc.f32").count(), 1, "{ptx}");
    assert_eq!(ptx.matches("ld.global.nc.b32").count(), 1, "{ptx}");
    assert_eq!(
        ptx.matches("$Lelse_").count(),
        6,
        "one guard per slot: {ptx}"
    );
}

#[test]
fn half_vectors_ride_b32_words() {
    let k = kernel(|b| {
        let source = ptr(b, "in", Scalar::F16);
        let out = ptr(b, "out", Scalar::F16);
        let read = b.load(source, 4, false);
        let x = b.bind_value_as("x", Ty::vec(Scalar::F16, 4), read);
        let doubled = b.add(x.clone(), x);
        b.store(out, doubled);
    });
    let ptx = render(&k, 90);
    assert!(ptx.contains("ld.global.v2.b32"), "{ptx}");
    assert_eq!(ptx.matches("add.rn.f16x2").count(), 2, "{ptx}");
    assert!(ptx.contains("st.global.v2.b32"), "{ptx}");
}

/// Load a sub-word integer, widen it to `f32`, store.
fn cast_to_f32_kernel(elem: Scalar) -> Launch {
    kernel(|b| {
        let source = ptr(b, "in", elem);
        let out = ptr(b, "out", Scalar::F32);
        let read = b.load(source, 1, false);
        let widened = b.cast(Scalar::F32, read);
        let x = b.bind_value_as("x", Ty::scalar(Scalar::F32), widened);
        b.store(out, x);
    })
}

#[test]
fn sub_word_ints_ride_b16_regs_and_cvt_to_float() {
    let s8 = render(&cast_to_f32_kernel(Scalar::S8), 120);
    assert!(s8.contains(".reg .b16 %rs"), "{s8}");
    assert!(s8.contains("ld.global.s8 %rs"), "{s8}");
    assert!(s8.contains("cvt.rn.f32.s8"), "{s8}");

    let u16 = render(&cast_to_f32_kernel(Scalar::U16), 120);
    assert!(u16.contains("ld.global.u16 %rs"), "{u16}");
    assert!(u16.contains("cvt.rn.f32.u16"), "{u16}");
}

#[test]
fn byte_values_convert_through_u32_for_predicates() {
    let k = kernel(|b| {
        let source = ptr(b, "in", Scalar::U8);
        let out = ptr(b, "out", Scalar::U8);
        let read = b.load(source, 1, false);
        let wide = b.cast(Scalar::U32, read);
        let narrow = b.cast(Scalar::U8, wide);
        b.store(out, narrow);
    });
    let ptx = render(&k, 80);
    assert!(ptx.contains("cvt.u32.u8"), "{ptx}");
    assert!(ptx.contains("cvt.u8.u32"), "{ptx}");
    assert!(ptx.contains("st.global.u8"), "{ptx}");
}

#[test]
fn sub_word_math_and_comparisons_widen_to_32_bits() {
    let k = kernel(|b| {
        let source = ptr(b, "in", Scalar::S8);
        let out = ptr(b, "out", Scalar::S8);
        let read = b.load(source, 1, false);
        let x = b.bind_value_as("x", Ty::scalar(Scalar::S8), read);
        let zero = b.zero(Scalar::S8);
        let positive = b.cmp(CmpOp::Gt, x.clone(), zero);
        let one = b.int(Scalar::S8, 1);
        let up = b.add(x.clone(), one);
        let down = b.neg(x);
        let chosen = b.select(positive, up, down);
        b.store(out, chosen);
    });
    let ptx = render(&k, 80);
    assert!(ptx.contains("cvt.s32.s8"), "{ptx}");
    assert!(ptx.contains("add.s32"), "{ptx}");
    assert!(ptx.contains("setp.gt.s32"), "{ptx}");
    assert!(ptx.contains("sub.s32"), "{ptx}");
    assert!(ptx.contains("cvt.s8.s32"), "{ptx}");
    assert!(!ptx.contains("add.s8"), "{ptx}");
    assert!(!ptx.contains("setp.gt.s8"), "{ptx}");
}

#[test]
fn byte_shuffles_use_the_b16_register_container() {
    for scalar in [Scalar::S8, Scalar::U8] {
        let k = kernel(|b| {
            let source = ptr(b, "in", scalar);
            let out = ptr(b, "out", scalar);
            let read = b.load(source, 1, false);
            let one = b.i32(1);
            let shuffled = b.shfl_down(read, one);
            b.store(out, shuffled);
        });
        let ptx = render(&k, 80);
        assert!(ptx.contains("shfl.sync.down.b32"), "{ptx}");
        assert!(ptx.contains("mov.b32 {%rs"), "{ptx}");
    }
}

/// `bf16` fma is a native Ampere op; add and mul need sm_90, so below it they
/// take the `f32` detour.
#[test]
fn bf16_fma_is_native_from_sm80_but_add_detours_below_sm90() {
    let op_kernel = |fused: bool| {
        kernel(move |b| {
            let source = ptr(b, "in", Scalar::BF16);
            let out = ptr(b, "out", Scalar::BF16);
            let read = b.load(source, 1, false);
            let x = b.bind_value_as("x", Ty::scalar(Scalar::BF16), read);
            let value = if fused {
                b.fma(x.clone(), x.clone(), x)
            } else {
                b.add(x.clone(), x)
            };
            b.store(out, value);
        })
    };
    let fma80 = render(&op_kernel(true), 80);
    assert!(fma80.contains("fma.rn.bf16"), "{fma80}");
    assert!(!fma80.contains("fma.rn.f32"), "{fma80}");

    let add80 = render(&op_kernel(false), 80);
    assert!(add80.contains("cvt.f32.bf16"), "{add80}");
    assert!(add80.contains("add.rn.f32"), "{add80}");
    let add90 = render(&op_kernel(false), 90);
    assert!(add90.contains("add.rn.bf16"), "{add90}");
}

#[test]
fn shared_arrays_declare_and_address() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let buf = b.shared_array(SharedArr {
            name: "buf".into(),
            elem: Scalar::F32,
            len: 32,
        });
        let tid = b.tid_x();
        let tid = b.cast(Scalar::S32, tid);
        let slot = b.offset(buf.clone(), tid);
        let one = b.float(Scalar::F32, 1.0);
        b.store(slot, one);
        b.barrier();
        let zero = b.i32(0);
        let first = b.offset(buf, zero);
        let read = b.load(first, 1, false);
        b.store(out, read);
    });
    let ptx = render(&k, 90);
    assert!(ptx.contains(".shared .align 16 .b8 shm_buf[128];"), "{ptx}");
    assert!(ptx.contains("st.shared.f32"), "{ptx}");
    assert!(ptx.contains("bar.sync 0;"), "{ptx}");
    assert!(ptx.contains("ld.shared.f32"), "{ptx}");
}

#[test]
fn constant_offsets_fold_into_addressing() {
    let k = kernel(|b| {
        let out = ptr(b, "out", Scalar::F32);
        let three = b.i32(3);
        let at = b.offset(out, three);
        let zero = b.float(Scalar::F32, 0.0);
        b.store(at, zero);
    });
    let ptx = render(&k, 90);
    assert!(ptx.contains("+12]"), "{ptx}");
    assert!(!ptx.contains("mad.lo.s64"), "{ptx}");
}

#[test]
fn maxnreg_directive_follows_maxntid_when_capped() {
    let trivial = |max_registers| {
        capped(max_registers, |b| {
            let out = ptr(b, "out", Scalar::F32);
            let zero = b.float(Scalar::F32, 0.0);
            b.store(out, zero);
        })
    };
    let ptx = render(&trivial(Some(128)), 90);
    assert!(ptx.contains(".maxntid 64, 1, 1\n.maxnreg 128\n"), "{ptx}");
    let ptx = render(&trivial(None), 90);
    assert!(!ptx.contains(".maxnreg"), "{ptx}");
}

/// Every operation remains in exactly one block, including expressions whose
/// results are re-evaluated at each use.
#[test]
fn every_operation_remains_in_a_block_when_steps_are_sliced() {
    let mut b = Builder::new();
    let out = ptr(&mut b, "out", Scalar::F32);
    let zero = b.i32(0);
    let unread = b.const_load(
        "unreachable",
        Scalar::F32,
        [1.0f32].into_iter().flat_map(f32::to_le_bytes).collect(),
        zero,
    );
    b.bind_value_as("dropped", Ty::scalar(Scalar::F32), unread);
    let kept = b.float(Scalar::F32, 0.0);
    b.store(out, kept);
    let f = b.into_function();
    let mut occurrences = vec![0; f.num_ops()];
    for block in f.live_blocks() {
        for &op in &f.block(block).ops {
            occurrences[op.index()] += 1;
        }
    }
    assert!(occurrences.into_iter().all(|count| count == 1));
}

#[test]
fn subword_kernel_parameters_use_their_abi_width() {
    for (scalar, parameter, load) in [
        (Scalar::S8, ".param .b8 x_param_0", "ld.param.s8"),
        (Scalar::U8, ".param .b8 x_param_0", "ld.param.u8"),
        (Scalar::S16, ".param .b16 x_param_0", "ld.param.s16"),
        (Scalar::U16, ".param .b16 x_param_0", "ld.param.u16"),
    ] {
        let k = kernel(|b| {
            let x = b.parameter(Param {
                name: "x".into(),
                ty: ParamTy::Scalar(scalar),
            });
            let _held = b.bind_value(Ty::scalar(scalar), x);
        });
        let ptx = render(&k, 90);
        assert!(ptx.contains(parameter), "{scalar:?}: {ptx}");
        assert!(ptx.contains(load), "{scalar:?}: {ptx}");
    }
}

#[test]
fn overflowing_integer_folds_and_pointer_offsets_do_not_panic() {
    let arithmetic = kernel(|b| {
        let out = ptr(b, "out", Scalar::S64);
        let max = b.i64(i64::MAX);
        let one = b.i64(1);
        let wrapped = b.add(max, one);
        b.store(out, wrapped);
    });
    let ptx = render(&arithmetic, 90);
    assert!(ptx.contains("add.s64"), "{ptx}");

    let address = kernel(|b| {
        let out = ptr(b, "out", Scalar::F64);
        let far = b.i64(i64::MAX);
        let at = b.offset(out, far);
        let one = b.float(Scalar::F64, 1.0);
        b.store(at, one);
    });
    let ptx = render(&address, 90);
    assert!(ptx.contains("st.global.f64"), "{ptx}");
}
