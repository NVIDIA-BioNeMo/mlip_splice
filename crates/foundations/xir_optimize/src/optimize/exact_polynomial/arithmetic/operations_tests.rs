// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use algebraic_real::AlgebraicReal;

type Polynomial = ExactPolynomial<u32>;

fn polynomial(terms: &[(Vec<u32>, Vec<AlgebraicReal>)]) -> Polynomial {
    Polynomial::from_terms(
        terms
            .iter()
            .cloned()
            .map(|(factors, coefficient)| ExactMonomial::new(factors, coefficient))
            .collect(),
        1024,
    )
    .unwrap()
}

#[test]
fn borrowed_arithmetic_preserves_exact_identities_and_inputs() {
    let root = AlgebraicReal::sqrt(2u32);
    let p = polynomial(&[
        (vec![0], vec![root.clone(), AlgebraicReal::integer(3)]),
        (vec![1, 1], vec![AlgebraicReal::integer(2), -&root]),
    ]);
    let q = polynomial(&[
        (vec![0], vec![-&root, AlgebraicReal::integer(4)]),
        (vec![], vec![AlgebraicReal::one(), root]),
    ]);
    let before = (p.clone(), q.clone());
    assert_eq!(
        p.add(&q, 1024, &mut |_| true),
        q.add(&p, 1024, &mut |_| true)
    );
    let root = AlgebraicReal::sqrt(2u32);
    let difference = polynomial(&[
        (vec![], vec![-&AlgebraicReal::one(), -&root]),
        (vec![0], vec![&root + &root, AlgebraicReal::integer(-1)]),
        (vec![1, 1], vec![AlgebraicReal::integer(2), -&root]),
    ]);
    assert_eq!(p.subtract(&q, 1024, &mut |_| true).unwrap(), difference);
    assert_eq!(p.subtract(&p, 1024, &mut |_| true).unwrap().len(), 0);
    assert_eq!(
        p.negate(&mut |_| true)
            .unwrap()
            .negate(&mut |_| true)
            .unwrap(),
        p
    );
    assert_eq!(
        p.multiply(&q, 1024, &mut |_| true),
        q.multiply(&p, 1024, &mut |_| true)
    );
    let sum = p.add(&q, 1024, &mut |_| true).unwrap();
    let square = p.multiply(&p, 1024, &mut |_| true).unwrap();
    let product = p.multiply(&q, 1024, &mut |_| true).unwrap();
    assert_eq!(
        p.multiply(&sum, 1024, &mut |_| true),
        square.add(&product, 1024, &mut |_| true)
    );
    assert_eq!((p, q), before);
}

#[test]
fn zero_is_symmetric_and_products_coalesce() {
    let p = polynomial(&[
        (vec![0], vec![AlgebraicReal::one()]),
        (vec![1], vec![AlgebraicReal::one()]),
    ]);
    let zero = Polynomial::from_terms(Vec::new(), 0).unwrap();
    assert_eq!(p.multiply(&zero, 0, &mut |_| true).unwrap(), zero);
    assert_eq!(zero.multiply(&p, 0, &mut |_| true).unwrap(), zero);
    assert_eq!(p.add(&zero, 1024, &mut |_| true).unwrap(), p);
    assert_eq!(
        zero.subtract(&p, 1024, &mut |_| true).unwrap(),
        p.negate(&mut |_| true).unwrap()
    );
    let square = p.multiply(&p, 1024, &mut |_| true).unwrap();
    let expected = polynomial(&[
        (vec![0, 0], vec![AlgebraicReal::one()]),
        (vec![0, 1], vec![AlgebraicReal::integer(2)]),
        (vec![1, 1], vec![AlgebraicReal::one()]),
    ]);
    assert_eq!(square, expected);
    assert!(p.multiply(&p, 3, &mut |_| true).is_none());
}

mod allocations {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static BYTES: Cell<Option<usize>> = const { Cell::new(None) };
    }

    struct Allocator;

    fn record(bytes: usize) {
        let _ = BYTES.try_with(|counter| {
            if let Some(total) = counter.get() {
                counter.set(Some(total.saturating_add(bytes)));
            }
        });
    }

    unsafe impl GlobalAlloc for Allocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
            record(size);
            unsafe { System.realloc(pointer, layout, size) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
    }

    #[global_allocator]
    static ALLOCATOR: Allocator = Allocator;

    pub(super) fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
        BYTES.with(|counter| counter.set(Some(0)));
        let output = f();
        let bytes = BYTES.with(|counter| counter.replace(None).unwrap());
        (output, bytes)
    }
}

#[test]
fn exhausted_work_allocates_no_result_or_operand_copies() {
    let p = polynomial(&[(vec![0, 1, 1], vec![AlgebraicReal::sqrt(2u32)])]);
    for (total, expression) in [(0, 100), (100, 0), (0, 0)] {
        for operation in 0..4 {
            let mut work = crate::optimize::AlgebraicWorkBudget::new(total, expression);
            let (result, bytes) = allocations::measure(|| {
                let mut attempt = work.begin_attempt();
                let mut charge = |cost| attempt.reserve(cost);
                match operation {
                    0 => p.add(&p, 1024, &mut charge),
                    1 => p.subtract(&p, 1024, &mut charge),
                    2 => p.multiply(&p, 1024, &mut charge),
                    _ => p.negate(&mut charge),
                }
            });
            assert!(result.is_none());
            assert_eq!(bytes, 0);
            assert_eq!(work.consumed, 0);
            assert_eq!(work.abandoned_count(), 1);
        }
    }
}

#[test]
fn coefficient_storage_is_reserved_before_allocating_it() {
    let p = polynomial(&[(vec![0], vec![AlgebraicReal::one(); 512])]);
    for (total, expression) in [(1, 1000), (1000, 1)] {
        let mut work = crate::optimize::AlgebraicWorkBudget::new(total, expression);
        let (result, bytes) = allocations::measure(|| {
            let mut attempt = work.begin_attempt();
            p.multiply(&p, 1024, &mut |cost| attempt.reserve(cost))
        });
        assert!(result.is_none());
        assert_eq!(bytes, std::mem::size_of::<ExactMonomial<u32>>());
        assert!(work.consumed <= total);
    }
}

#[test]
fn cancellation_does_not_copy_input_factors() {
    let p = Polynomial::from_terms(
        vec![ExactMonomial::new(
            vec![0; 8192],
            vec![AlgebraicReal::one()],
        )],
        1 << 20,
    )
    .unwrap();
    let (zero, bytes) = allocations::measure(|| p.subtract(&p, 1 << 20, &mut |_| true));
    assert_eq!(zero.unwrap().len(), 0);
    assert!(
        bytes < 4096,
        "cancellation copied a factor array: {bytes} bytes"
    );
    assert_eq!(p.terms()[0].factors().len(), 8192);
}

#[test]
fn multiplication_allocates_factors_only_for_the_result() {
    let p = Polynomial::from_terms(
        vec![ExactMonomial::new(
            vec![0; 8192],
            vec![AlgebraicReal::one()],
        )],
        1 << 20,
    )
    .unwrap();
    let (square, bytes) = allocations::measure(|| p.multiply(&p, 1 << 20, &mut |_| true));
    assert_eq!(square.unwrap().terms()[0].factors().len(), 16384);
    let factor_bytes = 16384 * std::mem::size_of::<u32>();
    assert!(
        bytes < factor_bytes + 4096,
        "multiplication copied input factors: {bytes} bytes"
    );
}
