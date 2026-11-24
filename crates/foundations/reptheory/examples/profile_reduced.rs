// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use reptheory::{Irreps, O3Irrep};

fn irrep(l: u32, parity: i8) -> O3Irrep {
    O3Irrep::new(l, parity)
}

fn irreps(s: &str) -> Irreps<O3Irrep> {
    Irreps::parse(s).unwrap()
}

fn median_ms(reps: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..reps)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed().as_secs_f64() * 1e3
        })
        .collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

fn main() {
    let reps = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(7usize);

    let report = |name: &str, ms: f64| {
        println!("{name:<26} {ms:>8.2} ms/call (median of {reps})");
    };

    report(
        "Sym^3(1o)",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&irreps("1x1o"), 3, None);
        }),
    );
    report(
        "Sym^4(0e+1o)",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&irreps("1x0e+1x1o"), 4, None);
        }),
    );
    report(
        "Sym^5(1o)",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&irreps("1x1o"), 5, None);
        }),
    );
    report(
        "(2e)^3 ij=ji",
        median_ms(reps, || {
            let group = reptheory::perm::parse_formula("ijk=jik").unwrap().1;
            reptheory::reduced_tensor_product(&vec![irreps("1x2e"); 3], &group, None);
        }),
    );

    // The kept-irrep set is threaded into every subtree, so the fusion only
    // builds irreps that can still reach a kept output once the remaining
    // indices are fused back. Keeping only the scalar 0e is far cheaper than
    // the full fusion it would otherwise filter at the end.
    let keep = [irrep(0, 1)];
    report(
        "Sym^4(1o) keep 0e",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&irreps("1x1o"), 4, Some(&keep));
        }),
    );
    report(
        "Sym^4(1o) full",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&irreps("1x1o"), 4, None);
        }),
    );

    let three = irreps("1x0e+1x1o+1x2e");
    let symmetric4 = reptheory::SignedPermGroup::symmetric(4);
    let keep0e = [irrep(0, 1)];

    report(
        "Sym^4(0e+1o+2e) keep 0e  [projection]",
        median_ms(reps, || {
            reptheory::reduced_tensor_product(&vec![three.clone(); 4], &symmetric4, Some(&keep0e));
        }),
    );
    report(
        "Sym^4(0e+1o+2e) keep 0e  [multinomial]",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&three, 4, Some(&keep0e));
        }),
    );
    report(
        "Sym^4(0e+1o+2e) full     [multinomial]",
        median_ms(reps, || {
            reptheory::reduced_symmetric_tensor_product(&three, 4, None);
        }),
    );

    // The full (no keep) projection of Sym^4(0e+1o+2e) materializes the whole
    // 9⁴ Clebsch-Gordan fusion and runs into the minutes — the very case the
    // multinomial path exists to avoid. Pass `slow` to include it.
    if std::env::args().any(|a| a == "slow") {
        report(
            "Sym^4(0e+1o+2e) full     [projection]",
            median_ms(reps, || {
                reptheory::reduced_tensor_product(&vec![three.clone(); 4], &symmetric4, None);
            }),
        );
    }
}
