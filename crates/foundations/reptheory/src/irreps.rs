// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::fmt;

use crate::rep::{CouplingIrrep, Irrep};

pub(crate) fn selection_rule_product<I: CouplingIrrep>(
    keep: &BTreeSet<I>,
    outside: &[I],
) -> BTreeSet<I> {
    let mut out = BTreeSet::new();
    for a in keep {
        for b in outside {
            for c in a.tensor_product(b) {
                out.insert(c);
            }
        }
    }
    out
}

pub(crate) fn selection_rule_power<I: CouplingIrrep>(
    trivial: &I,
    irs: &BTreeSet<I>,
    extra: usize,
) -> BTreeSet<I> {
    let outside: Vec<I> = irs.iter().cloned().collect();
    let mut out = BTreeSet::from([trivial.clone()]);
    for _ in 0..extra {
        out = selection_rule_product(&out, &outside);
    }
    out
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Irreps<I: CouplingIrrep> {
    pub entries: Vec<(usize, I)>,
}

impl<I: CouplingIrrep> Irreps<I> {
    pub fn new(entries: Vec<(usize, I)>) -> Self {
        Irreps { entries }
    }

    pub fn dim(&self) -> usize {
        self.entries
            .iter()
            .map(|(mul, ir)| mul * ir.dimension())
            .sum()
    }

    pub fn num_irreps(&self) -> usize {
        self.entries.iter().map(|(mul, _)| mul).sum()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (usize, I)> {
        self.entries.iter()
    }

    pub fn regroup(&self) -> Self {
        let mut sorted = self.entries.clone();
        sorted.retain(|(mul, _)| *mul > 0);
        sorted.sort_by(|(_, a), (_, b)| a.cmp(b));
        let mut merged: Vec<(usize, I)> = Vec::new();
        for (mul, ir) in sorted {
            match merged.last_mut() {
                Some((acc, last)) if *last == ir => *acc += mul,
                _ => merged.push((mul, ir)),
            }
        }
        Irreps { entries: merged }
    }
}

impl<I: Irrep> Irreps<I> {
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Irreps { entries: vec![] });
        }
        let mut entries = Vec::new();
        for term in s.split('+') {
            let term = term.trim();
            let (mul, ir_str) = match term.split_once('x') {
                Some((m, rest)) => (m.trim().parse::<usize>().ok()?, rest.trim()),
                None => (1, term),
            };
            entries.push((mul, I::parse(ir_str)?));
        }
        Some(Irreps { entries })
    }
}

impl<'a, I: CouplingIrrep> IntoIterator for &'a Irreps<I> {
    type Item = &'a (usize, I);
    type IntoIter = std::slice::Iter<'a, (usize, I)>;
    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

impl<I: CouplingIrrep + fmt::Display> fmt::Display for Irreps<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.entries.is_empty() {
            return write!(f, "");
        }
        let terms: Vec<String> = self
            .entries
            .iter()
            .map(|(mul, ir)| format!("{mul}x{ir}"))
            .collect();
        write!(f, "{}", terms.join("+"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::O3Irrep;

    #[test]
    fn parse_and_dim() {
        let irreps = Irreps::<O3Irrep>::parse("2x1o+1x2e").unwrap();
        assert_eq!(irreps.entries.len(), 2);
        assert_eq!(irreps.dim(), 2 * 3 + 5);
        assert_eq!(irreps.num_irreps(), 3);
    }

    #[test]
    fn parse_implicit_multiplicity() {
        let irreps = Irreps::<O3Irrep>::parse("1o+2e").unwrap();
        assert_eq!(
            irreps.entries,
            vec![(1, O3Irrep::new(1, -1)), (1, O3Irrep::new(2, 1))]
        );
    }

    #[test]
    fn regroup_sorts_and_merges() {
        let irreps = Irreps::<O3Irrep>::parse("1x2e+2x1o+1x2e")
            .unwrap()
            .regroup();
        assert_eq!(
            irreps.entries,
            vec![(2, O3Irrep::new(1, -1)), (2, O3Irrep::new(2, 1))]
        );
    }

    #[test]
    fn display_round_trips() {
        let irreps = Irreps::<O3Irrep>::parse("2x1o+1x2e").unwrap();
        assert_eq!(irreps.to_string(), "2x1o+1x2e");
    }
}
