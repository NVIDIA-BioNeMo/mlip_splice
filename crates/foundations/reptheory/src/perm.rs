// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct SignedPerm {
    pub sign: i8,
    pub perm: Vec<usize>,
}

impl SignedPerm {
    pub fn identity(n: usize) -> Self {
        SignedPerm {
            sign: 1,
            perm: (0..n).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.perm.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    pub fn compose(&self, other: &SignedPerm) -> SignedPerm {
        SignedPerm {
            sign: self.sign * other.sign,
            perm: other.perm.iter().map(|&i| self.perm[i]).collect(),
        }
    }

    pub fn inverse(&self) -> SignedPerm {
        let mut inv = vec![0usize; self.perm.len()];
        for (i, &pi) in self.perm.iter().enumerate() {
            inv[pi] = i;
        }
        SignedPerm {
            sign: self.sign,
            perm: inv,
        }
    }
}

pub fn permutation_parity(perm: &[usize]) -> i8 {
    let n = perm.len();
    let mut seen = vec![false; n];
    let mut sign: i8 = 1;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut len = 0usize;
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            i = perm[i];
            len += 1;
        }
        if len.is_multiple_of(2) {
            sign = -sign;
        }
    }
    sign
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SignedPermGroup {
    pub degree: usize,
    pub elements: Vec<SignedPerm>,
    pub generators: Vec<SignedPerm>,
}

impl SignedPermGroup {
    pub fn generate(
        degree: usize,
        generators: impl IntoIterator<Item = SignedPerm>,
    ) -> SignedPermGroup {
        let identity = SignedPerm::identity(degree);
        let mut generators_unique: Vec<SignedPerm> = Vec::new();
        let mut seen: BTreeSet<SignedPerm> = BTreeSet::new();
        for g in generators {
            assert_eq!(g.len(), degree, "generator acts on the wrong index count");
            if g != identity && seen.insert(g.clone()) {
                generators_unique.push(g);
            }
        }

        let mut set: BTreeSet<SignedPerm> = BTreeSet::new();
        set.insert(identity);
        set.extend(generators_unique.iter().cloned());
        loop {
            let n = set.len();
            let snapshot: Vec<SignedPerm> = set.iter().cloned().collect();
            for a in &snapshot {
                set.insert(a.inverse());
                for b in &snapshot {
                    set.insert(a.compose(b));
                }
            }
            if set.len() == n {
                break;
            }
        }
        SignedPermGroup {
            degree,
            elements: set.into_iter().collect(),
            generators: generators_unique,
        }
    }

    pub fn symmetric(n: usize) -> SignedPermGroup {
        SignedPermGroup::generate(n, adjacent_transpositions(n, 1))
    }

    pub fn antisymmetric(n: usize) -> SignedPermGroup {
        SignedPermGroup::generate(
            n,
            adjacent_transpositions(n, -1)
                .into_iter()
                .map(|sp| SignedPerm {
                    sign: permutation_parity(&sp.perm),
                    perm: sp.perm,
                }),
        )
    }

    pub fn order(&self) -> usize {
        self.elements.len()
    }
}

fn adjacent_transpositions(n: usize, sign: i8) -> Vec<SignedPerm> {
    (0..n.saturating_sub(1))
        .map(|k| {
            let mut perm: Vec<usize> = (0..n).collect();
            perm.swap(k, k + 1);
            SignedPerm { sign, perm }
        })
        .collect()
}

pub fn parse_formula(formula: &str) -> Result<(Vec<char>, SignedPermGroup), String> {
    let mut terms = formula.split('=');
    let first = terms.next().ok_or("empty formula")?;
    if first.starts_with('-') {
        return Err("the first formula term must be unsigned".into());
    }
    let f0: Vec<char> = first.chars().collect();
    if f0.is_empty() {
        return Err("formula has no indices".into());
    }
    let canonical: BTreeSet<char> = f0.iter().copied().collect();
    if canonical.len() != f0.len() {
        return Err(format!("repeated index in '{first}'"));
    }

    let mut generators = vec![SignedPerm::identity(f0.len())];
    for term in std::iter::once(first).chain(terms) {
        let (sign, body) = match term.strip_prefix('-') {
            Some(rest) => (-1i8, rest),
            None => (1i8, term),
        };
        let labels: Vec<char> = body.chars().collect();
        if labels.iter().copied().collect::<BTreeSet<_>>() != canonical {
            return Err(format!("'{body}' is not a permutation of '{first}'"));
        }
        // perm[k] is the position of the canonical label f0[k] in this term.
        let perm: Vec<usize> = f0
            .iter()
            .map(|c| labels.iter().position(|d| d == c).unwrap())
            .collect();
        generators.push(SignedPerm { sign, perm });
    }

    Ok((f0, SignedPermGroup::generate(canonical.len(), generators)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perm(sign: i8, perm: &[usize]) -> SignedPerm {
        SignedPerm {
            sign,
            perm: perm.to_vec(),
        }
    }

    #[test]
    fn parity_of_transposition_and_cycle() {
        assert_eq!(permutation_parity(&[0, 1, 2]), 1);
        assert_eq!(permutation_parity(&[1, 0, 2]), -1);
        assert_eq!(permutation_parity(&[1, 2, 0]), 1);
        assert_eq!(permutation_parity(&[1, 0, 3, 2]), 1);
    }

    #[test]
    fn compose_and_inverse() {
        let a = perm(1, &[1, 2, 0]);
        let inv = a.inverse();
        assert_eq!(a.compose(&inv), SignedPerm::identity(3));
        assert_eq!(inv.compose(&a), SignedPerm::identity(3));
        let s = perm(-1, &[1, 0, 2]);
        assert_eq!(s.compose(&s), SignedPerm::identity(3));
    }

    #[test]
    fn symmetric_group_orders() {
        assert_eq!(SignedPermGroup::symmetric(1).order(), 1);
        assert_eq!(SignedPermGroup::symmetric(2).order(), 2);
        assert_eq!(SignedPermGroup::symmetric(3).order(), 6);
        assert_eq!(SignedPermGroup::symmetric(4).order(), 24);
        assert!(
            SignedPermGroup::symmetric(4)
                .elements
                .iter()
                .all(|e| e.sign == 1)
        );
    }

    #[test]
    fn antisymmetric_signs_match_parity() {
        let g = SignedPermGroup::antisymmetric(4);
        assert_eq!(g.order(), 24);
        for e in &g.elements {
            assert_eq!(e.sign, permutation_parity(&e.perm));
        }
    }

    #[test]
    fn formula_symmetric_pair() {
        let (f0, g) = parse_formula("ij=ji").unwrap();
        assert_eq!(f0, vec!['i', 'j']);
        assert_eq!(g.order(), 2);
        assert!(g.elements.iter().all(|e| e.sign == 1));
    }

    #[test]
    fn formula_antisymmetric_pair() {
        let (_f0, g) = parse_formula("ij=-ji").unwrap();
        assert_eq!(g.order(), 2);
        let swap = g.elements.iter().find(|e| e.perm == vec![1, 0]).unwrap();
        assert_eq!(swap.sign, -1);
    }

    #[test]
    fn formula_generates_full_symmetric_from_two_swaps() {
        let (_f0, g) = parse_formula("ijk=jik=ikj").unwrap();
        assert_eq!(g.order(), 6);
        assert!(g.elements.iter().all(|e| e.sign == 1));
    }

    #[test]
    fn formula_mixed_sign_closure() {
        let (_f0, g) = parse_formula("ijk=-jik").unwrap();
        let mut perms = std::collections::BTreeSet::new();
        for e in &g.elements {
            assert!(perms.insert(e.perm.clone()), "a permutation got two signs");
        }
    }

    #[test]
    fn rejects_bad_formulas() {
        assert!(parse_formula("-ij=ji").is_err());
        assert!(parse_formula("ij=jk").is_err());
        assert!(parse_formula("ii=ii").is_err());
    }
}
