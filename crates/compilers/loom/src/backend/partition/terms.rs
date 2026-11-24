// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, Function, OpKind, Value, ValueId};

use super::{BTreeMap, BTreeSet, BinOp, Bindings, GuardKey, GuardProvenance};

/// One additive term of an output element: values the evaluation defines, with
/// the distribution context applied around them.
///
/// A term is not a value the evaluation holds — enumeration distributes products
/// over sums, so most terms are values the kernel that selects them will define.
/// Keeping a term as what to build rather than as an arena value is what lets
/// the planner enumerate terms over an evaluation it only reads, and what lets
/// the same term be built into whichever kernel arena selects it.
#[derive(Clone, Debug)]
pub(in super::super) enum Term {
    /// A value the evaluation defines.
    Held(ValueId),
    Negated(Box<Term>),
    Product(Box<Term>, Box<Term>),
    /// `pred ? taken : zero`, the guarded form a select distributes through.
    Guarded {
        pred: ValueId,
        taken: Box<Term>,
        zero: ValueId,
    },
}

impl Term {
    fn held(value: ValueId) -> Term {
        Term::Held(value)
    }

    /// The places this term reads.
    pub(in super::super) fn add_reads(&self, f: &Function, into: &mut BTreeSet<ValueId>) {
        match self {
            Term::Held(value) => f.add_reads(*value, into),
            Term::Negated(inner) => inner.add_reads(f, into),
            Term::Product(left, right) => {
                left.add_reads(f, into);
                right.add_reads(f, into);
            }
            Term::Guarded { pred, taken, zero } => {
                f.add_reads(*pred, into);
                taken.add_reads(f, into);
                f.add_reads(*zero, into);
            }
        }
    }

    pub(in super::super) fn reads(&self, f: &Function) -> BTreeSet<ValueId> {
        let mut reads = BTreeSet::new();
        self.add_reads(f, &mut reads);
        reads
    }

    /// Builds this term in `b`, whose arena is the one `held` names.
    pub(in super::super) fn build(&self, b: &mut Builder) -> Value {
        match self {
            Term::Held(value) => b.value_of(*value),
            Term::Negated(inner) => {
                let inner = inner.build(b);
                b.neg(inner)
            }
            Term::Product(left, right) => {
                let left = left.build(b);
                let right = right.build(b);
                b.mul(left, right)
            }
            Term::Guarded { pred, taken, zero } => {
                let pred = b.value_of(*pred);
                let taken = taken.build(b);
                let zero = b.value_of(*zero);
                b.select(pred, taken, zero)
            }
        }
    }
}

/// A contiguous run of enumerated terms that folds back to one unexpanded term:
/// term enumeration emits every additive subtree's monomials in DFS order, so
/// each subtree is a range, and its recorded term is the subtree as written —
/// factored — with the distribution context applied around it rather than
/// through it.
pub(super) type TermSpan = (usize, usize, Term);

pub(super) fn additive_terms(f: &Function, value: ValueId, bindings: &Bindings) -> Vec<Term> {
    additive_term_spans(f, value, bindings).0
}

pub(super) fn additive_term_spans(
    f: &Function,
    value: ValueId,
    bindings: &Bindings,
) -> (Vec<Term>, Vec<TermSpan>) {
    fn spanned(
        term: Term,
        terms: Vec<Term>,
        mut spans: Vec<TermSpan>,
    ) -> (Vec<Term>, Vec<TermSpan>) {
        if terms.len() > 1 {
            spans.push((0, terms.len(), term));
        }
        (terms, spans)
    }

    fn chain(
        mut lhs: (Vec<Term>, Vec<TermSpan>),
        rhs: (Vec<Term>, Vec<TermSpan>),
        map: impl Fn(Term) -> Term,
    ) -> (Vec<Term>, Vec<TermSpan>) {
        let shift = lhs.0.len();
        lhs.0.extend(rhs.0.into_iter().map(&map));
        lhs.1.extend(
            rhs.1
                .into_iter()
                .map(|(start, end, term)| (start + shift, end + shift, map(term))),
        );
        lhs
    }

    fn negated(term: Term) -> Term {
        Term::Negated(Box::new(term))
    }

    fn visit(f: &Function, value: ValueId, bindings: &Bindings) -> (Vec<Term>, Vec<TermSpan>) {
        let expanded = bindings.expanded(value);
        let Some(op) = f.operand_op(expanded) else {
            return (vec![Term::held(value)], Vec::new());
        };
        let operands = f.op(op).operands.clone();
        match f.op(op).kind {
            OpKind::Bin(BinOp::Add) => {
                let lhs = visit(f, operands[0], bindings);
                let rhs = visit(f, operands[1], bindings);
                let (terms, spans) = chain(lhs, rhs, |term| term);
                spanned(Term::held(value), terms, spans)
            }
            OpKind::Bin(BinOp::Sub) => {
                let lhs = visit(f, operands[0], bindings);
                let rhs = visit(f, operands[1], bindings);
                let (terms, spans) = chain(lhs, rhs, negated);
                spanned(Term::held(value), terms, spans)
            }
            OpKind::Un(gpu_ir::UnOp::Neg) => {
                let inner = visit(f, operands[0], bindings);
                let (terms, spans) = chain((Vec::new(), Vec::new()), inner, negated);
                spanned(Term::held(value), terms, spans)
            }
            OpKind::Bin(BinOp::Mul) => {
                visit_product(f, operands[0], operands[1], Term::held(value), bindings)
            }
            OpKind::Fma => {
                // The product is enumerated as though it were written out, so a
                // contracted term distributes exactly as its uncontracted form
                // would have.
                let whole = Term::Product(
                    Box::new(Term::held(operands[0])),
                    Box::new(Term::held(operands[1])),
                );
                let product = visit_product(f, operands[0], operands[1], whole, bindings);
                let rest = visit(f, operands[2], bindings);
                let (terms, spans) = chain(product, rest, |term| term);
                spanned(Term::held(value), terms, spans)
            }
            OpKind::Select if is_zero(f, operands[2]) => {
                let yes = visit(f, operands[1], bindings);
                if yes.0.len() > 1 {
                    let (pred, zero) = (operands[0], operands[2]);
                    let (terms, spans) =
                        chain((Vec::new(), Vec::new()), yes, |term| Term::Guarded {
                            pred,
                            taken: Box::new(term),
                            zero,
                        });
                    spanned(Term::held(value), terms, spans)
                } else {
                    (vec![Term::held(value)], Vec::new())
                }
            }
            _ => (vec![Term::held(value)], Vec::new()),
        }
    }

    /// A product, enumerated: a zero factor collapses it, and a factor that is
    /// itself a sum distributes over the other. `whole` is the term the product
    /// folds back to where neither happens.
    fn visit_product(
        f: &Function,
        left: ValueId,
        right: ValueId,
        whole: Term,
        bindings: &Bindings,
    ) -> (Vec<Term>, Vec<TermSpan>) {
        if is_zero(f, left) {
            return (vec![Term::held(left)], Vec::new());
        }
        if is_zero(f, right) {
            return (vec![Term::held(right)], Vec::new());
        }
        let lhs = visit(f, left, bindings);
        if lhs.0.len() > 1 {
            let (terms, spans) = chain((Vec::new(), Vec::new()), lhs, |term| {
                Term::Product(Box::new(term), Box::new(Term::held(right)))
            });
            return spanned(whole, terms, spans);
        }
        let rhs = visit(f, right, bindings);
        if rhs.0.len() > 1 {
            let (terms, spans) = chain((Vec::new(), Vec::new()), rhs, |term| {
                Term::Product(Box::new(Term::held(left)), Box::new(term))
            });
            spanned(whole, terms, spans)
        } else {
            (vec![whole], Vec::new())
        }
    }

    fn is_zero(f: &Function, value: ValueId) -> bool {
        let Some(op) = f.operand_op(value) else {
            return false;
        };
        match f.op(op).kind {
            OpKind::IntConst { value, .. } => value == 0,
            OpKind::FloatConst { value, .. } => value == 0.0,
            OpKind::Splat(_) => is_zero(f, f.op(op).operands[0]),
            _ => false,
        }
    }

    visit(f, value, bindings)
}

/// Folds a selected term subset through the largest complete additive subtrees
/// it covers: spans are taken widest-first when every term they hold is selected
/// and still uncovered, and the leftovers fall back to the enumerated
/// (distributed) terms. Pieces come back in term order. `None` when an index is
/// out of range.
pub(super) fn cover_terms(
    indices: &[usize],
    terms: &[Term],
    spans: &[TermSpan],
) -> Option<Vec<Term>> {
    let mut selected = vec![false; terms.len()];
    for &index in indices {
        *selected.get_mut(index)? = true;
    }
    let mut covered = vec![false; terms.len()];
    let mut widest = spans.iter().collect::<Vec<_>>();
    widest.sort_by_key(|(start, end, _)| (std::cmp::Reverse(end - start), *start));
    let mut pieces = Vec::new();
    for (start, end, term) in widest {
        if (*start..*end).all(|term| selected[term] && !covered[term]) {
            covered[*start..*end].fill(true);
            pieces.push((*start, term.clone()));
        }
    }
    for &index in indices {
        if !covered[index] {
            pieces.push((index, terms[index].clone()));
        }
    }
    pieces.sort_by_key(|(start, _)| *start);
    Some(pieces.into_iter().map(|(_, term)| term).collect())
}

/// The guard every place a term reads shares, if they share one.
///
/// The key and nothing else: a part travels between plans, and a key names the
/// guard in the program rather than in one evaluation's arena.
pub(super) fn term_guard(
    f: &Function,
    term: &Term,
    statement_guards: &BTreeMap<ValueId, GuardProvenance>,
) -> Option<GuardKey> {
    let mut guards = term
        .reads(f)
        .into_iter()
        .filter_map(|value| statement_guards.get(&value));
    let first = guards.next()?.key;
    guards.all(|guard| guard.key == first).then_some(first)
}

pub(super) fn additive_term_guards(
    f: &Function,
    terms: &[Term],
    spans: &[TermSpan],
    statement_guards: &BTreeMap<ValueId, GuardProvenance>,
) -> Vec<Option<GuardKey>> {
    let mut guards = terms
        .iter()
        .map(|term| term_guard(f, term, statement_guards))
        .collect::<Vec<_>>();
    for (start, end, term) in spans {
        let Some(span_guard) = term_guard(f, term, statement_guards) else {
            continue;
        };
        for guard in &mut guards[*start..*end] {
            if guard.is_none() {
                *guard = Some(span_guard);
            }
        }
    }
    guards
}
