// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! How the element blocks of a body are sequenced, and what a sequence looks
//! like before any of it reaches ptxas.
//!
//! The whole mission of this file is the element order: the two sequences, and
//! the deterministic summary of a sequence that a decision may read without
//! rendering PTX or assembling a cubin.

use gpu_ir::{Function, OpId, OpKind, ValueId};

use super::{
    BTreeMap, BTreeSet, LiftedProgram, Scalar, SelectedTerms, Term, Ty, ValTy, evaluation_profile,
};

/// How the element blocks of a body are sequenced.
///
/// A binding several elements share is live from the first of them to the
/// last, so the order of the blocks — not their contents — sets how many
/// shared bindings pile up at once. Which sequence is faster is a property of
/// the program, not of the geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ElementOrder {
    /// The order the elements appear in the trace.
    #[default]
    Trace,
    /// Fewest shared bindings left live: greedily take the element that births
    /// the shared bindings no earlier element reached and retires those it is
    /// the last to need.
    Liveness,
}

impl ElementOrder {
    /// The name the profile and Python report this order by.
    pub fn name(self) -> &'static str {
        match self {
            ElementOrder::Trace => "trace",
            ElementOrder::Liveness => "liveness",
        }
    }
}

impl std::str::FromStr for ElementOrder {
    type Err = String;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        [ElementOrder::Trace, ElementOrder::Liveness]
            .into_iter()
            .find(|order| order.name() == name)
            .ok_or_else(|| format!("loom element order must be trace or liveness, got {name:?}"))
    }
}

/// The bindings these places are, as indices into the flat binding list. A place
/// the list does not define is an input load and contributes no root.
pub(super) fn roots_of(
    read: &BTreeSet<ValueId>,
    index_of: &BTreeMap<ValueId, usize>,
) -> BTreeSet<usize> {
    read.iter()
        .filter_map(|value| index_of.get(value).copied())
        .collect()
}

/// The bindings a set of terms reads.
fn term_roots(
    f: &Function,
    terms: &[Term],
    index_of: &BTreeMap<ValueId, usize>,
) -> BTreeSet<usize> {
    let mut read = BTreeSet::new();
    for term in terms {
        term.add_reads(f, &mut read);
    }
    roots_of(&read, index_of)
}

/// Every output element that has terms, with the bindings its terms read.
pub(super) fn element_roots(
    f: &Function,
    terms: &SelectedTerms,
    index_of: &BTreeMap<ValueId, usize>,
) -> Vec<(usize, usize, BTreeSet<usize>)> {
    terms
        .iter()
        .enumerate()
        .flat_map(|(output, values)| {
            values
                .iter()
                .enumerate()
                .filter_map(move |(element, value)| Some((output, element, value.as_ref()?)))
        })
        .map(|(output, element, value)| (output, element, term_roots(f, value, index_of)))
        .collect()
}

/// Each element's full dependency closure over the flat binding list.
pub(super) fn element_closures(
    elements: &[(usize, usize, BTreeSet<usize>)],
    operands: &[Vec<usize>],
) -> Vec<Vec<usize>> {
    let mut seen = vec![usize::MAX; operands.len()];
    elements
        .iter()
        .enumerate()
        .map(|(mark, (_, _, roots))| {
            let mut closure = Vec::new();
            let mut stack = roots.iter().copied().collect::<Vec<_>>();
            while let Some(index) = stack.pop() {
                if seen[index] == mark {
                    continue;
                }
                seen[index] = mark;
                closure.push(index);
                stack.extend(operands[index].iter().copied());
            }
            closure
        })
        .collect()
}

/// Where each binding is defined, and which earlier bindings it reads.
pub(super) type BindingGraph = (BTreeMap<ValueId, usize>, Vec<Vec<usize>>);

pub(super) fn binding_operands(f: &Function, ops: &[OpId]) -> Result<BindingGraph, String> {
    let mut index_of = BTreeMap::new();
    let mut operands: Vec<Vec<usize>> = Vec::with_capacity(ops.len());
    for (index, &op) in ops.iter().enumerate() {
        if !matches!(f.op(op).kind, OpKind::Bind) {
            return Err("loom element ordering expects flat bindings".into());
        }
        let held = f.op(op);
        operands.push(
            f.reads(held.operands[0])
                .iter()
                .filter_map(|value| index_of.get(value).copied())
                .collect(),
        );
        index_of.insert(held.results[0], index);
    }
    Ok((index_of, operands))
}

/// Both orders described without emitting, rendering, or assembling either.
///
/// This runs the program's symbolic expansion once — far below the cost of a
/// single PTX render, and independent of cap, placement, and geometry — so a
/// decision made here can keep an order from ever reaching the compiler. The
/// expansion's inputs are parameters of its own arena rather than loads, so its
/// bindings are arithmetic only: no load or address class appears here.
pub(in super::super) fn program_order_summaries(
    loom: &LiftedProgram,
) -> Result<Vec<(ElementOrder, OrderSummary)>, String> {
    let (outputs, body) = evaluation_profile(loom)?;
    let f = body.function();
    let ops = body.steps().to_vec();
    let (index_of, operands) = binding_operands(f, &ops)?;
    let elements = outputs
        .iter()
        .enumerate()
        .flat_map(|(output, values)| {
            values
                .iter()
                .enumerate()
                .map(move |(element, &value)| (output, element, value))
        })
        .map(|(output, element, value)| (output, element, roots_of(&f.reads(value), &index_of)))
        .collect::<Vec<_>>();
    let closures = element_closures(&elements, &operands);
    let facts = binding_facts(f, &ops);
    Ok([ElementOrder::Trace, ElementOrder::Liveness]
        .into_iter()
        .map(|order| {
            let sequence = sequence(order, &closures, ops.len());
            (order, summarize(&sequence, &closures, &facts))
        })
        .collect())
}

/// The block sequence `order` prescribes for these elements. Trace order reads
/// no dependency, so its closures are never built.
pub(super) fn block_sequence(
    order: ElementOrder,
    elements: &[(usize, usize, BTreeSet<usize>)],
    operands: &[Vec<usize>],
    num_bindings: usize,
) -> Vec<usize> {
    match order {
        ElementOrder::Trace => (0..elements.len()).collect(),
        ElementOrder::Liveness => {
            liveness_order(&element_closures(elements, operands), num_bindings)
        }
    }
}

/// The block sequence `order` prescribes for these closures.
pub(super) fn sequence(
    order: ElementOrder,
    closures: &[Vec<usize>],
    num_bindings: usize,
) -> Vec<usize> {
    match order {
        ElementOrder::Trace => (0..closures.len()).collect(),
        ElementOrder::Liveness => liveness_order(closures, num_bindings),
    }
}

fn liveness_order(closures: &[Vec<usize>], num_bindings: usize) -> Vec<usize> {
    let mut pending = vec![0usize; num_bindings];
    for closure in closures {
        for &index in closure {
            pending[index] += 1;
        }
    }
    let mut emitted = vec![false; num_bindings];
    let mut taken = vec![false; closures.len()];
    let mut order = Vec::with_capacity(closures.len());
    for _ in 0..closures.len() {
        let next = (0..closures.len())
            .filter(|&element| !taken[element])
            .min_by_key(|&element| {
                let live = closures[element].iter().fold(0i64, |live, &index| {
                    match (emitted[index], pending[index] > 1) {
                        (true, false) => live - 1,
                        (false, true) => live + 1,
                        _ => live,
                    }
                });
                (live, element)
            })
            .expect("an element remains");
        for &index in &closures[next] {
            pending[index] -= 1;
            emitted[index] = true;
        }
        taken[next] = true;
        order.push(next);
    }
    order
}

/// What one binding costs to keep live and what kind of instruction defines it.
///
/// `slots` counts 32-bit registers; a predicate lives in the predicate file and
/// occupies none of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct BindingFacts {
    pub slots: u32,
    pub class: BindingClass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(in super::super) enum BindingClass {
    Load,
    Address,
    Float,
    Integer,
    Predicate,
    Move,
}

impl BindingClass {
    pub(in super::super) const ALL: [BindingClass; 6] = [
        BindingClass::Load,
        BindingClass::Address,
        BindingClass::Float,
        BindingClass::Integer,
        BindingClass::Predicate,
        BindingClass::Move,
    ];

    /// The name this class's population carries in an [`OrderSummary`].
    pub(in super::super) fn count_field(self) -> &'static str {
        match self {
            BindingClass::Load => "class_load",
            BindingClass::Address => "class_address",
            BindingClass::Float => "class_float",
            BindingClass::Integer => "class_integer",
            BindingClass::Predicate => "class_predicate",
            BindingClass::Move => "class_move",
        }
    }
}

fn register_slots(ty: Ty) -> u32 {
    if ty.scalar == Scalar::Pred {
        return 0;
    }
    let bytes = ty.scalar.size() as u32 * ty.lanes as u32;
    bytes.div_ceil(4)
}

/// Classifies a binding by the instruction its initializer heads. Loads and
/// address arithmetic are distinguished from math because they set different
/// hardware limits — memory latency to hide versus a register file to fill.
fn classify(f: &Function, ty: ValTy, init: ValueId) -> BindingClass {
    let head = f.operand_op(init).map(|op| &f.op(op).kind);
    match head {
        Some(OpKind::Load { .. }) => BindingClass::Load,
        Some(OpKind::Offset) => BindingClass::Address,
        Some(OpKind::Cmp(_) | OpKind::WarpAny) => BindingClass::Predicate,
        // A binding of a place, or of a constant, only moves it.
        None | Some(OpKind::IntConst { .. } | OpKind::FloatConst { .. } | OpKind::Sreg(_)) => {
            BindingClass::Move
        }
        _ => match ty {
            ValTy::Ptr(_) => BindingClass::Address,
            ValTy::Data(ty) if ty.scalar == Scalar::Pred => BindingClass::Predicate,
            ValTy::Data(ty) if ty.scalar.is_float() => BindingClass::Float,
            ValTy::Data(_) => BindingClass::Integer,
        },
    }
}

pub(in super::super) fn binding_facts(f: &Function, ops: &[OpId]) -> Vec<BindingFacts> {
    ops.iter()
        .map(|&op| match f.op(op).kind {
            OpKind::Bind => {
                let ty = f.ty(f.op(op).results[0]);
                BindingFacts {
                    slots: match ty {
                        // A pointer is two 32-bit registers.
                        ValTy::Ptr(_) => 2,
                        ValTy::Data(ty) => register_slots(ty),
                    },
                    class: classify(f, ty, f.op(op).operands[0]),
                }
            }
            _ => BindingFacts {
                slots: 0,
                class: BindingClass::Move,
            },
        })
        .collect()
}

/// Everything a decision may know about one element order before ptxas sees it.
///
/// The order-invariant fields (counts, class histogram) describe the program;
/// only the liveness and span fields move with the sequence. All of it is
/// computed in time linear in the closure edges, so summarizing both orders
/// costs far less than rendering either.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(in super::super) struct OrderSummary {
    pub elements: usize,
    pub bindings: usize,
    /// Bindings reached by more than one element closure.
    pub shared_bindings: usize,
    pub closure_edges: usize,
    /// Register slots live across the busiest element boundary.
    pub live_peak: u64,
    /// Summed live slots over every boundary — the ∫live dt of the sequence.
    pub live_area: u64,
    pub live_median: u64,
    pub live_p90: u64,
    /// Slots born and slots retired, summed over blocks.
    pub births: u64,
    pub deaths: u64,
    /// Shared bindings crossing the busiest boundary, and summed over them.
    pub crossings_peak: usize,
    pub crossings_total: usize,
    /// Longest and summed shared-binding span, in element blocks.
    pub span_max: usize,
    pub span_total: usize,
    /// Largest emitted block, in bindings and in slots.
    pub block_bindings_max: usize,
    pub block_slots_max: u64,
    pub class_counts: [usize; BindingClass::ALL.len()],
}

impl OrderSummary {
    /// The summary as named quantities, in a fixed order — the feature vector
    /// a ranker reads and the dataset stores.
    pub(in super::super) fn fields(&self) -> Vec<(&'static str, u64)> {
        let mut fields = vec![
            ("elements", self.elements as u64),
            ("bindings", self.bindings as u64),
            ("shared_bindings", self.shared_bindings as u64),
            ("closure_edges", self.closure_edges as u64),
            ("live_peak", self.live_peak),
            ("live_area", self.live_area),
            ("live_median", self.live_median),
            ("live_p90", self.live_p90),
            ("births", self.births),
            ("deaths", self.deaths),
            ("crossings_peak", self.crossings_peak as u64),
            ("crossings_total", self.crossings_total as u64),
            ("span_max", self.span_max as u64),
            ("span_total", self.span_total as u64),
            ("block_bindings_max", self.block_bindings_max as u64),
            ("block_slots_max", self.block_slots_max),
        ];
        for class in BindingClass::ALL {
            fields.push((
                class.count_field(),
                self.class_counts[class as usize] as u64,
            ));
        }
        fields
    }
}

/// The pre-ptxas description of running `sequence` over `closures`.
///
/// A binding is emitted by the first block whose closure reaches it and stays
/// live until the last block that needs it, which is exactly the lifetime the
/// emitted body gives it.
pub(in super::super) fn summarize(
    sequence: &[usize],
    closures: &[Vec<usize>],
    facts: &[BindingFacts],
) -> OrderSummary {
    let blocks = sequence.len();
    let num_bindings = facts.len();
    let mut birth = vec![usize::MAX; num_bindings];
    let mut death = vec![0usize; num_bindings];
    let mut uses = vec![0usize; num_bindings];
    let mut closure_edges = 0;
    for (step, &element) in sequence.iter().enumerate() {
        for &index in &closures[element] {
            closure_edges += 1;
            uses[index] += 1;
            birth[index] = birth[index].min(step);
            death[index] = death[index].max(step);
        }
    }

    let mut summary = OrderSummary {
        elements: blocks,
        bindings: num_bindings,
        closure_edges,
        ..OrderSummary::default()
    };
    for facts in facts {
        summary.class_counts[facts.class as usize] += 1;
    }

    // Live slots at each boundary, as a difference series: a binding
    // contributes from the block that emits it to the block that last needs it.
    let mut slot_delta = vec![0i64; blocks + 1];
    let mut shared_delta = vec![0i64; blocks + 1];
    let mut block_bindings = vec![0usize; blocks];
    let mut block_slots = vec![0u64; blocks];
    for index in 0..num_bindings {
        if birth[index] == usize::MAX {
            continue;
        }
        let (birth, death) = (birth[index], death[index]);
        let slots = facts[index].slots as i64;
        slot_delta[birth] += slots;
        slot_delta[death + 1] -= slots;
        summary.births += slots as u64;
        summary.deaths += slots as u64;
        block_bindings[birth] += 1;
        block_slots[birth] += slots as u64;
        if uses[index] > 1 {
            summary.shared_bindings += 1;
            summary.span_total += death - birth;
            summary.span_max = summary.span_max.max(death - birth);
            shared_delta[birth] += 1;
            shared_delta[death + 1] -= 1;
        }
    }
    summary.block_bindings_max = block_bindings.into_iter().max().unwrap_or(0);
    summary.block_slots_max = block_slots.into_iter().max().unwrap_or(0);

    let mut live = Vec::with_capacity(blocks);
    let mut running = 0i64;
    let mut shared_running = 0i64;
    for step in 0..blocks {
        running += slot_delta[step];
        shared_running += shared_delta[step];
        live.push(running as u64);
        summary.live_area += running as u64;
        summary.live_peak = summary.live_peak.max(running as u64);
        summary.crossings_total += shared_running as usize;
        summary.crossings_peak = summary.crossings_peak.max(shared_running as usize);
    }
    live.sort_unstable();
    summary.live_median = quantile(&live, 50);
    summary.live_p90 = quantile(&live, 90);
    summary
}

/// The `percent` quantile of an ascending series, by nearest rank.
fn quantile(ascending: &[u64], percent: usize) -> u64 {
    if ascending.is_empty() {
        return 0;
    }
    let rank = (ascending.len() * percent).div_ceil(100).max(1);
    ascending[rank - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(slots: &[u32]) -> Vec<BindingFacts> {
        slots
            .iter()
            .map(|&slots| BindingFacts {
                slots,
                class: BindingClass::Float,
            })
            .collect()
    }

    /// Two elements share binding 0; each also owns a private one.
    fn shared_pair() -> Vec<Vec<usize>> {
        vec![vec![0, 1], vec![0, 2]]
    }

    #[test]
    fn a_shared_binding_lives_across_the_boundary_it_spans() {
        let summary = summarize(&[0, 1], &shared_pair(), &facts(&[1, 1, 1]));

        assert_eq!(summary.shared_bindings, 1);
        assert_eq!(summary.span_max, 1);
        assert_eq!(summary.crossings_peak, 1);
        // Block 0 emits all of {0, 1}; only binding 0 survives into block 1,
        // which adds binding 2.
        assert_eq!(summary.live_peak, 2);
        assert_eq!(summary.live_area, 4);
    }

    #[test]
    fn slot_weights_follow_the_scalar_width() {
        assert_eq!(register_slots(Ty::scalar(Scalar::F64)), 2);
        assert_eq!(register_slots(Ty::scalar(Scalar::F32)), 1);
        assert_eq!(register_slots(Ty::scalar(Scalar::F16)), 1);
        assert_eq!(register_slots(Ty::vec(Scalar::F32, 4)), 4);
        assert_eq!(register_slots(Ty::vec(Scalar::F16, 2)), 1);
        assert_eq!(register_slots(Ty::scalar(Scalar::Pred)), 0);
    }

    #[test]
    fn order_invariant_facts_do_not_move_with_the_sequence() {
        let closures = vec![vec![0, 1], vec![0, 2], vec![3]];
        let facts = facts(&[1, 2, 1, 1]);
        let forward = summarize(&[0, 1, 2], &closures, &facts);
        let backward = summarize(&[2, 1, 0], &closures, &facts);

        assert_eq!(forward.elements, backward.elements);
        assert_eq!(forward.bindings, backward.bindings);
        assert_eq!(forward.shared_bindings, backward.shared_bindings);
        assert_eq!(forward.closure_edges, backward.closure_edges);
        assert_eq!(forward.class_counts, backward.class_counts);
        assert_eq!(forward.births, backward.births);
    }

    #[test]
    fn separating_the_sharers_widens_the_span_it_pays_for() {
        // Elements 0 and 2 share binding 0; element 1 is unrelated.
        let closures = vec![vec![0, 1], vec![2], vec![0, 3]];
        let facts = facts(&[1, 1, 1, 1]);

        let adjacent = summarize(&[0, 2, 1], &closures, &facts);
        let separated = summarize(&[0, 1, 2], &closures, &facts);

        assert_eq!(adjacent.span_max, 1);
        assert_eq!(separated.span_max, 2);
        assert!(separated.live_area > adjacent.live_area);
    }

    #[test]
    fn liveness_sequences_the_sharers_together() {
        // Elements 0 and 2 share bindings; trace order separates them.
        let closures = vec![vec![0, 1], vec![2], vec![0, 1, 3]];
        let order = sequence(ElementOrder::Liveness, &closures, 4);

        let position = |element: usize| order.iter().position(|&e| e == element).unwrap();
        assert_eq!(position(0).abs_diff(position(2)), 1);
    }

    #[test]
    fn both_orders_are_deterministic() {
        let closures = vec![vec![0, 1], vec![0, 2], vec![1, 2, 3], vec![3]];
        for order in [ElementOrder::Trace, ElementOrder::Liveness] {
            let first = sequence(order, &closures, 4);
            let second = sequence(order, &closures, 4);
            assert_eq!(first, second, "{order:?}");
            let mut sorted = first.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..closures.len()).collect::<Vec<_>>());
        }
    }

    #[test]
    fn quantiles_take_the_nearest_rank() {
        let series = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(quantile(&series, 50), 5);
        assert_eq!(quantile(&series, 90), 9);
        assert_eq!(quantile(&series, 100), 10);
        assert_eq!(quantile(&[], 50), 0);
    }

    #[test]
    fn classification_names_the_defining_instruction() {
        let f32 = ValTy::Data(Ty::scalar(Scalar::F32));
        let pred = ValTy::Data(Ty::scalar(Scalar::Pred));
        let i32 = ValTy::Data(Ty::scalar(Scalar::S32));
        let mut b = gpu_ir::Builder::new();
        let pointer = b.parameter(gpu_ir::Param {
            name: "p".into(),
            ty: gpu_ir::ParamTy::Ptr(Scalar::F32),
        });
        let a = b.parameter(gpu_ir::Param {
            name: "a".into(),
            ty: gpu_ir::ParamTy::Scalar(Scalar::F32),
        });
        let other = b.parameter(gpu_ir::Param {
            name: "b".into(),
            ty: gpu_ir::ParamTy::Scalar(Scalar::F32),
        });
        let loaded = b.load(pointer.clone(), 1, true);
        let one = b.i32(1);
        let address = b.offset(pointer, one);
        let sum = b.add(a.clone(), other.clone());
        let below = b.lt(a.clone(), other);
        let (loaded, address, sum, below, moved) = {
            let f = b.function();
            (
                loaded.id_in(f),
                address.id_in(f),
                sum.id_in(f),
                below.id_in(f),
                a.id_in(f),
            )
        };
        let f = b.function();
        assert_eq!(classify(f, f32, loaded), BindingClass::Load);
        assert_eq!(classify(f, f32, address), BindingClass::Address);
        assert_eq!(classify(f, f32, sum), BindingClass::Float);
        assert_eq!(classify(f, i32, sum), BindingClass::Integer);
        assert_eq!(classify(f, f32, moved), BindingClass::Move);
        assert_eq!(classify(f, pred, below), BindingClass::Predicate);
    }
}
