// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An order and a materialization set for the scalar DAG an [`IndexedTensor`]
//! denotes.
//!
//! `emit_indexed` expands an indexed tensor into operand values, which have no
//! storage: a value reached twice is evaluated twice. Binding a value is the
//! only way to share its computation, and a bound value occupies a register
//! from its definition to its last read. So the emission choice is a pebble game
//! on this DAG — hold a value or rebuild it — under the register budget.
//!
//! Naming is *not* what makes a fused multiply-add possible. A product written
//! inline is read exactly once by construction, so contraction absorbs it just
//! as it absorbs a named single-use product. Nothing here needs to reason about
//! fusion, and no value is named to protect one.
//!
//! So the policy is the plain trade. Everything shared is named, which is the
//! least work. Pressure is then relieved exactly where it occurs: the order is
//! walked once, and at any step holding more values than the budget allows the
//! live value cheapest to rebuild is given back. A value the budget never
//! squeezes is never given back, so nothing is recomputed to satisfy a global
//! average.
//!
//! Deciding at the congested step is what keeps the choice honest. A score
//! computed once over the whole program goes stale the moment a value is given
//! back — every later decision then ranks values by a liveness that no longer
//! exists. Here the live set *is* the state the walk carries, so every decision
//! reads the liveness it is actually deciding under.
//!
//! Memory reads are never rebuilt: keeping one value live is cheaper than
//! repeating a load and its address arithmetic at every use. Values whose
//! operands have already died are also kept, as are values that read something
//! already given back. When no safe candidate remains, a step over budget stays
//! over budget: the walk reaches a fixed point rather than a hard bound.
//!
//! Refusing compounded rebuilds is also what makes the pass linear. A value read
//! once has exactly one consumer, so it belongs to the private cone of exactly
//! one named value; those cones are disjoint, every candidate is examined at
//! most once, and the examinations together are one pass over the DAG instead of
//! one per decision. With the heap that ranks candidates, `O((V + E) log V)` for
//! `V` values and `E` edges. That bound is the point: the DAGs that reach here
//! are the large ones, because that is what recovery leaves behind.

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::sync::Arc;

use super::{IndexedExpression, IndexedTensor};

/// Identity of one scalar value: the defining expression node and its element.
/// Re-indexing nodes are resolved away before a key is formed, so every path
/// that reaches a value produces the same key and the DAG is hash-consed.
pub(super) type ValueKey = (usize, usize);

/// A value that is worth naming. Constants and dense immediates are excluded:
/// they encode directly into a consumer's operand, so holding one in a register
/// is pure loss and recomputing it is free.
///
/// `operands` is deduplicated: a value read twice by one consumer occupies one
/// register and is retired once, so the dependency and the pressure rules both
/// want it listed once. `uses` counts every read, which is what decides whether
/// naming the value pays for itself.
#[derive(Clone)]
pub(super) struct ScalarNode {
    pub tensor: IndexedTensor,
    pub element: usize,
    pub operands: Vec<usize>,
    pub uses: usize,
}

pub(super) struct ScalarDag {
    pub nodes: Vec<ScalarNode>,
    /// Emission order: a permutation of node indices chosen to keep the number
    /// of simultaneously live values low.
    pub order: Vec<usize>,
    /// Nodes left unnamed, to be recomputed at each use.
    pub inlined: Vec<bool>,
}

#[cfg(test)]
thread_local! {
    /// Values stepped over by the two walks that credit reads and examine
    /// rebuilds.
    ///
    /// The bound this module claims is a statement about exactly this number —
    /// that both walks together step over each value a constant number of times,
    /// rather than once per decision — so the tests assert on it instead of on a
    /// clock. A wall-clock measurement of the same claim measures the machine.
    static STEPPED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
fn step() {
    STEPPED.with(|count| count.set(count.get() + 1));
}

#[cfg(not(test))]
fn step() {}

/// The step count since the last call, which also resets it.
#[cfg(test)]
fn steps_taken() -> usize {
    STEPPED.with(|count| count.replace(0))
}

/// Whether a value is materialized. A value read once is rebuilt where it is
/// read, which costs nothing extra and holds no register; a shared value is
/// named unless the budget gave it back.
///
/// Everything else here is stated in terms of this predicate: an unnamed value
/// defines nothing, occupies nothing, and is transparent to the reads that pass
/// through it.
fn is_named(nodes: &[ScalarNode], inlined: &[bool], node: usize) -> bool {
    nodes[node].uses > 1 && !inlined[node]
}

fn must_materialize(tensor: &IndexedTensor) -> bool {
    matches!(
        tensor.expression.as_ref(),
        IndexedExpression::Pointer { .. } | IndexedExpression::Gather { .. }
    )
}

impl ScalarDag {
    pub(super) fn names(&self, node: usize) -> bool {
        is_named(&self.nodes, &self.inlined, node)
    }

    pub(super) fn key(&self, node: usize) -> ValueKey {
        let node = &self.nodes[node];
        (Arc::as_ptr(&node.tensor.expression) as usize, node.element)
    }
}

/// True when a value is cheap enough that a register never pays for itself.
fn is_immediate(tensor: &IndexedTensor) -> bool {
    matches!(
        tensor.expression.as_ref(),
        IndexedExpression::Dense(_) | IndexedExpression::Constant { .. }
    )
}

struct Builder {
    nodes: Vec<ScalarNode>,
    interned: HashMap<ValueKey, usize>,
}

impl Builder {
    /// Resolves re-indexing wrappers, which denote no value of their own, and
    /// returns the tensor and element that actually define the value.
    fn resolve(tensor: &IndexedTensor, element: usize) -> Result<(IndexedTensor, usize), String> {
        let mut tensor = tensor.clone();
        let mut element = element;
        loop {
            if element >= tensor.elements {
                return Err(format!(
                    "indexed element {element} is out of range for {} values",
                    tensor.elements
                ));
            }
            let next = match tensor.expression.as_ref() {
                IndexedExpression::Reindex { input, indices } => {
                    (input.clone(), indices.values[element])
                }
                IndexedExpression::Concatenate { inputs, offsets } => {
                    let input = offsets.partition_point(|&offset| offset <= element) - 1;
                    (inputs[input].clone(), element - offsets[input])
                }
                IndexedExpression::Pad {
                    input,
                    pad,
                    input_indices,
                    inside,
                } => {
                    if inside.values[element] == 1 {
                        (input.clone(), input_indices.values[element])
                    } else {
                        (pad.clone(), 0)
                    }
                }
                _ => return Ok((tensor, element)),
            };
            tensor = next.0;
            element = next.1;
        }
    }

    /// Interns a value and everything it depends on, returning its node index,
    /// or `None` when the value is an immediate that is never named.
    ///
    /// The walk carries its own stack rather than the machine's. A scalar DAG is
    /// as deep as the arithmetic it came from, and recursing once per level
    /// aborts the process on a program that is merely long — which is the shape
    /// recovery is here to handle in the first place.
    fn intern(&mut self, tensor: &IndexedTensor, element: usize) -> Result<Option<usize>, String> {
        // A value is interned only after everything it reads, so `finished`
        // holds one entry per completed `Read` and `Bind` collects the entries
        // its own reads left there.
        enum Step {
            Read(IndexedTensor, usize),
            Bind(ValueKey, IndexedTensor, usize, usize),
        }
        let mut work = vec![Step::Read(tensor.clone(), element)];
        let mut finished = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Read(tensor, element) => {
                    let (tensor, element) = Self::resolve(&tensor, element)?;
                    if is_immediate(&tensor) {
                        finished.push(None);
                        continue;
                    }
                    let key = (Arc::as_ptr(&tensor.expression) as usize, element);
                    if let Some(&node) = self.interned.get(&key) {
                        self.nodes[node].uses += 1;
                        finished.push(Some(node));
                        continue;
                    }
                    let reads = reads_of(&tensor, element);
                    work.push(Step::Bind(key, tensor, element, reads.len()));
                    // Reversed so the reads are taken in order, which keeps the
                    // node indices a topological order of the DAG.
                    for (input, index) in reads.into_iter().rev() {
                        work.push(Step::Read(input, index));
                    }
                }
                Step::Bind(key, tensor, element, count) => {
                    let mut operands = finished
                        .split_off(finished.len() - count)
                        .into_iter()
                        .flatten()
                        .collect::<Vec<_>>();
                    operands.sort_unstable();
                    operands.dedup();
                    operands.shrink_to_fit();
                    let node = self.nodes.len();
                    self.nodes.push(ScalarNode {
                        tensor,
                        element,
                        operands,
                        uses: 1,
                    });
                    self.interned.insert(key, node);
                    finished.push(Some(node));
                }
            }
        }
        Ok(finished.pop().flatten())
    }
}

/// The values one value reads. Dynamic reads — a gather's data-dependent index,
/// a reduction whose term list is not statically known — contribute no static
/// edges, so those nodes are DAG leaves and stay opaque.
fn reads_of(tensor: &IndexedTensor, element: usize) -> Vec<(IndexedTensor, usize)> {
    match tensor.expression.as_ref() {
        IndexedExpression::Elementwise { inputs, .. } => inputs
            .iter()
            .map(|input| {
                let index = if input.elements == 1 { 0 } else { element };
                (input.clone(), index)
            })
            .collect(),
        IndexedExpression::Convert { input, .. } => vec![(input.clone(), element)],
        IndexedExpression::Reduce {
            input,
            init,
            init_per_output,
            input_indices,
            offsets,
            ..
        } => {
            let mut reads = vec![(init.clone(), if *init_per_output { element } else { 0 })];
            let (Some(&start), Some(&stop)) =
                (offsets.values.get(element), offsets.values.get(element + 1))
            else {
                return reads;
            };
            // Past the unroll bound the backend walks the terms with a loop,
            // reading them under a runtime index. Those reads are not static, so
            // naming them would name values the emission never asks for.
            if !super::unrolls(stop - start) {
                return reads;
            }
            reads.extend(
                input_indices.values[start..stop]
                    .iter()
                    .map(|&term| (input.clone(), term)),
            );
            reads
        }
        IndexedExpression::Dense(_)
        | IndexedExpression::Pointer { .. }
        | IndexedExpression::Constant { .. }
        | IndexedExpression::Gather { .. }
        | IndexedExpression::Reindex { .. }
        | IndexedExpression::Concatenate { .. }
        | IndexedExpression::Pad { .. } => Vec::new(),
    }
}

/// The Sethi-Ullman number: the peak live count of the subtree rooted here when
/// it is evaluated alone. Shared values count as one, because they are named and
/// so cost exactly one register wherever they are read. Evaluating the operand
/// with the larger number first is what makes the bound tight on trees.
fn evaluation_widths(nodes: &[ScalarNode]) -> Vec<usize> {
    let mut widths = vec![1usize; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        let mut operand_widths = node
            .operands
            .iter()
            .map(|&operand| {
                if nodes[operand].uses > 1 {
                    1
                } else {
                    widths[operand]
                }
            })
            .collect::<Vec<_>>();
        operand_widths.sort_unstable_by(|a, b| b.cmp(a));
        widths[index] = operand_widths
            .into_iter()
            .enumerate()
            .map(|(rank, width)| width + rank)
            .max()
            .unwrap_or(1)
            .max(1);
    }
    widths
}

fn depths(nodes: &[ScalarNode]) -> Vec<u32> {
    let mut depths = vec![0u32; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        depths[index] = node
            .operands
            .iter()
            .map(|&operand| depths[operand] + 1)
            .max()
            .unwrap_or(0);
    }
    depths
}

/// The priority of a ready node. `pressure` is the net change in live values if
/// this node is emitted next: one for the value it defines, minus one for each
/// operand it retires. Choosing the smallest first is the standard
/// register-pressure rule; the width and depth tie-breaks make it agree with
/// Sethi-Ullman where the DAG is a tree.
fn priority(pressure: i64, width: usize, depth: u32, node: usize) -> (i64, i64, i64, usize) {
    (pressure, -(width as i64), -(depth as i64), node)
}

/// The step each named value is last read at, or `None` for one nothing reads.
///
/// A read of an unnamed value is really a read of whatever that value reads,
/// because the value is rebuilt at its consumer rather than held. So the walk
/// descends through everything unnamed and credits the named values underneath,
/// which is what operand evaluation does.
///
/// Only named values are walked from, and that is what makes this linear rather
/// than quadratic. An unnamed value is emitted at its consumer, so starting a
/// walk at its own step would say nothing new — and a chain of them would then
/// be descended once per link, which is the whole `O(V²)` on a program that is
/// mostly private arithmetic. Started only from named values, and with nothing
/// yet given back, every unnamed value has exactly one consumer and so is
/// descended into exactly once across the entire pass.
fn last_reads(nodes: &[ScalarNode], order: &[usize], inlined: &[bool]) -> Vec<Option<usize>> {
    let mut last = vec![None; nodes.len()];
    // Stamped per step so a value reached by several paths is walked once.
    let mut seen = vec![usize::MAX; nodes.len()];
    let mut stack = Vec::new();
    for (step, &node) in order.iter().enumerate() {
        if !is_named(nodes, inlined, node) {
            continue;
        }
        stack.extend(nodes[node].operands.iter().copied());
        while let Some(operand) = stack.pop() {
            self::step();
            if std::mem::replace(&mut seen[operand], step) == step {
                continue;
            }
            if is_named(nodes, inlined, operand) {
                last[operand] = Some(step);
            } else {
                stack.extend(nodes[operand].operands.iter().copied());
            }
        }
    }
    last
}

/// Live values at each step, and the peak — the closed form of what the walk in
/// [`relieve_pressure`] maintains incrementally, and what the tests check it
/// against.
#[cfg(test)]
fn liveness(nodes: &[ScalarNode], order: &[usize], inlined: &[bool]) -> (Vec<usize>, usize) {
    let reads = last_reads(nodes, order, inlined);
    let end = order.len().saturating_sub(1);
    let mut delta = vec![0i64; order.len() + 1];
    for (step, &node) in order.iter().enumerate() {
        if !is_named(nodes, inlined, node) {
            continue;
        }
        // A value nothing reads is an output: it stays live to the end.
        let dies_at = reads[node].unwrap_or(end).max(step);
        delta[step] += 1;
        delta[dies_at + 1] -= 1;
    }
    let mut live = Vec::with_capacity(order.len());
    let mut running = 0i64;
    let mut peak = 0usize;
    for &change in &delta[..order.len()] {
        running += change;
        let count = running.max(0) as usize;
        peak = peak.max(count);
        live.push(count);
    }
    (live, peak)
}

/// Orders the DAG so that few values are live at once, then names every value
/// that survives the budget. Returns the emission order and the inline set.
pub(super) fn plan(roots: &[(IndexedTensor, usize)], budget: usize) -> Result<ScalarDag, String> {
    let mut builder = Builder {
        nodes: Vec::new(),
        interned: HashMap::new(),
    };
    let mut root_nodes = Vec::with_capacity(roots.len());
    for (tensor, element) in roots {
        root_nodes.extend(builder.intern(tensor, *element)?);
    }
    let mut nodes = builder.nodes;
    let mut is_root = vec![false; nodes.len()];
    // A root is read by the kernel's output store, which the DAG does not model,
    // so it is shared even when nothing inside the DAG reads it.
    for &root in &root_nodes {
        nodes[root].uses += 1;
        is_root[root] = true;
    }
    let widths = evaluation_widths(&nodes);
    let depths = depths(&nodes);

    let mut consumers = vec![Vec::new(); nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        for &operand in &node.operands {
            consumers[operand].push(index);
        }
    }
    for list in &mut consumers {
        list.sort_unstable();
        list.dedup();
    }

    let mut remaining_uses = vec![0usize; nodes.len()];
    for node in &nodes {
        for &operand in &node.operands {
            remaining_uses[operand] += 1;
        }
    }
    let mut unscheduled_operands = nodes
        .iter()
        .map(|node| node.operands.len())
        .collect::<Vec<_>>();

    let pressure = |node: usize, nodes: &[ScalarNode], remaining: &[usize]| -> i64 {
        1 - nodes[node]
            .operands
            .iter()
            .filter(|&&operand| remaining[operand] == 1)
            .count() as i64
    };
    let key = |node: usize, nodes: &[ScalarNode], remaining: &[usize]| {
        priority(
            pressure(node, nodes, remaining),
            widths[node],
            depths[node],
            node,
        )
    };

    let mut ready = BTreeSet::new();
    for (index, &count) in unscheduled_operands.iter().enumerate() {
        if count == 0 {
            ready.insert(key(index, &nodes, &remaining_uses));
        }
    }

    let mut order = Vec::with_capacity(nodes.len());
    while let Some(&entry) = ready.iter().next() {
        ready.remove(&entry);
        let node = entry.3;
        order.push(node);
        for &operand in &nodes[node].operands {
            remaining_uses[operand] -= 1;
            if remaining_uses[operand] == 1 {
                // This operand now dies at whichever consumer reads it next,
                // which lowers those consumers' pressure; refresh their keys.
                for &consumer in &consumers[operand] {
                    if unscheduled_operands[consumer] == 0 {
                        let mut stale = key(consumer, &nodes, &remaining_uses);
                        stale.0 += 1;
                        if ready.remove(&stale) {
                            ready.insert(key(consumer, &nodes, &remaining_uses));
                        }
                    }
                }
            }
        }
        for &consumer in &consumers[node] {
            unscheduled_operands[consumer] -= 1;
            if unscheduled_operands[consumer] == 0 {
                ready.insert(key(consumer, &nodes, &remaining_uses));
            }
        }
    }
    if order.len() != nodes.len() {
        return Err(format!(
            "indexed schedule covered {} of {} values",
            order.len(),
            nodes.len()
        ));
    }

    let mut inlined = vec![false; nodes.len()];
    if budget > 0 {
        relieve_pressure(&nodes, &order, &is_root, &mut inlined, budget);
    }
    Ok(ScalarDag {
        nodes,
        order,
        inlined,
    })
}

/// Walks the order once and gives values back wherever the live count exceeds
/// the budget, cheapest to rebuild first.
///
/// The live set is carried along the walk, so a decision at a congested step is
/// made against the liveness at that step and nowhere else. That is also why
/// there is no "how long is this value live" term in the score: a value only
/// ever competes with the values live beside it, and a value the budget never
/// squeezes is never even considered.
///
/// What this leaves behind is *not* a peak within the budget, and saying so
/// would be a lie. It is a fixed point: when the walk finishes, no step over
/// budget holds a value that could have been given back — every one of them is
/// either read by something that already died, or would replay another rebuild.
/// The budget is what the pass works toward, not what it can promise, because
/// the refusals protect real memory traffic and arithmetic.
///
/// Cost is `O((V + E) log V)`, and each of the three terms is bounded rather
/// than hoped for. Every value is walked from at most once — a pop either gives
/// it back or drops it for good — and a walk covers only that value's private
/// cone, which by [`rebuild_reads`] is disjoint from every other candidate's. So
/// the walks together are one pass over the DAG, the heap contributes the
/// logarithm, and the live set is maintained by difference.
fn relieve_pressure(
    nodes: &[ScalarNode],
    order: &[usize],
    is_root: &[bool],
    inlined: &mut [bool],
    budget: usize,
) {
    let steps = order.len();
    if steps == 0 {
        return;
    }
    let end = steps - 1;
    let mut position = vec![0usize; nodes.len()];
    for (step, &node) in order.iter().enumerate() {
        position[node] = step;
    }
    let reads = last_reads(nodes, order, inlined);
    let mut death = (0..nodes.len())
        .map(|node| reads[node].unwrap_or(end).max(position[node]))
        .collect::<Vec<_>>();
    let costs = rebuild_costs(nodes);

    let mut retiring = vec![Vec::new(); steps];
    let mut alive = vec![false; nodes.len()];
    let mut visited = vec![false; nodes.len()];
    let mut heap = BinaryHeap::new();
    let mut below = Vec::new();
    let mut live = 0usize;

    for step in 0..steps {
        let node = order[step];
        if is_named(nodes, inlined, node) {
            alive[node] = true;
            live += 1;
            retiring[death[node]].push(node);
            // A root is read by the kernel's output store, which happens after
            // the last step, so giving one back would rebuild it there.
            if !is_root[node] && !must_materialize(&nodes[node].tensor) {
                heap.push(Reverse((costs[node], node)));
            }
        }
        while live > budget {
            let Some(given) = cheapest(nodes, inlined, &alive, &mut heap, &mut visited, &mut below)
            else {
                // Nothing live here can be given back for what it would free.
                // The step stays over budget rather than paying for arithmetic
                // that buys no register.
                break;
            };
            inlined[given] = true;
            alive[given] = false;
            live -= 1;
            // What it read must now survive until its last rebuild, which is
            // where its readers are — that is exactly where it used to die.
            for &held in &below {
                if death[given] > death[held] {
                    death[held] = death[given];
                    retiring[death[given]].push(held);
                }
            }
        }
        for held in std::mem::take(&mut retiring[step]) {
            if alive[held] && death[held] == step {
                alive[held] = false;
                live -= 1;
            }
        }
    }
}

/// The cheapest live value that can be given back at this step, with the values
/// a rebuild of it would read left in `below`.
fn cheapest(
    nodes: &[ScalarNode],
    inlined: &[bool],
    alive: &[bool],
    heap: &mut BinaryHeap<Reverse<(u64, usize)>>,
    visited: &mut [bool],
    below: &mut Vec<usize>,
) -> Option<usize> {
    while let Some(Reverse((_, node))) = heap.pop() {
        if inlined[node] || !alive[node] {
            continue;
        }
        // Giving a value back keeps everything it reads alive until its last
        // rebuild. If one of those has already died, reviving it takes back the
        // very register this was meant to free — and takes it back at steps the
        // walk has already left behind, where nothing can be done about it.
        //
        // Either refusal is permanent, which is what lets the entry be dropped
        // here rather than put back. What died stays dead, because nothing is
        // ever revived; and a value that reads a rebuild goes on reading one,
        // because nothing is ever taken back.
        if rebuild_reads(nodes, inlined, node, visited, below)
            && below.iter().all(|&held| alive[held])
        {
            return Some(node);
        }
    }
    None
}

/// The values a rebuild of `node` would read, or `false` if the rebuild would
/// replay another rebuild.
///
/// The walk descends through what is private to `node` — values read once, which
/// are emitted inside it and so are replayed with it — and stops at every value
/// that is named, which the rebuild simply reads. Reaching a value that was
/// *given back* is the one thing it refuses: rebuilding `node` would then rebuild
/// that one too, and rebuilds that compound are how a register budget turns into
/// an arithmetic explosion. Nothing is ever rebuilt twice over.
///
/// That refusal is also what bounds the walk. A value read once has exactly one
/// consumer, so it belongs to the private cone of exactly one named value above
/// it; the cones of distinct candidates are disjoint, and the walks together
/// cover the DAG once rather than once per decision.
fn rebuild_reads(
    nodes: &[ScalarNode],
    inlined: &[bool],
    node: usize,
    visited: &mut [bool],
    out: &mut Vec<usize>,
) -> bool {
    out.clear();
    let mut stack = nodes[node].operands.clone();
    let mut touched = Vec::new();
    let mut private = true;
    while let Some(operand) = stack.pop() {
        step();
        if std::mem::replace(&mut visited[operand], true) {
            continue;
        }
        touched.push(operand);
        if inlined[operand] {
            private = false;
            break;
        }
        if is_named(nodes, inlined, operand) {
            out.push(operand);
        } else {
            stack.extend(nodes[operand].operands.iter().copied());
        }
    }
    for operand in touched {
        visited[operand] = false;
    }
    private
}

/// Operations rebuilding each value costs: its own, plus everything private to
/// it, which is replayed with it.
///
/// A value read once counts here even though nothing gave it back: it was never
/// named to begin with, so rebuilding its consumer replays it. Pricing only what
/// the budget released would make a long chain of private products look as cheap
/// as a lone load.
///
/// Nothing given back appears in any of these figures, and nothing needs to.
/// A value that reads a rebuild is refused outright, so the cost of a value the
/// pass will actually consider never changes — which is why this is computed
/// once and why the heap needs no re-keying.
fn rebuild_costs(nodes: &[ScalarNode]) -> Vec<u64> {
    let mut costs = vec![0u64; nodes.len()];
    for (index, node) in nodes.iter().enumerate() {
        costs[index] = node
            .operands
            .iter()
            .filter(|&&operand| nodes[operand].uses <= 1)
            .fold(intrinsic_cost(node), |total, &operand| {
                total.saturating_add(costs[operand])
            });
    }
    costs
}

/// Operations one value performs on its own, counting a reduction's combines.
fn intrinsic_cost(node: &ScalarNode) -> u64 {
    match node.tensor.expression.as_ref() {
        IndexedExpression::Reduce { offsets, .. } => offsets
            .values
            .get(node.element + 1)
            .zip(offsets.values.get(node.element))
            .map_or(1, |(stop, start)| (stop - start).max(1) as u64),
        _ => 1,
    }
}

#[cfg(test)]
mod tests;
