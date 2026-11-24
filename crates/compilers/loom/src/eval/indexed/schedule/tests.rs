// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, Evict, Param, ParamTy, Scalar};
use xir::dtype::DType;
use xir::ops::ScalarOp;

use super::*;

fn operand(b: &mut Builder, elements: usize) -> IndexedTensor {
    let base = b.parameter(Param {
        name: format!("operand_{}", b.function().num_values()),
        ty: ParamTy::Ptr(Scalar::F32),
    });
    IndexedTensor::pointer(
        base,
        Scalar::F32,
        1,
        1,
        1,
        false,
        None,
        Evict::Normal,
        elements,
    )
}

fn binary(op: ScalarOp, left: IndexedTensor, right: IndexedTensor) -> IndexedTensor {
    let elements = left.elements.max(right.elements);
    IndexedTensor::elementwise(op, DType::F32, vec![left, right], elements)
}

/// Combines many values into one without nesting the tensors thousands deep.
fn balanced(op: ScalarOp, mut values: Vec<IndexedTensor>) -> IndexedTensor {
    while values.len() > 1 {
        values = values
            .chunks(2)
            .map(|pair| match pair {
                [left, right] => binary(op, left.clone(), right.clone()),
                [only] => only.clone(),
                _ => unreachable!(),
            })
            .collect();
    }
    values.pop().expect("at least one value")
}

fn roots(tensor: &IndexedTensor) -> Vec<(IndexedTensor, usize)> {
    (0..tensor.elements)
        .map(|element| (tensor.clone(), element))
        .collect()
}

/// The order must be a real topological order: every operand is emitted before
/// the value that reads it, or the emitted program reads an undefined register.
fn assert_topological(dag: &ScalarDag) {
    let mut position = vec![usize::MAX; dag.nodes.len()];
    for (step, &node) in dag.order.iter().enumerate() {
        position[node] = step;
    }
    for (node, scalar_node) in dag.nodes.iter().enumerate() {
        for &operand in &scalar_node.operands {
            assert!(
                position[operand] < position[node],
                "operand {operand} scheduled after its consumer {node}"
            );
        }
    }
}

#[test]
fn reindexing_wrappers_share_one_value() {
    let mut b = Builder::new();
    let source = operand(&mut b, 4);
    let doubled = binary(ScalarOp::Multiply, source.clone(), source.clone());
    // Two different paths onto the same underlying element: directly, and
    // through a permutation that maps element 0 back onto element 0.
    let permuted = doubled.reindex("perm".into(), vec![0, 1, 2, 3]).unwrap();
    let sum = binary(ScalarOp::Add, doubled.clone(), permuted);

    let dag = plan(&roots(&sum), 0).unwrap();
    assert_topological(&dag);
    // One product per element, not two: the reindex denotes no value of its own.
    let products = dag
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.tensor.expression.as_ref(),
                IndexedExpression::Elementwise {
                    op: ScalarOp::Multiply,
                    ..
                }
            )
        })
        .count();
    assert_eq!(products, 4);
}

#[test]
fn a_value_read_twice_is_interned_once_and_counts_both_reads() {
    let mut b = Builder::new();
    let source = operand(&mut b, 1);
    let shared = binary(ScalarOp::Multiply, source.clone(), source.clone());
    let left = binary(ScalarOp::Add, shared.clone(), source.clone());
    let right = binary(ScalarOp::Subtract, shared.clone(), source.clone());
    let total = binary(ScalarOp::Add, left, right);

    let dag = plan(&roots(&total), 0).unwrap();
    assert_topological(&dag);
    let shared_key = (Arc::as_ptr(&shared.expression) as usize, 0);
    let node = dag
        .nodes
        .iter()
        .position(|node| {
            (Arc::as_ptr(&node.tensor.expression) as usize, node.element) == shared_key
        })
        .expect("the shared product is a DAG node");
    assert_eq!(dag.nodes[node].uses, 2);
}

#[test]
fn constants_are_never_named() {
    let mut b = Builder::new();
    let source = operand(&mut b, 2);
    let immediate = vec![b.float(Scalar::F32, 1.0), b.float(Scalar::F32, 2.0)];
    let constant = IndexedTensor::constant("weights".into(), Scalar::F32, vec![0; 8], 2, immediate);
    let scaled = binary(ScalarOp::Multiply, source.clone(), constant);

    let dag = plan(&roots(&scaled), 0).unwrap();
    assert!(
        dag.nodes.iter().all(|node| !is_immediate(&node.tensor)),
        "an immediate was given a register"
    );
}

/// A balanced sum of independent products: any order works, but a
/// pressure-blind one holds every product at once. Scheduling by the live-count
/// rule must not exceed what the tree actually needs.
#[test]
fn independent_products_do_not_all_stay_live() {
    let mut b = Builder::new();
    let source = operand(&mut b, 16);
    let products = (0..8)
        .map(|i| {
            let left = source.reindex(format!("l{i}"), vec![i]).unwrap();
            let right = source.reindex(format!("r{i}"), vec![i + 8]).unwrap();
            binary(ScalarOp::Multiply, left, right)
        })
        .collect::<Vec<_>>();
    let mut total = products[0].clone();
    for product in &products[1..] {
        total = binary(ScalarOp::Add, total, product.clone());
    }

    let dag = plan(&roots(&total), 0).unwrap();
    assert_topological(&dag);
    // Each product is read once, so a good order retires it immediately: the
    // accumulator, the product being folded in, and its two loads.
    let peak = liveness(&dag.nodes, &dag.order, &dag.inlined).1;
    assert!(
        peak <= 5,
        "peak of {peak} live values for a linear accumulation"
    );
}

#[test]
fn every_value_is_named_when_the_budget_allows() {
    let mut b = Builder::new();
    let source = operand(&mut b, 4);
    let a = binary(ScalarOp::Multiply, source.clone(), source.clone());
    let b = binary(ScalarOp::Add, a.clone(), source.clone());
    let c = binary(ScalarOp::Multiply, a, b);

    let dag = plan(&roots(&c), 64).unwrap();
    assert!(
        dag.inlined.iter().all(|&inlined| !inlined),
        "values were recomputed despite a budget that fits"
    );
    assert!(liveness(&dag.nodes, &dag.order, &dag.inlined).1 <= 64);
}

/// With a budget of one register the schedule cannot hold everything, so it must
/// give some values up — and it must still cover the whole DAG.
#[test]
fn a_tight_budget_trades_arithmetic_for_registers() {
    let mut b = Builder::new();
    let source = operand(&mut b, 8);
    let loads = (0..8)
        .map(|i| source.reindex(format!("v{i}"), vec![i]).unwrap())
        .collect::<Vec<_>>();
    // Every load is read twice, so each is a candidate for recomputation.
    let mut total = binary(ScalarOp::Multiply, loads[0].clone(), loads[0].clone());
    for load in &loads[1..] {
        let square = binary(ScalarOp::Multiply, load.clone(), load.clone());
        total = binary(ScalarOp::Add, total, square);
    }

    let unconstrained = plan(&roots(&total), 0).unwrap();
    let constrained = plan(&roots(&total), 2).unwrap();
    assert_topological(&constrained);
    assert_eq!(constrained.nodes.len(), unconstrained.nodes.len());
    let peak = |dag: &ScalarDag| liveness(&dag.nodes, &dag.order, &dag.inlined).1;
    assert!(
        peak(&constrained) <= peak(&unconstrained),
        "tightening the budget raised the peak"
    );
}

/// What the budget gives back, and in what order.
///
/// Naming is not what makes a fused multiply-add possible — an inline product is
/// read once by construction and contracts just the same — so nothing is held
/// for fusion's sake. What decides is cost against liveness: a value that is
/// cheap to rebuild and holds a register a long time goes first, and a value
/// with a deep cone below it stays because rebuilding it would replay all of it.
#[test]
fn the_budget_gives_back_what_is_cheapest_to_rebuild() {
    let mut b = Builder::new();
    let source = operand(&mut b, 16);
    // A deep chain, expensive to rebuild, read twice.
    let mut deep = source.reindex("d0".into(), vec![0]).unwrap();
    for step in 0..6 {
        deep = binary(
            ScalarOp::Multiply,
            deep,
            source
                .reindex(format!("d{}", step + 1), vec![step + 1])
                .unwrap(),
        );
    }
    // A lone product, cheap to rebuild, also read twice.
    let cheap = binary(
        ScalarOp::Multiply,
        source.reindex("c0".into(), vec![8]).unwrap(),
        source.reindex("c1".into(), vec![9]).unwrap(),
    );
    let left = binary(ScalarOp::Add, deep.clone(), cheap.clone());
    let right = binary(ScalarOp::Subtract, deep.clone(), cheap.clone());
    let total = binary(ScalarOp::Add, left, right);

    let generous = plan(&roots(&total), 64).unwrap();
    let node_of = |dag: &ScalarDag, wanted: &IndexedTensor| {
        dag.nodes.iter().position(|node| {
            (Arc::as_ptr(&node.tensor.expression) as usize, node.element)
                == (Arc::as_ptr(&wanted.expression) as usize, 0)
        })
    };
    let deep_node = node_of(&generous, &deep).expect("the chain is a node");
    let cheap_node = node_of(&generous, &cheap).expect("the product is a node");
    assert!(
        !generous.inlined[deep_node] && !generous.inlined[cheap_node],
        "a budget that fits gives nothing back"
    );

    // The chain is only ever given back once the cheap product already has been:
    // rebuilding it replays six multiplies and seven loads, all private and so
    // all recomputed with it, against the product's three. And nothing is given
    // back at all once the budget covers what the program already holds.
    let held = liveness(&generous.nodes, &generous.order, &generous.inlined).1;
    for budget in 1..8 {
        let tight = plan(&roots(&total), budget).unwrap();
        let deep_node = node_of(&tight, &deep).expect("the chain is a node");
        let cheap_node = node_of(&tight, &cheap).expect("the product is a node");
        assert!(
            !(tight.inlined[deep_node] && !tight.inlined[cheap_node]),
            "budget {budget} gave back the expensive chain but kept the cheap product"
        );
        assert_eq!(
            tight.inlined[cheap_node],
            budget < held,
            "budget {budget} gave the cheap product back when it did not have to"
        );
    }
}

/// Every value the pass could still give back at a step over budget, which the
/// fixed point says must be none.
fn affordable_at(dag: &ScalarDag, step: usize, live: &[usize]) -> Vec<usize> {
    let mut position = vec![0usize; dag.nodes.len()];
    for (at, &node) in dag.order.iter().enumerate() {
        position[node] = at;
    }
    let reads = last_reads(&dag.nodes, &dag.order, &dag.inlined);
    let end = dag.order.len().saturating_sub(1);
    let mut visited = vec![false; dag.nodes.len()];
    let mut below = Vec::new();
    let alive = |node: usize| {
        dag.names(node)
            && position[node] <= step
            && reads[node].unwrap_or(end).max(position[node]) >= step
    };
    let _ = live;
    (0..dag.nodes.len())
        .filter(|&node| alive(node) && dag.nodes[node].uses > 1)
        .filter(|&node| !must_materialize(&dag.nodes[node].tensor))
        // A root is stored after the last step, so it is never a candidate.
        .filter(|&node| !dag.order.is_empty() && reads[node].is_some())
        .filter(|&node| {
            rebuild_reads(&dag.nodes, &dag.inlined, node, &mut visited, &mut below)
                && below.iter().all(|&held| alive(held))
        })
        .collect()
}

/// What the pass actually promises. It is not a peak within the budget — two
/// refusals stand in the way of that, and both are deliberate — but a fixed
/// point: wherever a step is left over budget, nothing live there could have
/// been given back.
#[test]
fn a_step_left_over_the_budget_had_nothing_left_to_give_back() {
    // Every value is read twice and every one of them reads the one below, so
    // after a single give-back the whole column above it is refused as a
    // compounded rebuild and the budget cannot be reached.
    let mut b = Builder::new();
    let source = operand(&mut b, 4);
    let mut column = source.reindex("v0".into(), vec![0]).unwrap();
    let mut held = Vec::new();
    for step in 0..8 {
        column = binary(
            ScalarOp::Multiply,
            column.clone(),
            source.reindex(format!("w{step}"), vec![1]).unwrap(),
        );
        held.push(column.clone());
    }
    let mut total = held[0].clone();
    for value in &held {
        total = binary(ScalarOp::Add, total, value.clone());
    }

    let dag = plan(&roots(&total), 2).unwrap();
    let (live, peak) = liveness(&dag.nodes, &dag.order, &dag.inlined);
    assert!(peak > 2, "this DAG was supposed to stay over its budget");
    for (step, &count) in live.iter().enumerate() {
        if count > 2 {
            let affordable = affordable_at(&dag, step, &live);
            assert!(
                affordable.is_empty(),
                "step {step} held {count} values and could still have given back {affordable:?}"
            );
        }
    }
}

/// Where the refusals do not bite, the fixed point is the budget itself.
#[test]
fn the_walk_leaves_no_step_over_the_budget() {
    // Thirty-two arithmetic values, each read twice, all live across one long
    // accumulation. Their inputs are immediates, so every held value can be
    // rebuilt without extending another value's lifetime.
    let mut b = Builder::new();
    let values = (0..32)
        .map(|i| {
            let left = IndexedTensor::dense(vec![b.float(Scalar::F32, i as f64 + 1.0)]);
            let right = IndexedTensor::dense(vec![b.float(Scalar::F32, i as f64 + 2.0)]);
            binary(ScalarOp::Multiply, left, right)
        })
        .collect::<Vec<_>>();
    let mut total = values[0].clone();
    for value in &values[1..] {
        total = binary(ScalarOp::Add, total, value.clone());
    }
    for value in &values {
        total = binary(ScalarOp::Multiply, total, value.clone());
    }

    for budget in 1..8 {
        let dag = plan(&roots(&total), budget).unwrap();
        assert_topological(&dag);
        let peak = liveness(&dag.nodes, &dag.order, &dag.inlined).1;
        assert!(peak <= budget, "budget {budget} left a step holding {peak}");
    }
}

#[test]
fn a_tight_budget_never_rebuilds_memory_reads() {
    let mut b = Builder::new();
    let source = operand(&mut b, 32);
    let values = (0..32)
        .map(|element| {
            source
                .reindex(format!("v{element}"), vec![element])
                .unwrap()
        })
        .collect::<Vec<_>>();
    let first = balanced(ScalarOp::Add, values.clone());
    let second = balanced(ScalarOp::Multiply, values);
    let dag = plan(&roots(&binary(ScalarOp::Add, first, second)), 1).unwrap();

    let pointers = dag
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, node)| {
            matches!(
                node.tensor.expression.as_ref(),
                IndexedExpression::Pointer { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(pointers.len(), 32, "distinct elements were merged");
    assert!(
        pointers
            .iter()
            .all(|(node, value)| value.uses == 2 && !dag.inlined[*node]),
        "a shared memory read was rebuilt"
    );
}

/// A square lattice of shared values: every value but the leaves is read twice,
/// so every one of them is a naming decision and none of them is private.
fn lattice(side: usize) -> IndexedTensor {
    let mut b = Builder::new();
    let source = operand(&mut b, side);
    let mut level = (0..side)
        .map(|i| source.reindex(format!("v{i}"), vec![i]).unwrap())
        .collect::<Vec<_>>();
    for _ in 0..side {
        level = (0..side)
            .map(|i| {
                binary(
                    ScalarOp::Multiply,
                    level[i].clone(),
                    level[(i + 1) % side].clone(),
                )
            })
            .collect();
    }
    let mut total = level[0].clone();
    for value in &level[1..] {
        total = binary(ScalarOp::Add, total, value.clone());
    }
    total
}

/// Plans a DAG and reports how many values the two walks stepped over, against
/// the `V + E` the bound allows each of them.
fn steps_against_size(tensor: &IndexedTensor, budget: usize) -> (usize, usize, ScalarDag) {
    let _ = steps_taken();
    let dag = plan(&roots(tensor), budget).unwrap();
    let stepped = steps_taken();
    let edges = dag
        .nodes
        .iter()
        .map(|node| node.operands.len())
        .sum::<usize>();
    (stepped, dag.nodes.len() + edges, dag)
}

/// The claim, checked as a claim rather than as a stopwatch.
///
/// Crediting reads steps over each value once and each edge once, and so does
/// examining rebuilds, because a value read once belongs to the private cone of
/// exactly one candidate and every candidate is examined at most once. Two walks
/// over `V + E` is the whole bound, and it holds at any size — which is a
/// stronger statement than two timings whose ratio happens to look linear, and
/// one that says nothing about the machine it ran on.
#[test]
fn the_walks_step_over_the_dag_a_bounded_number_of_times() {
    for tensor in [lattice(16), lattice(64), lattice(128)] {
        let (stepped, size, dag) = steps_against_size(&tensor, 32);
        assert_topological(&dag);
        assert!(
            stepped <= 2 * size,
            "{stepped} steps over a DAG of {size} values and edges"
        );
    }
}

/// The rule that bounds every walk, stated directly: a rebuild may replay what
/// was private to the value, and may not replay another rebuild.
#[test]
fn a_rebuild_that_would_replay_another_rebuild_is_refused() {
    let mut b = Builder::new();
    let source = operand(&mut b, 4);
    let shared = binary(
        ScalarOp::Multiply,
        source.reindex("a".into(), vec![0]).unwrap(),
        source.reindex("b".into(), vec![1]).unwrap(),
    );
    let above = binary(
        ScalarOp::Multiply,
        shared.clone(),
        source.reindex("c".into(), vec![2]).unwrap(),
    );
    // Both are read twice, so both are named and either could be given back.
    let total = binary(
        ScalarOp::Add,
        binary(ScalarOp::Add, shared.clone(), above.clone()),
        binary(ScalarOp::Subtract, shared.clone(), above.clone()),
    );

    let dag = plan(&roots(&total), 0).unwrap();
    let node_of = |wanted: &IndexedTensor| {
        dag.nodes
            .iter()
            .position(|node| {
                (Arc::as_ptr(&node.tensor.expression) as usize, node.element)
                    == (Arc::as_ptr(&wanted.expression) as usize, 0)
            })
            .expect("the value is a node")
    };
    let mut inlined = vec![false; dag.nodes.len()];
    let mut visited = vec![false; dag.nodes.len()];
    let mut below = Vec::new();

    // With nothing given back, the value above reads the shared product.
    assert!(rebuild_reads(
        &dag.nodes,
        &inlined,
        node_of(&above),
        &mut visited,
        &mut below
    ));
    assert_eq!(below, vec![node_of(&shared)]);

    // Once the shared product is given back, rebuilding the value above would
    // rebuild it a second time, so it is refused instead.
    inlined[node_of(&shared)] = true;
    assert!(
        !rebuild_reads(
            &dag.nodes,
            &inlined,
            node_of(&above),
            &mut visited,
            &mut below
        ),
        "a rebuild was allowed to replay another rebuild"
    );
}

/// One value read once by the next, `depth` deep, `count` times over. The chain
/// is the shape the lattice cannot express: it is all private cone and no
/// sharing, which is what a walk started at every value descends over and over.
fn chains(count: usize, depth: usize) -> IndexedTensor {
    let mut b = Builder::new();
    let source = operand(&mut b, 2);
    let shared = binary(
        ScalarOp::Multiply,
        source.reindex("a".into(), vec![0]).unwrap(),
        source.reindex("b".into(), vec![1]).unwrap(),
    );
    let mut ends = Vec::with_capacity(count);
    for chain in 0..count {
        let offset = IndexedTensor::dense(vec![b.float(Scalar::F32, chain as f64)]);
        let mut link = binary(ScalarOp::Add, shared.clone(), offset);
        for _ in 1..depth {
            link = binary(ScalarOp::Multiply, link, shared.clone());
        }
        ends.push(link);
    }
    balanced(ScalarOp::Add, ends)
}

/// The adversary for the walk that credits reads. Every link is private to the
/// one above it, so a walk started at every value descends the whole chain below
/// it — quadratic in the depth, and invisible to a lattice that shares
/// everything. Started only at named values, each link is descended into once.
#[test]
fn a_long_private_chain_is_stepped_over_once_and_not_once_per_link() {
    for depth in [64, 512, 4096] {
        let (stepped, size, dag) = steps_against_size(&chains(8, depth), 4);
        assert_topological(&dag);
        assert!(
            stepped <= 2 * size,
            "{stepped} steps for chains {depth} deep over {size} values and edges"
        );
    }
}

/// The other adversary: one value read by very many consumers, where anything
/// that walks a fan-out per decision shows up as steps far past `V + E`.
#[test]
fn a_value_with_a_large_fan_out_is_stepped_over_no_more_than_its_edges() {
    let fan = |readers: usize| {
        let mut b = Builder::new();
        let source = operand(&mut b, 2);
        let shared = binary(
            ScalarOp::Multiply,
            source.reindex("a".into(), vec![0]).unwrap(),
            source.reindex("b".into(), vec![1]).unwrap(),
        );
        // Every reader is itself read twice, so every one of them is a candidate.
        let held = (0..readers)
            .map(|i| {
                let offset = IndexedTensor::dense(vec![b.float(Scalar::F32, i as f64)]);
                binary(ScalarOp::Add, shared.clone(), offset)
            })
            .collect::<Vec<_>>();
        binary(
            ScalarOp::Add,
            balanced(ScalarOp::Add, held.clone()),
            balanced(ScalarOp::Multiply, held),
        )
    };
    for readers in [256, 4096] {
        let (stepped, size, dag) = steps_against_size(&fan(readers), 4);
        assert_topological(&dag);
        assert!(
            stepped <= 2 * size,
            "{stepped} steps for {readers} readers over {size} values and edges"
        );
    }
}

/// A DAG deeper than any call stack would survive. Interning walks it with its
/// own stack, so depth costs memory rather than the process.
#[test]
fn a_dag_deeper_than_the_call_stack_is_interned_without_recursing() {
    let deep = chains(1, 1 << 17);
    let dag = plan(&roots(&deep), 4).unwrap();
    assert!(dag.nodes.len() > 1 << 17, "{} values", dag.nodes.len());
    assert_topological(&dag);
}

/// The whole point of interning: a value reached from many places is one node,
/// so the emitter names it once and every reader loads that register. Without
/// this each consumer evaluates the operand again.
#[test]
fn a_value_reached_from_many_places_is_one_node() {
    let mut b = Builder::new();
    let source = operand(&mut b, 1);
    let shared = binary(ScalarOp::Multiply, source.clone(), source.clone());
    // Thirty-two independent readers of the same product.
    let mut readers = Vec::new();
    for i in 0..32 {
        let offset = IndexedTensor::dense(vec![b.float(Scalar::F32, i as f64)]);
        readers.push(binary(ScalarOp::Add, shared.clone(), offset));
    }
    let mut total = readers[0].clone();
    for reader in &readers[1..] {
        total = binary(ScalarOp::Add, total, reader.clone());
    }

    let dag = plan(&roots(&total), 0).unwrap();
    let products = dag
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.tensor.expression.as_ref(),
                IndexedExpression::Elementwise {
                    op: ScalarOp::Multiply,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(products.len(), 1, "the shared product was duplicated");
    assert_eq!(products[0].uses, 32);
}

#[test]
fn reductions_read_every_term_of_their_own_output() {
    let mut b = Builder::new();
    let source = operand(&mut b, 6);
    let init = IndexedTensor::dense(vec![b.float(Scalar::F32, 0.0)]);
    let squares = binary(ScalarOp::Multiply, source.clone(), source.clone());
    let reduced = IndexedTensor::reduce(
        "sum".into(),
        squares,
        init,
        xir::ir::FuncId(0),
        &[0, 0, 0, 1, 1, 1],
        2,
    )
    .unwrap();

    let dag = plan(&roots(&reduced), 0).unwrap();
    assert_topological(&dag);
    let reduce_nodes = dag
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                node.tensor.expression.as_ref(),
                IndexedExpression::Reduce { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(reduce_nodes.len(), 2);
    for node in reduce_nodes {
        assert_eq!(node.operands.len(), 3, "a reduction lost a term");
    }
}
