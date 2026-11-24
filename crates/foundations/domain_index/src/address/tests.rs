// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the index emitters build, read back as the text of each operation.

use crate::{AlignedView, Domain, Graph, IndexMap};

use super::*;

/// An index algebra whose values are the text of what was built, so a test
/// can read off exactly which leaf reached a gather.
struct Symbolic;

impl IndexBuilder for Symbolic {
    type Value = String;

    fn lit(&mut self, v: i64) -> String {
        v.to_string()
    }

    fn gather(&mut self, array: String, idx: String) -> String {
        format!("{array}[{idx}]")
    }

    fn max0(&mut self, value: String) -> String {
        format!("max0({value})")
    }

    fn to_i64(&mut self, value: String) -> String {
        format!("i64({value})")
    }

    fn mul_lit(&mut self, value: String, k: i64) -> String {
        format!("({value} * {k})")
    }

    fn add(&mut self, a: String, b: String) -> String {
        format!("({a} + {b})")
    }

    fn nonneg(&mut self, array: String, idx: String) -> String {
        format!("{array}[{idx}] >= 0")
    }

    fn nonneg_value(&mut self, value: String) -> String {
        format!("{value} >= 0")
    }

    fn and(&mut self, a: String, b: String) -> String {
        format!("({a} && {b})")
    }
}

/// One operand on one axis, reached through the bridge it was built with.
struct Index {
    graph: Graph,
    bridge: AxisBridge,
}

impl DomainIndex for Index {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn num_axes(&self) -> usize {
        1
    }

    fn iter_domain(&self, _axis: usize) -> i32 {
        0
    }

    fn num_operands(&self) -> usize {
        1
    }

    fn operand_domain(&self, _operand: usize, _axis: usize) -> i32 {
        0
    }

    fn axis_bridge(&self, _operand: usize, _axis: usize) -> AxisBridge {
        self.bridge
    }
}

fn graph(index_maps: Vec<IndexMap>) -> Graph {
    Graph {
        domains: vec![Domain { extent: 8 }, Domain { extent: 4 }],
        index_maps,
        aligned_views: vec![AlignedView {
            source_id: 0,
            anchor_target_id: 0,
            other_target_id: 1,
        }],
    }
}

fn index_map(may_have_unmapped: bool) -> IndexMap {
    IndexMap {
        source_id: 0,
        target_id: 1,
        is_injective: false,
        is_sorted: true,
        may_have_unmapped,
    }
}

/// The view's other domain is refined only when it leaves cells unmapped, which
/// is what [`Graph::aligned_view_other_may_have_unmapped`] reads.
fn index(other_has_unmapped: bool) -> Index {
    let index_maps = if other_has_unmapped {
        vec![index_map(true)]
    } else {
        Vec::new()
    };
    Index {
        graph: graph(index_maps),
        bridge: AxisBridge::AlignedLookup { view_idx: 0 },
    }
}

fn gathered(may_have_unmapped: bool) -> Index {
    Index {
        graph: graph(vec![index_map(may_have_unmapped)]),
        bridge: AxisBridge::MapLookup { map_idx: 0 },
    }
}

struct Ctx {
    index: Index,
    element: Option<String>,
    fiber: Option<String>,
}

impl AddressSource<Symbolic> for Ctx {
    type Index = Index;

    fn index(&self) -> &Index {
        &self.index
    }

    fn axis_index(&self, _b: &mut Symbolic, axis: usize) -> String {
        format!("axis{axis}")
    }

    fn target_indices_array(&self, _b: &mut Symbolic, map_idx: usize) -> String {
        format!("target_indices{map_idx}")
    }

    fn aligned_view_array(&self, _b: &mut Symbolic, view_idx: usize) -> String {
        format!("view{view_idx}")
    }

    fn view_element(&self, _b: &mut Symbolic, _view_idx: usize) -> Option<String> {
        self.element.clone()
    }

    fn fiber_index(&self) -> String {
        self.fiber
            .clone()
            .expect("a view is bridged only inside the fiber loop")
    }
}

/// The value the context is iterating on is the one that gathers — the
/// emitter has no name of its own to reach for.
#[test]
fn a_view_bridge_gathers_at_the_index_the_context_supplies() {
    let ctx = Ctx {
        index: index(false),
        element: None,
        fiber: Some("this_loops_index".into()),
    };
    let flat = operand_flat_index(&ctx, &mut Symbolic, 0, Some("edge".into()));
    assert_eq!(flat, "i64(view0[this_loops_index])");
}

/// And it decides the predicate the same way.
#[test]
fn an_unmapped_view_is_tested_at_the_index_the_context_supplies() {
    let ctx = Ctx {
        index: index(true),
        element: None,
        fiber: Some("this_loops_index".into()),
    };
    let alive = operand_alive(&ctx, &mut Symbolic, 0);
    assert_eq!(alive.as_deref(), Some("view0[this_loops_index] >= 0"));
}

/// A context that pre-bound the element is not asked for an index at all,
/// so the fiber it is on need not be nameable here.
#[test]
fn a_pre_bound_view_element_stands_on_its_own() {
    let ctx = Ctx {
        index: index(true),
        element: Some("bound_element".into()),
        fiber: None,
    };
    let flat = operand_flat_index(&ctx, &mut Symbolic, 0, Some("edge".into()));
    assert_eq!(flat, "i64(max0(bound_element))");
    let alive = operand_alive(&ctx, &mut Symbolic, 0);
    assert_eq!(alive.as_deref(), Some("bound_element >= 0"));
}

/// A context outside any fiber says so, rather than gathering at a name
/// nothing bound.
#[test]
#[should_panic(expected = "a view is bridged only inside the fiber loop")]
fn a_view_bridged_outside_a_fiber_refuses() {
    let ctx = Ctx {
        index: index(false),
        element: None,
        fiber: None,
    };
    operand_flat_index(&ctx, &mut Symbolic, 0, Some("edge".into()));
}

/// A gather is tested at the axis index, which is the one place that index is
/// read: the view arm never asks for it.
#[test]
fn an_unmapped_gather_is_tested_at_the_axis_index() {
    let ctx = Ctx {
        index: gathered(true),
        element: None,
        fiber: None,
    };
    let alive = operand_alive(&ctx, &mut Symbolic, 0);
    assert_eq!(alive.as_deref(), Some("target_indices0[axis0] >= 0"));
}

/// A caller with only the index decides the shape of an expansion the predicate
/// will land in, so the two must answer alike on every bridge — for both of
/// them, over every axis, the operand is masked or it is not.
#[test]
fn asking_whether_an_operand_is_masked_answers_as_building_the_predicate_does() {
    for (what, index, masked) in [
        ("mapped view", index(false), false),
        ("unmapped view", index(true), true),
        ("mapped gather", gathered(false), false),
        ("unmapped gather", gathered(true), true),
    ] {
        assert_eq!(
            operand_has_alive(&index, 0),
            masked,
            "{what} is masked: {masked}"
        );
        let ctx = Ctx {
            index,
            element: None,
            fiber: Some("this_loops_index".into()),
        };
        assert_eq!(
            operand_has_alive(ctx.index(), 0),
            operand_alive(&ctx, &mut Symbolic, 0).is_some(),
            "{what}"
        );
    }
}

/// An operand masked on no axis has no predicate at all, rather than one that
/// is always true — a predicate the expansion would have to read a name for.
#[test]
fn an_unmasked_operand_has_no_predicate() {
    let ctx = Ctx {
        index: index(false),
        element: None,
        fiber: Some("this_loops_index".into()),
    };
    assert_eq!(operand_alive(&ctx, &mut Symbolic, 0), None);
}
