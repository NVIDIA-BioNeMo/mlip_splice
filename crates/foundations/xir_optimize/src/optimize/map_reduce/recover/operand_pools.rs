// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Flat operand pools and the gathers that read them.
//!
//! A batched map reads its operands by index, so every leaf a batch can select
//! must live at a known offset of one indexable value. Parameters become one
//! flat reshape each; scalar literals of a common type become one constant
//! vector. A leaf's pool then decides which batch it can join at all: two terms
//! share a batch only when each of their leaf positions draws from the same
//! pool, because one gather reads one pool.

use std::collections::HashMap;

use crate::dim_numbers::GatherDimensionNumbers;
use crate::dtype::DType;
use crate::ir::{Def, Func, ValueId};
use crate::ops::{Constant, Gather, Literal, Reshape, Slice};
use crate::shape::{Dims, DomainAxes, Shape};
use smallvec::{SmallVec, smallvec};

use super::super::super::rebuild::Rebuilder;
use super::Region;
use super::infer_shapes::{ShapeId, Shapes, producer};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct PoolKey {
    dtype: DType,
    domain_axes: DomainAxes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct Location {
    pub pool: usize,
    pub element: usize,
}

pub(super) struct LeafPools {
    values: Vec<ValueId>,
    locations: HashMap<ValueId, Location>,
}

impl LeafPools {
    pub(super) fn location(&self, leaf: ValueId) -> Option<Location> {
        self.locations.get(&leaf).copied()
    }

    pub(super) fn value(&self, pool: usize) -> ValueId {
        self.values[pool]
    }

    /// Lays every leaf out in an indexable pool. Parameters keep their natural
    /// element order, so a gather index is just the flat offset the scalar
    /// program already sliced.
    pub(super) fn build(
        rebuilt: &mut Rebuilder,
        old: &Func,
        leaves: &[ValueId],
    ) -> Result<Self, String> {
        let mut pools = Self {
            values: Vec::new(),
            locations: HashMap::new(),
        };
        let mut parameter_pools = HashMap::new();
        for &leaf in leaves {
            let Some((parameter, element)) = parameter_element(old, leaf) else {
                continue;
            };
            let pool = match parameter_pools.get(&parameter).copied() {
                Some(pool) => pool,
                None => {
                    let shape = old.value_shape(parameter);
                    let elements = i64::try_from(shape.element_count()?)
                        .map_err(|_| "a pooled parameter has no i64 element count".to_string())?;
                    let value = rebuilt.emit_fresh(
                        shape.with_dims([elements]),
                        Reshape {
                            out_dims: Dims::from_slice(&[elements]),
                        },
                        [rebuilt.mapped(parameter)?],
                    );
                    let pool = pools.values.len();
                    pools.values.push(value);
                    parameter_pools.insert(parameter, pool);
                    pool
                }
            };
            pools.locations.insert(leaf, Location { pool, element });
        }

        let mut groups: HashMap<PoolKey, Vec<(ValueId, Literal)>> = HashMap::new();
        for &leaf in leaves {
            if pools.locations.contains_key(&leaf) {
                continue;
            }
            // A leaf that is neither a parameter element nor a literal — the
            // result of a gather or a contraction, say — cannot be laid out at a
            // known offset. It simply gets no location, and any batch that would
            // have read it is declined rather than the whole pass failing.
            let Some(literal) = scalar_literal(old, leaf) else {
                continue;
            };
            let shape = old.value_shape(leaf);
            groups
                .entry(PoolKey {
                    dtype: shape.dtype,
                    domain_axes: shape.domain_axes.clone(),
                })
                .or_default()
                .push((leaf, literal.clone()));
        }
        let mut groups = groups.into_iter().collect::<Vec<_>>();
        groups.sort_by_key(|(_, values)| values[0].0.index());
        for (_, mut values) in groups {
            values.sort_unstable_by_key(|(value, _)| value.index());
            let shape = values[0].1.shape.with_dims([values.len() as i64]);
            let mut data = Vec::with_capacity(values.len() * shape.dtype.size_bytes());
            for (_, literal) in &values {
                data.extend_from_slice(&literal.data);
            }
            let exact = values
                .iter()
                .map(|(_, literal)| {
                    literal
                        .exact
                        .as_ref()
                        .and_then(|values| values.first().cloned())
                })
                .collect::<Option<Vec<_>>>();
            let literal = match exact {
                Some(exact) => Literal::exact(shape.clone(), exact)?,
                None => Literal::new(shape.clone(), data),
            };
            let value = rebuilt.emit_fresh(shape, Constant { literal }, []);
            let pool = pools.values.len();
            pools.values.push(value);
            for (element, (leaf, _)) in values.into_iter().enumerate() {
                pools.locations.insert(leaf, Location { pool, element });
            }
        }
        Ok(pools)
    }
}

/// A constant index vector, shaped as the scatter and gather dimension numbers
/// in this module expect: one row per lane, one index component per row.
pub(super) fn index_vector(
    rebuilt: &mut Rebuilder,
    values: &[usize],
    domain_axes: &DomainAxes,
) -> Result<ValueId, String> {
    let bytes = values
        .iter()
        .map(|&value| {
            i32::try_from(value)
                .map(i32::to_le_bytes)
                .map_err(|_| format!("indexed map index {value} does not fit i32"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let shape = Shape::new(DType::I32, [values.len() as i64, 1])
        .with_domain_axes(domain_axes.iter().copied());
    Ok(rebuilt.emit_fresh(
        shape.clone(),
        Constant {
            literal: Literal::new(shape, bytes),
        },
        [],
    ))
}

pub(super) fn is_parameter(func: &Func, value: ValueId) -> bool {
    matches!(func.values[value.index()].def, Def::Param(_))
}

/// Terms batch together when their atoms have the same shapes in the same order
/// *and* every leaf is drawn from the same pool, because one gather reads one
/// pool and one replay emits one operation sequence.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct GroupKey {
    pub(super) atoms: Vec<ShapeId>,
    pub(super) pools: Vec<usize>,
    pub(super) wrappers: Vec<String>,
}

pub(super) struct Group {
    pub(super) key: GroupKey,
    pub(super) terms: Vec<usize>,
    /// Every term's leaves, concatenated across its atoms in replay order.
    pub(super) leaves: Vec<Vec<ValueId>>,
}

/// Sorts terms into batches. A term's atoms are ordered canonically first, which
/// is sound because the product is commutative, and is what lets two terms that
/// wrote their factors in different orders share one batch.
pub(super) fn group_terms(
    func: &Func,
    region: &mut Region,
    shapes: &Shapes,
    assigned: &[Option<ShapeId>],
    pools: &LeafPools,
) -> Option<Vec<Group>> {
    let mut index_of: HashMap<GroupKey, usize> = HashMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for index in 0..region.terms.len() {
        let mut described = Vec::with_capacity(region.terms[index].len());
        for &atom in &region.terms[index] {
            let shape = assigned[atom.index()]?;
            let mut leaves = Vec::new();
            super::infer_shapes::leaf_order(func, atom, shapes, assigned, &mut leaves)?;
            let signature = leaves
                .iter()
                .map(|&leaf| pools.location(leaf).map(|location| location.pool))
                .collect::<Option<Vec<_>>>()?;
            described.push((shape, signature, leaves, atom));
        }
        described.sort_by(|left, right| {
            (left.0, &left.1, left.3.index()).cmp(&(right.0, &right.1, right.3.index()))
        });
        region.terms[index] = described.iter().map(|atom| atom.3).collect();
        let key = GroupKey {
            atoms: described.iter().map(|atom| atom.0).collect(),
            pools: described.iter().flat_map(|atom| atom.1.clone()).collect(),
            wrappers: region.wrappers[index]
                .iter()
                .map(|op| format!("{op:?}"))
                .collect(),
        };
        let leaves = described
            .into_iter()
            .flat_map(|atom| atom.2)
            .collect::<Vec<_>>();
        let group = match index_of.get(&key) {
            Some(&group) => group,
            None => {
                index_of.insert(key.clone(), groups.len());
                groups.push(Group {
                    key,
                    terms: Vec::new(),
                    leaves: Vec::new(),
                });
                groups.len() - 1
            }
        };
        groups[group].terms.push(index);
        groups[group].leaves.push(leaves);
    }
    Some(groups)
}

pub(super) fn take_vector() -> Gather {
    Gather {
        dnums: GatherDimensionNumbers {
            offset_dims: SmallVec::new(),
            collapsed_slice_dims: smallvec![0],
            operand_batching_dims: SmallVec::new(),
            start_indices_batching_dims: SmallVec::new(),
            start_index_map: smallvec![0],
            index_vector_dim: 1,
        },
        slice_sizes: Dims::from_slice(&[1]),
        indices_are_sorted: false,
    }
}

fn parameter_element(func: &Func, value: ValueId) -> Option<(ValueId, usize)> {
    match func.values[value.index()].def {
        Def::Param(_) => {
            return (func.value_shape(value).num_elements() == Some(1)).then_some((value, 0));
        }
        Def::NodeResult { .. } => {}
    }
    let node = producer(func, value)?;
    if node.op.as_any().is::<Reshape>() {
        return parameter_element(func, node.inputs[0]);
    }
    let slice = node.op.as_any().downcast_ref::<Slice>()?;
    if func.value_shape(value).num_elements() != Some(1) {
        return None;
    }
    let source = node.inputs[0];
    if !is_parameter(func, source) {
        return None;
    }
    let dims = &func.value_shape(source).dims;
    if dims.len() != slice.start_indices.len() {
        return None;
    }
    dims.iter()
        .zip(&slice.start_indices)
        .try_fold(0usize, |flat, (&extent, &coordinate)| {
            let extent = usize::try_from(extent).ok()?;
            let coordinate = usize::try_from(coordinate).ok()?;
            (coordinate < extent).then_some(flat.checked_mul(extent)?.checked_add(coordinate)?)
        })
        .map(|element| (source, element))
}

fn scalar_literal(func: &Func, value: ValueId) -> Option<&Literal> {
    crate::analysis::constant_literal(func, value)
        .filter(|literal| literal.shape.num_elements() == Some(1))
}
