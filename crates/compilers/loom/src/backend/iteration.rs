// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{DomainIndex, Graph, IterPlacement, OperandGraph, Slot};
use gpu_ir::Scalar;
use xir::dtype::DType;

use super::super::LiftedProgram;
use super::super::eval::math_scalar;
use super::partition::ElementOrder;
use crate::domain::Layout;
use crate::domain::reduction::Folds;

/// The operation's outputs grouped by the domains their axes iterate, coarse
/// defaults resolved against `default` — the iteration domain each axis takes
/// when the binding names none.
pub(super) fn output_domain_groups(
    loom: &LiftedProgram,
    default: &[i32],
) -> std::collections::BTreeMap<Vec<i32>, Vec<usize>> {
    let mut groups = std::collections::BTreeMap::<Vec<i32>, Vec<usize>>::new();
    for (output, binding) in loom.outputs().iter().enumerate() {
        let key = binding
            .axis_domains
            .iter()
            .enumerate()
            .map(
                |(axis, &domain)| {
                    if domain < 0 { default[axis] } else { domain }
                },
            )
            .collect();
        groups.entry(key).or_default().push(output);
    }
    groups
}

/// Which iteration a plan is built for. `FiberDirect` names one of the
/// promoted placements: it iterates a coarse anchor and writes finer-domain
/// outputs inside the fiber loop, where the walk covers each of their cells
/// exactly once. Only emitters that implement those writes may ask for it.
/// Placement `0` anchors on the domain the outputs name; the rest anchor
/// wherever else the operand graph reaches, which is where the *read* traffic
/// may live.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Placement {
    Default,
    FiberDirect(usize),
}

impl Placement {
    pub(super) fn name(self) -> String {
        match self {
            Self::Default => "default".to_string(),
            Self::FiberDirect(axis) => format!("fiber_direct_{axis}"),
        }
    }
}

pub(super) struct Plan {
    pub index: Index,
    /// How the element blocks of a kernel body are sequenced.
    pub element_order: ElementOrder,
    pub layout: Layout,
    pub folds: Folds,
    pub slots: Vec<Slot>,
    /// Symbolic expansions of the program, keyed by lane width alone. Planning
    /// emits dozens of kernel variants per compile and the expansion is the
    /// same for all of them, so it is computed once and shared.
    pub eval_cache: EvalCache,
}

/// Symbolic expansions keyed by lane width.
///
/// The expansion reads the operands off the names an access binding gives them
/// and says nothing about where those addresses come from, so it depends on the
/// program and the lane width and on nothing about the kernel emitting it. The
/// addresses themselves are the kernel's and are rebuilt for each one.
pub(super) type EvalCache =
    std::sync::Mutex<Vec<(u8, std::sync::Arc<super::super::eval::Evaluation>)>>;

pub(super) struct Index {
    graph: Graph,
    iter_domain_per_axis: Vec<i32>,
    operand_domains: Vec<Vec<i32>>,
}

impl Plan {
    pub(super) fn prepare(loom: &LiftedProgram) -> Result<Self, String> {
        Self::prepare_for(loom, Placement::Default)
            .transpose()
            .expect("default placement always exists")
    }

    /// The fiber-direct placement: outputs may sit on the fiber's finer
    /// domain (written inside the fiber loop, which covers each of their cells
    /// exactly once), which keeps the promotion onto the coarse output domain
    /// feasible for mixed-domain programs. `None` when the promotion does not
    /// move the iteration off the default placement — the plan would duplicate
    /// [`Plan::prepare`].
    pub(super) fn prepare_fiber_direct(loom: &LiftedProgram) -> Result<Option<Self>, String> {
        Self::prepare_for(loom, Placement::FiberDirect(0))
    }

    /// How many promoted placements this program admits.
    pub(super) fn fiber_direct_placements(loom: &LiftedProgram) -> Result<usize, String> {
        Ok(promoted_placements(loom)?.len())
    }

    pub(super) fn prepare_for(
        loom: &LiftedProgram,
        placement: Placement,
    ) -> Result<Option<Self>, String> {
        let operand_domains = loom
            .operands
            .iter()
            .map(|binding| binding.axis_domains.clone())
            .collect::<Vec<_>>();
        let operand_graph = operand_graph(loom, &operand_domains)?;
        let (iter_per_axis, used_views) = match placement {
            Placement::Default => operand_graph.promoted_iter(1, false)?,
            // A promoted placement that lands on the default iteration would
            // duplicate `Plan::prepare`, so it is not offered.
            Placement::FiberDirect(index) => {
                let placements = promoted_placements(loom)?;
                let Some((iter_per_axis, used_views)) = placements.get(index).cloned() else {
                    return Ok(None);
                };
                let (plain, _) = operand_graph.promoted_iter(1, false)?;
                if plain == iter_per_axis {
                    return Ok(None);
                }
                (iter_per_axis, used_views)
            }
        };
        let (graph, iter_domain_per_axis, slots) =
            operand_graph.build_index(&iter_per_axis, &used_views)?;
        let index = Index {
            graph,
            iter_domain_per_axis,
            operand_domains,
        };
        index.validate(&|operand| operand >= loom.num_inputs)?;

        let operand_scalars = loom
            .operands
            .iter()
            .map(|binding| storage_scalar(binding.dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let output_elements = loom
            .outputs()
            .iter()
            .enumerate()
            .map(|(output, binding)| {
                accumulated_elements(&index, loom.num_inputs + output, binding)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let func = loom.module.entry_func();
        let output_math_scalars = func
            .results
            .iter()
            .map(|&result| math_scalar(func.value_shape(result).dtype))
            .collect::<Result<Vec<_>, _>>()?;
        let defer_output_conversion = loom
            .outputs()
            .iter()
            .zip(&output_math_scalars)
            .map(|(binding, &math)| {
                Ok(binding.dtype != DType::Bool && storage_scalar(binding.dtype)? != math)
            })
            .collect::<Result<Vec<_>, String>>()?;
        let folds = Folds::new(
            &index,
            loom.num_inputs,
            &operand_scalars,
            &output_math_scalars,
            &defer_output_conversion,
            &output_elements,
        )?;
        let layout = Layout::new(
            &index,
            &slots,
            loom.num_inputs,
            folds.parameter_scalars(),
            &folds.bucket_selectors(),
        )?;

        Ok(Some(Plan {
            index,
            element_order: ElementOrder::Trace,
            layout,
            folds,
            slots,
            eval_cache: std::sync::Mutex::new(Vec::new()),
        }))
    }
}

/// The operand placement this program states, over the domains its operand
/// bindings name.
fn operand_graph<'a>(
    loom: &'a LiftedProgram,
    operand_domains: &'a [Vec<i32>],
) -> Result<OperandGraph<'a>, String> {
    OperandGraph::new(
        &loom.domains,
        &loom.index_maps,
        operand_domains,
        loom.num_axes,
        loom.num_inputs,
        &loom.available_views,
    )
}

/// The promoted placements this program admits, in
/// [`OperandGraph::promoted_placements`] order.
fn promoted_placements(loom: &LiftedProgram) -> Result<Vec<IterPlacement>, String> {
    let operand_domains = loom
        .operands
        .iter()
        .map(|binding| binding.axis_domains.clone())
        .collect::<Vec<_>>();
    operand_graph(loom, &operand_domains)?.promoted_placements(1, true)
}

impl DomainIndex for Index {
    fn graph(&self) -> &Graph {
        &self.graph
    }

    fn num_axes(&self) -> usize {
        self.iter_domain_per_axis.len()
    }

    fn iter_domain(&self, axis: usize) -> i32 {
        self.iter_domain_per_axis[axis]
    }

    fn num_operands(&self) -> usize {
        self.operand_domains.len()
    }

    fn operand_domain(&self, operand: usize, axis: usize) -> i32 {
        self.operand_domains[operand][axis]
    }
}

/// Every value an output accumulates over the whole operation: its cells on
/// the iteration domain times the visible shape times the hidden lane.
pub(super) fn accumulated_elements(
    index: &Index,
    operand: usize,
    binding: &super::super::FlatOperand,
) -> Result<usize, String> {
    let cells = usize::try_from(index.operand_batch_extent(operand)?)
        .map_err(|_| format!("loom operand {operand} has a negative cell extent"))?;
    cells
        .checked_mul(super::super::eval::num_elements(&binding.element_shape)?)
        .and_then(|elements| elements.checked_mul(binding.lane_extent as usize))
        .ok_or_else(|| format!("loom operand {operand} element count overflow"))
}

pub(super) fn storage_scalar(dtype: DType) -> Result<Scalar, String> {
    let scalar = match dtype {
        DType::Bool | DType::U8 => Scalar::U8,
        DType::F16 => Scalar::F16,
        DType::Bf16 => Scalar::BF16,
        DType::F32 => Scalar::F32,
        DType::F64 => Scalar::F64,
        DType::I8 => Scalar::S8,
        DType::I16 => Scalar::S16,
        DType::U16 => Scalar::U16,
        DType::I32 => Scalar::S32,
        DType::U32 => Scalar::U32,
        DType::I64 => Scalar::S64,
        DType::U64 => Scalar::U64,
        _ => {
            return Err(format!(
                "Loom PTX has no storage scalar for {}",
                dtype.name()
            ));
        }
    };
    Ok(scalar)
}
