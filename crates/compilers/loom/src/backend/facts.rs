// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Loom's axis-role facts, layered on [`xir::analysis::FuncFacts`]. The generic
//! table knows structure — use counts, liveness, provable zeros, statement
//! estimates. This layer adds what only Loom's operand layout knows: whether a
//! value varies along the hidden lane axis, whether it is invariant across it,
//! whether it derives from a DomainIndex-indexed operand, and a coarse register
//! estimate for one position body.
//!
//! Every fact is a single forward or single interval sweep over the entry
//! function, seeded from the per-operand lane extent and DomainIndex layout, so
//! the analysis stays near-linear and deterministic. It produces facts; the
//! backend consumes them.

use xir::analysis::FuncFacts;
use xir::ir::ValueId;

use super::super::LiftedProgram;

/// Per-value axis-role facts plus a register estimate for the entry function of
/// a [`LiftedProgram`].
pub(super) struct LoomFacts {
    structural: FuncFacts,
    hidden_lane: Vec<bool>,
    domain_index: Vec<bool>,
    computed: Vec<bool>,
    register_pressure: u64,
}

impl LoomFacts {
    pub(super) fn analyze(loom: &LiftedProgram) -> LoomFacts {
        let func = loom.module.entry_func();
        let structural = FuncFacts::analyze(func);
        let n = func.values.len();

        // Forward sweep: hidden-lane dependence and DomainIndex dependence flow
        // from the input operands to every value derived from them. A parameter
        // whose lane extent is the operation width varies along the hidden axis;
        // a width-1 parameter broadcasts and stays lane-uniform. Every loaded
        // operand is indexed by the DomainIndex; a constant is not.
        let mut hidden_lane = vec![false; n];
        let mut domain_index = vec![false; n];
        let mut computed = vec![false; n];
        for (input, &param) in func.params.iter().enumerate() {
            hidden_lane[param.index()] = loom.operands[input].lane_extent != 1;
            domain_index[param.index()] = true;
        }
        for node in &func.nodes {
            let varies = node.inputs.iter().any(|&v| hidden_lane[v.index()]);
            let indexed = node.inputs.iter().any(|&v| domain_index[v.index()]);
            for &result in &node.results {
                hidden_lane[result.index()] = varies;
                domain_index[result.index()] = indexed;
                computed[result.index()] = true;
            }
        }

        LoomFacts {
            register_pressure: register_pressure(loom),
            structural,
            hidden_lane,
            domain_index,
            computed,
        }
    }

    pub(super) fn structural(&self) -> &FuncFacts {
        &self.structural
    }

    /// Whether the value varies along the hidden lane axis.
    pub(super) fn hidden_lane(&self, value: ValueId) -> bool {
        self.hidden_lane[value.index()]
    }

    /// Whether the value is provably invariant across the hidden lane axis. In
    /// the current lane model — one hidden axis with no XIR reduction over it —
    /// this is exactly the complement of [`Self::hidden_lane`]: a width-1
    /// broadcast, a constant, or anything built only from those.
    pub(super) fn lane_uniform(&self, value: ValueId) -> bool {
        !self.hidden_lane(value)
    }

    /// Whether the value depends on a DomainIndex-indexed operand, i.e. it is
    /// derived from loaded data rather than being a compile-time constant.
    pub(super) fn domain_index(&self, value: ValueId) -> bool {
        self.domain_index[value.index()]
    }

    fn values(&self) -> impl Iterator<Item = ValueId> {
        (0..self.hidden_lane.len()).map(|i| ValueId(i as u32))
    }

    /// Computed values that are lane-uniform yet DomainIndex-dependent, and so
    /// can move out of the hidden-lane loop.
    pub(super) fn hoistable_computed_values(&self) -> usize {
        self.values()
            .filter(|&v| self.computed[v.index()] && self.lane_uniform(v) && self.domain_index(v))
            .count()
    }

    /// Total estimated scalar-VIR statements to expand the whole program.
    pub(super) fn estimated_statements(&self) -> u64 {
        self.structural()
            .values()
            .iter()
            .fold(0u64, |acc, facts| acc.saturating_add(facts.statements))
    }

    /// Peak simultaneously-live scalar registers across the position body, in
    /// 32-bit register slots. Coarse: a liveness-interval peak that weights each
    /// value by its visible element count and dtype width, ignoring lane
    /// replication (the hidden loop reuses a value's registers each trip).
    pub(super) fn register_pressure(&self) -> u64 {
        self.register_pressure
    }
}

fn register_pressure(loom: &LiftedProgram) -> u64 {
    let func = loom.module.entry_func();
    let len = func.nodes.len();
    let registers = |value: ValueId| -> u64 {
        let shape = func.value_shape(value);
        let slots = shape.dtype.size_bytes().div_ceil(4) as u64;
        saturating_product(shape.dims.iter().copied()).saturating_mul(slots)
    };

    // A value lives from its definition to its last use; a function result lives
    // to the end. Add its registers at birth and release them after its last
    // use; the running maximum is the peak.
    let mut add = vec![0u64; len + 1];
    let mut free = vec![0u64; len + 1];
    let intern = |birth: usize, death: usize, value: ValueId, add: &mut [u64], free: &mut [u64]| {
        let regs = registers(value);
        add[birth] = add[birth].saturating_add(regs);
        free[death] = free[death].saturating_add(regs);
    };

    let mut last_use = vec![None::<usize>; func.values.len()];
    for (index, node) in func.nodes.iter().enumerate() {
        for &input in &node.inputs {
            last_use[input.index()] = Some(index);
        }
    }
    for &result in &func.results {
        last_use[result.index()] = Some(len);
    }
    for &param in &func.params {
        let death = last_use[param.index()].unwrap_or(0);
        intern(0, death, param, &mut add, &mut free);
    }
    for (index, node) in func.nodes.iter().enumerate() {
        for &result in &node.results {
            let death = last_use[result.index()].unwrap_or(index);
            intern(index, death, result, &mut add, &mut free);
        }
    }

    let mut running = 0u64;
    let mut peak = 0u64;
    for t in 0..=len {
        running = running.saturating_add(add[t]);
        peak = peak.max(running);
        running = running.saturating_sub(free[t]);
    }
    peak
}

fn saturating_product(dims: impl IntoIterator<Item = i64>) -> u64 {
    dims.into_iter()
        .try_fold(1u64, |acc, dim| {
            u64::try_from(dim).ok().and_then(|dim| acc.checked_mul(dim))
        })
        .unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use domain_index::Domain;
    use xir::builder::ModuleBuilder;
    use xir::dtype::DType as Dtype;
    use xir::dtype::DType;
    use xir::ops::Literal;
    use xir::shape::Shape;

    use super::*;
    use crate::FlatOperand;

    fn program(
        build: impl FnOnce(&mut ModuleBuilder) -> Vec<ValueId>,
        operands: Vec<FlatOperand>,
        num_inputs: usize,
        u: i64,
    ) -> LiftedProgram {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let results = build(&mut builder);
        let main = builder.end_func(results);
        let module = builder.finish(main).unwrap();
        loom_program! {
            name: "facts".into(),
            module,
            lane_count: u,
            domains: vec![Domain {
                extent: 4,
            }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands,
            num_inputs,
            available_views: Vec::new(),
        }
    }

    fn binding(shape: &[i64], lane_extent: i64, dtype: Dtype) -> FlatOperand {
        FlatOperand {
            dtype,
            element_shape: shape.to_vec(),
            lane_extent,
            axis_domains: vec![0],
        }
    }

    #[test]
    fn hidden_lane_and_uniformity_track_the_operand_lane_extents() {
        // wide (lane_extent = u) input varies along the lane; a width-1 input
        // and a constant stay lane-uniform; a product with the wide input
        // becomes lane-varying.
        let loom = program(
            |f| {
                let wide = f.param(Shape::new(DType::F32, [2]));
                let narrow = f.param(Shape::new(DType::F32, [2]));
                let c = f.constant(Literal::zeros(Shape::new(DType::F32, [2])).unwrap());
                let uniform = f.multiply(narrow, c);
                let varying = f.multiply(wide, narrow);
                vec![varying, uniform]
            },
            vec![
                binding(&[2], 8, Dtype::F32),
                binding(&[2], 1, Dtype::F32),
                binding(&[2], 8, Dtype::F32),
                binding(&[2], 1, Dtype::F32),
            ],
            2,
            8,
        );
        let facts = LoomFacts::analyze(&loom);
        let func = loom.module.entry_func();
        let wide = func.params[0];
        let narrow = func.params[1];
        let constant = func.nodes[0].results[0];
        let uniform = func.nodes[1].results[0];
        let varying = func.nodes[2].results[0];
        assert!(facts.hidden_lane(wide));
        assert!(!facts.hidden_lane(narrow));
        assert!(facts.lane_uniform(narrow));
        assert!(facts.lane_uniform(constant));
        assert!(facts.lane_uniform(uniform), "narrow * const stays uniform");
        assert!(
            facts.hidden_lane(varying),
            "wide * narrow varies over the lane"
        );
        // A constant is not DomainIndex-dependent; loaded data is.
        assert!(facts.domain_index(wide));
        assert!(!facts.domain_index(constant));
        assert!(facts.domain_index(varying));
    }

    #[test]
    fn register_pressure_peaks_at_the_widest_live_set() {
        let loom = program(
            |f| {
                let x = f.param(Shape::new(DType::F64, [4]));
                let y = f.multiply(x, x);
                vec![y]
            },
            vec![binding(&[4], 8, Dtype::F64), binding(&[4], 8, Dtype::F64)],
            1,
            8,
        );
        let facts = LoomFacts::analyze(&loom);
        // x and y are both live across the product: 4 elements * 2 slots each,
        // times two live values.
        assert_eq!(facts.register_pressure(), 4 * 2 * 2);
    }

    #[test]
    fn analysis_is_deterministic() {
        let loom = program(
            |f| {
                let x = f.param(Shape::new(DType::F32, [2]));
                vec![f.multiply(x, x)]
            },
            vec![binding(&[2], 8, Dtype::F32), binding(&[2], 8, Dtype::F32)],
            1,
            8,
        );
        let first = LoomFacts::analyze(&loom);
        let second = LoomFacts::analyze(&loom);
        assert_eq!(first.register_pressure(), second.register_pressure());
        assert_eq!(first.structural().values(), second.structural().values());
    }
}
