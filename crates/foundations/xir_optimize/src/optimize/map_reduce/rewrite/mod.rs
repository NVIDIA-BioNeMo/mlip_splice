// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Exact rewrites through gathers and sums along finite maps.

use std::sync::Arc;

use rayon::prelude::*;

use crate::ir::{Def, Func, Module, ValueId};
use crate::ops::{Add, Multiply, Negate, Select, Subtract, SumAlongMap};
use crate::optimize::AlgebraicWorkBudget;
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;
use algebraic_real::AlgebraicReal;

mod egraph;
mod frontier;
pub(in crate::optimize) use frontier::{FiniteMapFrontier, describe_finite_map_frontier};
mod extraction {
    //! Bounded extraction of finite-map expressions by work on either side of a sum.

    use std::collections::{HashMap, HashSet, VecDeque};

    #[cfg(test)]
    use super::ExprTree;
    use super::FiniteMapWork;
    use super::egraph::RewriteGraph;
    use crate::ir::{Func, ValueId};
    use crate::optimize::AlgebraicWorkBudget;

    pub(super) mod expression_dag;
    mod select_candidates;

    use expression_dag::{ExpressionDag, ExpressionId, ExpressionNode};

    const MAX_FRONTIER: usize = 16;
    const MAX_EXTRACT_WORK: usize = 1 << 20;

    #[derive(Debug, PartialEq, Eq)]
    enum ExtractionLimit {
        ConstructionBudget,
        ExpressionSize,
    }

    struct ExtractionBudget {
        remaining: usize,
    }

    impl ExtractionBudget {
        fn new() -> Self {
            Self {
                remaining: MAX_EXTRACT_WORK,
            }
        }

        fn reserve(&mut self, work: usize) -> Result<(), ExtractionLimit> {
            self.remaining = self
                .remaining
                .checked_sub(work)
                .ok_or(ExtractionLimit::ConstructionBudget)?;
            Ok(())
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum AccumulationSide {
        BeforeAccumulation,
        AfterAccumulation,
    }

    impl AccumulationSide {
        fn index(self) -> usize {
            match self {
                Self::BeforeAccumulation => 0,
                Self::AfterAccumulation => 1,
            }
        }

        fn opposite(self) -> Self {
            match self {
                Self::BeforeAccumulation => Self::AfterAccumulation,
                Self::AfterAccumulation => Self::BeforeAccumulation,
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct CountedOperation {
        side: AccumulationSide,
        instances: u64,
        expression: ExpressionId,
    }

    #[derive(Clone, Debug)]
    struct ExtractionCandidate {
        work: FiniteMapWork,
        tree: ExpressionId,
        operations: HashSet<CountedOperation>,
        quality: (usize, usize, usize),
    }

    #[derive(Clone, Debug)]
    pub(super) struct RootExtraction {
        pub(super) work: FiniteMapWork,
        physical_work: u128,
        pub(super) roots: Vec<ExpressionId>,
        quality: (usize, usize, usize),
    }

    impl RootExtraction {
        fn dominates(&self, other: &Self) -> bool {
            self.physical_work <= other.physical_work
                && self.work.dominates(other.work)
                && (self.physical_work != other.physical_work
                    || self.work != other.work
                    || self.quality <= other.quality)
        }
    }

    pub(super) struct RootFrontier {
        #[cfg(test)]
        pub(super) refinement_exhausted: bool,
        pub(super) dag: ExpressionDag,
        pub(super) candidates: Vec<RootExtraction>,
    }

    #[derive(Clone, Debug)]
    struct RootCandidate {
        tree: ExpressionId,
        operations: Vec<usize>,
        nodes: usize,
    }

    #[derive(Clone, Debug)]
    struct CombinedRoots {
        extraction: RootExtraction,
        operations: Vec<u64>,
    }

    struct InternedRootOperations {
        candidates_by_root: HashMap<u32, Vec<RootCandidate>>,
        work_by_operation: Vec<CountedOperation>,
        opposite_sides: Vec<Option<usize>>,
    }

    impl ExtractionCandidate {
        fn value(value: ValueId, dag: &mut ExpressionDag) -> Self {
            Self {
                work: FiniteMapWork::default(),
                tree: dag.intern(ExpressionNode::Value(value)),
                operations: HashSet::new(),
                quality: (1, 0, 0),
            }
        }

        fn unary(&self, tree: ExpressionId, dag: &ExpressionDag) -> Self {
            Self {
                work: self.work,
                tree,
                operations: self.operations.clone(),
                quality: dag.quality([tree]),
            }
        }

        fn binary(
            left: &Self,
            right: &Self,
            side: AccumulationSide,
            instances: u64,
            make_node: impl FnOnce(ExpressionId, ExpressionId) -> ExpressionNode,
            dag: &mut ExpressionDag,
        ) -> Self {
            let tree = dag.intern(make_node(left.tree, right.tree));
            let mut operations = left.operations.clone();
            operations.extend(right.operations.iter().cloned());
            operations.insert(CountedOperation {
                side,
                instances,
                expression: tree,
            });
            let work = work_for_operations(&operations);
            Self {
                work,
                tree,
                operations,
                quality: dag.quality([tree]),
            }
        }
    }

    fn work_for_operations(operations: &HashSet<CountedOperation>) -> FiniteMapWork {
        operations
            .iter()
            .fold(FiniteMapWork::default(), |mut work, operation| {
                match operation.side {
                    AccumulationSide::BeforeAccumulation => {
                        work.before_accumulation =
                            work.before_accumulation.saturating_add(operation.instances);
                    }
                    AccumulationSide::AfterAccumulation => {
                        work.after_accumulation =
                            work.after_accumulation.saturating_add(operation.instances);
                    }
                }
                work
            })
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub(super) enum ENode {
        Value(ValueId),
        Coefficient(usize),
        Negate {
            instances: u64,
            value: u32,
        },
        Select {
            instances: u64,
            predicate: u32,
            on_true: u32,
            on_false: u32,
        },
        Add {
            instances: u64,
            left: u32,
            right: u32,
        },
        Multiply {
            instances: u64,
            left: u32,
            right: u32,
        },
        Gather {
            map: domain_index::FiniteMapId,
            axis: u32,
            value: u32,
        },
        Sum {
            map: domain_index::FiniteMapId,
            axis: u32,
            value: u32,
        },
    }

    pub(super) type Classes = Vec<Vec<ENode>>;
    type ClassFrontiers = Vec<[Vec<ExtractionCandidate>; 2]>;

    #[cfg(test)]
    pub(super) fn extract_roots(
        func: &Func,
        graph: &domain_index::FiniteMapGraph,
        originals: &[ExprTree],
    ) -> Result<Vec<RootExtraction>, String> {
        use crate::optimize::{
            DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK,
            DEFAULT_MAX_POLYNOMIAL_TERMS,
        };
        let mut budget = AlgebraicWorkBudget::new(
            DEFAULT_MAX_ALGEBRAIC_WORK,
            DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
        );
        let mut dag = ExpressionDag::default();
        let originals = originals
            .iter()
            .map(|tree| dag.import(tree))
            .collect::<Vec<_>>();
        Ok(extract_roots_with_limits(
            func,
            graph,
            dag,
            &originals,
            0,
            DEFAULT_MAX_POLYNOMIAL_TERMS,
            &mut budget,
        )?
        .candidates)
    }

    pub(super) fn extract_roots_with_limits(
        func: &Func,
        maps: &domain_index::FiniteMapGraph,
        mut dag: ExpressionDag,
        originals: &[ExpressionId],
        function: usize,
        max_terms: usize,
        budget: &mut AlgebraicWorkBudget,
    ) -> Result<RootFrontier, String> {
        let mut graph = RewriteGraph::new(func, maps)?;
        graph.import(&dag)?;
        let source_len = dag.len();
        let proofs = graph.rewrite(&mut dag, function, max_terms, budget)?;
        let (snapshot, source_classes) = graph.snapshot_dag(&dag)?;
        let instances = graph.instance_counts(&snapshot)?;
        let roots = originals
            .iter()
            .map(|id| source_classes[id.0])
            .collect::<Vec<_>>();
        let coefficients = graph
            .coefficients
            .iter()
            .map(|tree| dag.import(tree))
            .collect::<Vec<_>>();
        let mut candidates = select_candidates::extract(
            &snapshot.classes,
            &roots,
            originals,
            select_candidates::Witnesses {
                source_len,
                classes: &source_classes,
                proofs: &proofs,
            },
            &instances,
            &coefficients,
            &mut dag,
        );
        let refinements = select_candidates::refine_snapshot_roots(
            snapshot.classes,
            roots,
            &coefficients,
            &mut dag,
        );
        #[cfg(test)]
        let refinement_exhausted = refinements.is_err();
        if let Ok(refinements) = refinements {
            for candidate in refinements {
                select_candidates::insert(&mut candidates, candidate, originals, &dag);
            }
        }
        Ok(RootFrontier {
            dag,
            candidates,
            #[cfg(test)]
            refinement_exhausted,
        })
    }

    #[cfg(test)]
    mod projection_tests;

    #[cfg(test)]
    mod tests;

    #[cfg(test)]
    mod pipeline_tests;
}
mod proposals {
    mod factor;
    mod identities;
    mod recognize;

    pub(super) use factor::proposals;
    pub(super) use identities::proposals as identity_proposals;
}

pub(super) const MAX_FRONTIER: usize = 16;

/// Accumulation placement heuristic, not a physical operation union.
/// A shared operation used on both sides contributes to both fields.
/// `before + after * k` prices contexts for a partial-contribution ratio `k`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiniteMapWork {
    pub before_accumulation: u64,
    pub after_accumulation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartialContributionRatio {
    contributions: u64,
    outputs: u64,
}

impl PartialContributionRatio {
    pub fn new(contributions: u64, outputs: u64) -> Result<Self, String> {
        if contributions == 0 || outputs == 0 {
            return Err("finite-map partial-contribution ratio must be positive".into());
        }
        let divisor = gcd(contributions, outputs);
        Ok(Self {
            contributions: contributions / divisor,
            outputs: outputs / divisor,
        })
    }

    pub fn scaled_work(self, work: FiniteMapWork) -> u128 {
        u128::from(work.before_accumulation) * u128::from(self.outputs)
            + u128::from(work.after_accumulation) * u128::from(self.contributions)
    }
}

fn gcd(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        (left, right) = (right, left % right);
    }
    left
}

impl FiniteMapWork {
    pub fn instantiate(self, partial_contributions: u64) -> u64 {
        self.before_accumulation.saturating_add(
            self.after_accumulation
                .saturating_mul(partial_contributions),
        )
    }

    fn combine(self, other: Self) -> Self {
        Self {
            before_accumulation: self
                .before_accumulation
                .saturating_add(other.before_accumulation),
            after_accumulation: self
                .after_accumulation
                .saturating_add(other.after_accumulation),
        }
    }

    fn dominates(self, other: Self) -> bool {
        self.before_accumulation <= other.before_accumulation
            && self.after_accumulation <= other.after_accumulation
    }
}

#[derive(Clone, Debug)]
pub struct FiniteMapAlternative {
    pub module: Module,
    pub work: FiniteMapWork,
}

#[derive(Clone)]
struct ModuleExtraction {
    alternative: FiniteMapAlternative,
    physical_work: u128,
    changed: bool,
}

#[derive(Clone)]
struct FuncPath {
    previous: Option<usize>,
    func: Arc<Func>,
}

#[derive(Clone)]
struct PartialModuleExtraction {
    path: Option<usize>,
    work: FiniteMapWork,
    physical_work: u128,
    changed: bool,
}

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Tree fixtures share the coefficient proof-leaf type"
    )
)]
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ExprTree {
    Value(ValueId),
    Coefficient {
        shape: Shape,
        values: Arc<[AlgebraicReal]>,
    },
    Negate(Box<ExprTree>),
    Select(Box<ExprTree>, Box<ExprTree>, Box<ExprTree>),
    Add(Box<ExprTree>, Box<ExprTree>),
    Multiply(Box<ExprTree>, Box<ExprTree>),
    Gather {
        map: domain_index::FiniteMapId,
        axis: u32,
        value: Box<ExprTree>,
    },
    Sum {
        map: domain_index::FiniteMapId,
        axis: u32,
        value: Box<ExprTree>,
    },
}

pub(in crate::optimize) fn indexed_algebra_canonicalize(
    module: &mut Module,
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Result<bool, String> {
    if max_terms == 0 || budget.remaining == 0 || budget.expression_limit == 0 {
        return Ok(false);
    }
    let mut extractions = extract_module_frontier(module, max_terms, budget)?;
    let selected = extractions
        .iter()
        .enumerate()
        .min_by_key(|(_, extraction)| {
            (
                extraction.physical_work,
                extraction
                    .alternative
                    .module
                    .funcs
                    .iter()
                    .map(|func| func.nodes.len())
                    .sum::<usize>(),
                extraction.changed,
                extraction.alternative.work.before_accumulation,
                extraction.alternative.work.after_accumulation,
            )
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "finite-map module has no extraction".to_string())?;
    let extraction = extractions.swap_remove(selected);
    *module = extraction.alternative.module;
    Ok(extraction.changed)
}

fn original_module_extraction(module: &Module) -> Result<ModuleExtraction, String> {
    let work = module
        .funcs
        .iter()
        .try_fold(FiniteMapWork::default(), |work, func| {
            Ok::<_, String>(work.combine(original_work(func, &module.finite_map_graph)?))
        })?;
    Ok(ModuleExtraction {
        alternative: FiniteMapAlternative {
            module: module.clone(),
            work,
        },
        physical_work: module
            .funcs
            .iter()
            .map(|func| physical_work(func, &module.finite_map_graph))
            .sum::<Result<u128, _>>()?,
        changed: false,
    })
}

fn extract_module_frontier(
    module: &Module,
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Result<Vec<ModuleExtraction>, String> {
    Ok(frontier::describe_module_frontier(module, max_terms, budget, &mut || Ok(()))?.collect())
}

fn extract_func_frontier(
    old: &Func,
    graph: &domain_index::FiniteMapGraph,
    function: usize,
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
) -> Result<Vec<(Func, FiniteMapWork, bool)>, String> {
    let Some(source) = extraction::expression_dag::import(old)? else {
        return Ok(vec![(old.clone(), original_work(old, graph)?, false)]);
    };
    let roots = source.roots;
    let source_values = source.values;
    if roots.is_empty() {
        return Ok(vec![(old.clone(), original_work(old, graph)?, false)]);
    }
    let originals = roots.iter().map(|(_, id)| *id).collect::<Vec<_>>();
    let extraction::RootFrontier {
        dag,
        candidates: extracted,
        ..
    } = extraction::extract_roots_with_limits(
        old, graph, source.dag, &originals, function, max_terms, budget,
    )?;
    let latest = extraction::expression_dag::latest_definitions(&dag, old);
    // Alternatives share the proof graph, then rebuild their chosen root DAGs independently.
    extracted
        .into_par_iter()
        .filter(|extraction| {
            roots.iter().zip(&extraction.roots).all(|((value, _), id)| {
                let Def::NodeResult { node, .. } = old.values[value.index()].def else {
                    return false;
                };
                latest[id.0] <= node
            })
        })
        .map(|extraction| {
            if extraction.roots == originals {
                return Ok((old.clone(), original_work(old, graph)?, false));
            }
            let mut rebuilt = Rebuilder::new(old);
            let replacements = roots
                .iter()
                .map(|(value, _)| *value)
                .zip(&extraction.roots)
                .collect::<std::collections::HashMap<_, _>>();
            let live = extraction::expression_dag::live_nodes(old, &dag, &replacements);
            let mut changed = false;
            let mut emitted = std::collections::HashMap::new();
            for (index, node) in old.nodes.iter().enumerate() {
                if !live[index] {
                    continue;
                }
                let replacement = node
                    .results
                    .first()
                    .filter(|_| node.results.len() == 1)
                    .and_then(|result| replacements.get(result));
                let Some(tree) = replacement else {
                    let inputs = rebuilt.mapped_inputs(node)?;
                    let results = rebuilt.emit(old, &node.results, node.op.clone(), inputs)?;
                    for (&old, &new) in node.results.iter().zip(&results) {
                        if let Some(id) = source_values[old.index()] {
                            emitted.entry(id).or_insert(new);
                        }
                    }
                    continue;
                };
                let value = extraction::expression_dag::emit(
                    &mut rebuilt,
                    graph,
                    &dag,
                    **tree,
                    &mut emitted,
                )?;
                if rebuilt.func.value_shape(value) != old.value_shape(node.results[0]) {
                    return Err("finite-map extraction changed the result shape".into());
                }
                rebuilt.alias(&node.results, &[value])?;
                if let Some(id) = source_values[node.results[0].index()] {
                    emitted.entry(id).or_insert(value);
                }
                changed = true;
            }
            let func = rebuilt.finish(old)?;
            let work = original_work(&func, graph)?;
            Ok((func, work, changed))
        })
        .collect()
}

fn original_work(
    func: &Func,
    graph: &domain_index::FiniteMapGraph,
) -> Result<FiniteMapWork, String> {
    let mut pending = func
        .results
        .iter()
        .copied()
        .map(|value| (value, false))
        .collect::<Vec<_>>();
    let mut seen = std::collections::HashSet::new();
    let mut work = FiniteMapWork::default();
    while let Some((value, before)) = pending.pop() {
        if !seen.insert((value, before)) {
            continue;
        }
        let Def::NodeResult { node, .. } = func.values[value.index()].def else {
            continue;
        };
        let node = &func.nodes[node];
        if node.op.as_any().is::<Add>()
            || node.op.as_any().is::<Subtract>()
            || node.op.as_any().is::<Multiply>()
            || node.op.as_any().is::<Negate>()
            || node.op.as_any().is::<Select>()
        {
            let instances = egraph::instances(func.value_shape(value), graph)?;
            let total = if before {
                &mut work.before_accumulation
            } else {
                &mut work.after_accumulation
            };
            *total = total.saturating_add(instances);
        }
        let before = before || node.op.as_any().is::<SumAlongMap>();
        pending.extend(node.inputs.iter().copied().map(|value| (value, before)));
    }
    Ok(work)
}

/// Arithmetic instances in the reachable SSA union, with no placement multiplier.
fn physical_work(func: &Func, graph: &domain_index::FiniteMapGraph) -> Result<u128, String> {
    let mut pending = func.results.to_vec();
    let mut seen = std::collections::HashSet::new();
    let mut work = 0u128;
    while let Some(value) = pending.pop() {
        if !seen.insert(value) {
            continue;
        }
        let Def::NodeResult { node, .. } = func.values[value.index()].def else {
            continue;
        };
        let node = &func.nodes[node];
        if node.op.as_any().is::<Add>()
            || node.op.as_any().is::<Subtract>()
            || node.op.as_any().is::<Multiply>()
            || node.op.as_any().is::<Negate>()
            || node.op.as_any().is::<Select>()
        {
            let instances = egraph::instances(func.value_shape(value), graph)?;
            work += u128::from(instances);
        }
        pending.extend(node.inputs.iter().copied());
    }
    Ok(work)
}

fn insert_partial_module_extraction(
    frontier: &mut Vec<PartialModuleExtraction>,
    extraction: PartialModuleExtraction,
) {
    if frontier.iter().any(|existing| {
        existing.work.dominates(extraction.work)
            && existing.physical_work <= extraction.physical_work
    }) {
        return;
    }
    frontier.retain(|existing| {
        !(extraction.work.dominates(existing.work)
            && extraction.physical_work <= existing.physical_work)
    });
    frontier.push(extraction);
    frontier.sort_by_key(|extraction| {
        (
            extraction.physical_work,
            extraction.work.before_accumulation,
            extraction.work.after_accumulation,
        )
    });
    if frontier.len() > MAX_FRONTIER {
        let extractions = std::mem::take(frontier);
        let last = extractions.len() - 1;
        frontier.extend(
            (0..MAX_FRONTIER)
                .map(|position| extractions[position * last / (MAX_FRONTIER - 1)].clone()),
        );
    }
}

#[cfg(test)]
pub(in crate::optimize::map_reduce::rewrite) fn expression(
    func: &Func,
    value: ValueId,
    _nodes: &mut usize,
) -> Result<ExprTree, String> {
    let source = extraction::expression_dag::import(func)?
        .ok_or("source DAG exceeded its physical bound")?;
    Ok(source
        .dag
        .materialize(source.values[value.index()].ok_or("value is outside the supported DAG")?))
}

#[cfg(test)]
mod cost_tests {
    use crate::builder::ModuleBuilder;
    use crate::dtype::DType;
    use crate::optimize::{
        DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK, DEFAULT_MAX_ALGEBRAIC_WORK,
        DEFAULT_MAX_POLYNOMIAL_TERMS,
    };
    use crate::shape::Shape;

    use super::*;

    #[test]
    fn exact_partial_contribution_ratio_prices_both_sides_without_rounding() {
        let ratio = PartialContributionRatio::new(8, 3).unwrap();

        assert_eq!(
            ratio.scaled_work(FiniteMapWork {
                before_accumulation: 8,
                after_accumulation: 0,
            }),
            ratio.scaled_work(FiniteMapWork {
                before_accumulation: 0,
                after_accumulation: 3,
            })
        );
    }

    #[test]
    fn module_frontier_builds_many_functions_in_linear_space() {
        let mut graph = domain_index::FiniteMapGraph::default();
        let source = graph.add_set(1).unwrap();
        let target = graph.add_set(1).unwrap();
        graph.add_map(source, target).unwrap();
        let mut builder = ModuleBuilder::with_finite_map_graph(graph);
        let mut entry = None;
        for index in 0..20_000 {
            let name = format!("func_{index}");
            builder.begin_func(&name);
            let value = builder.param(Shape::scalar(DType::F32));
            entry = Some(builder.end_func(vec![value]));
        }
        let module = builder.finish(entry.unwrap()).unwrap();

        let extractions = extract_module_frontier(
            &module,
            DEFAULT_MAX_POLYNOMIAL_TERMS,
            &mut AlgebraicWorkBudget::new(
                DEFAULT_MAX_ALGEBRAIC_WORK,
                DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
            ),
        )
        .unwrap();

        assert_eq!(extractions.len(), 1);
        assert_eq!(extractions[0].alternative.module.funcs.len(), 20_000);
    }
}
