// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The README is the crate documentation, so `cargo test --doc` compiles and
// runs its example.
#![doc = include_str!("../README.md")]

#[cfg(test)]
macro_rules! loom_program {
    ($($fields:tt)*) => {
        crate::LiftedProgram::from_request_unchecked(
            crate::FlatProgram { $($fields)* },
        )
    };
}

mod backend;
mod domain;
mod eval;
mod frontier;
mod lift;
pub mod profile;
mod program;
mod progress;
mod scalarize;
mod search;
mod shortlist;
pub use profile::CompileProfile;
pub use shortlist::GpuTarget;

use domain_index::{AlignedView, Domain, IndexMap};
use std::io::Write;
use std::num::NonZeroUsize;
use std::time::Duration;
use xir::dtype::DType;
use xir::ir::Module;

pub use backend::compiler::CompiledProgram;
use program::LiftedProgram;

fn watchdog_event(kind: &str, phase: &str) {
    let path = std::env::var_os("LOOM_WATCHDOG_PATH");
    let Some(path) = path else {
        return;
    };
    if let Ok(mut events) = std::fs::OpenOptions::new().append(true).open(path) {
        let _ = writeln!(events, "{kind}\t{phase}");
    }
}

/// A 64-bit FNV-1a digest, rendered as hex. Identity for joining records
/// across processes — never a security claim.
pub(crate) fn digest(bytes: &[u8]) -> String {
    let hash = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    });
    format!("{hash:016x}")
}

/// Time one candidate visit costs the tuner, unless a caller states its own.
/// Eight production programs on one B200 spend 1.5 ms to 12 ms per admitted
/// candidate, so no single value is exact; 4 ms is the value whose predictions
/// stay closest to all eight at once. Too small a value admits a shortlist whose
/// serial candidate tuning cost exceeds the predicted realization time. This
/// is an estimate for selection, not a deadline enforced by the realizer.
const DEFAULT_TUNING_COST_PER_CANDIDATE: Duration = Duration::from_millis(4);

/// Assemblies the realizer overlaps, unless a caller states its own. The same
/// eight programs reach 1.23 to 3.60 assemblies at once — never their host's
/// core count, because units are assembled on demand in tuning order rather than
/// issued together. It is a constant and not this host's available parallelism
/// so that one source stream yields one shortlist on every machine.
const DEFAULT_ASSUMED_ASSEMBLY_WORKERS: NonZeroUsize = NonZeroUsize::new(3).unwrap();

/// Default weighted source-search size estimate limit (16 GiB), not process RAM.
const DEFAULT_SEARCH_SIZE_ESTIMATE_LIMIT: u64 = 16 * 1024 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub struct CompileOptions {
    /// Wall-clock time optional source search may take. The first viable
    /// candidate is produced regardless; `None` leaves search unbounded.
    pub search_budget: Option<Duration>,
    /// Coarse source-search memory estimate: twice reported PTX bytes plus
    /// 16 KiB per complete candidate. Defaults to 16 GiB; `None` disables it.
    /// Repeated units are charged again. This bounds an estimate, not process
    /// RAM: no RSS or exact allocation tracking.
    /// Checks follow source attempts and can overrun; complete plans are saved.
    pub search_size_estimate_limit: Option<u64>,
    /// Predicted realization time for a frozen source shortlist:
    /// `max(assembly / workers, serial candidate tuning cost)`, not assembly alone.
    /// The first ranked candidate and its support are retained even if they exceed it.
    /// `None` retains every source configuration, in generation order.
    pub shortlist_budget: Option<Duration>,
    /// Predicted time one shortlisted candidate costs the realizer's tuner: the
    /// host walk and resolution of its choice plus its expected on-device probe.
    /// The tuner visits candidates one after another, so this is the serial term
    /// `shortlist_budget` has to pay, and for a large shortlist it decides the
    /// count. Its own dispersion does not matter much: the budget needs the sum
    /// of many candidates, and a sum concentrates around the mean.
    pub tuning_cost_per_candidate: Duration,
    /// Assemblies the realizer is assumed to build at once, which is what makes
    /// a shared compilation cheap per candidate. Loom starts no workers; this
    /// only enters the shortlist's cost estimate.
    pub assumed_assembly_workers: NonZeroUsize,
    /// The GPU the kernels are rendered and shortlisted for. `None` reads the
    /// current device; a compile with no device must state it.
    pub target: Option<GpuTarget>,
    pub max_kernel_statements: Option<NonZeroUsize>,
    /// Exact-arithmetic work available to one XIR execution. Each algebraic
    /// frontier alternative starts with its own reserve.
    pub max_exact_work: NonZeroUsize,
    /// Exact-arithmetic work allowed for one expression.
    pub max_exact_work_per_expression: NonZeroUsize,
    /// Maximum scratch bytes any implementation candidate may require. A
    /// candidate above the bound is removed before it can enter a recipe
    /// choice; `None` leaves workspace unconstrained.
    pub max_workspace_bytes: Option<usize>,
    /// Keep every rendered kernel's PTX in the compile profile. The text is
    /// many times the size of the recipe it produces, so only a caller that
    /// will read it asks for it.
    pub keep_ptx: bool,
    /// Compile exactly this element order instead of crossing every eligible
    /// one. The offline order campaign needs a build whose ground-truth label
    /// is unambiguous; production leaves it unset.
    pub only_element_order: Option<ElementOrder>,
    /// Compile exactly this post-arithmetic-pruning algebraic frontier graph.
    /// Production leaves it unset; calibration uses it for ground-truth runs.
    pub only_frontier_graph: Option<usize>,
    /// Record the structural baseline — every binding's dependency set and
    /// every partition boundary — in the compile profile. Both cost a second
    /// symbolic expansion per graph, so only a caller comparing structure
    /// across a lowering change asks for them.
    pub record_dependencies_and_partitions: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            search_budget: None,
            search_size_estimate_limit: Some(DEFAULT_SEARCH_SIZE_ESTIMATE_LIMIT),
            shortlist_budget: Some(Duration::from_secs(600)),
            tuning_cost_per_candidate: DEFAULT_TUNING_COST_PER_CANDIDATE,
            assumed_assembly_workers: DEFAULT_ASSUMED_ASSEMBLY_WORKERS,
            target: None,
            max_kernel_statements: None,
            max_exact_work: NonZeroUsize::new(xir_optimize::DEFAULT_MAX_ALGEBRAIC_WORK)
                .expect("xir_optimize's default exact work is positive"),
            max_exact_work_per_expression: NonZeroUsize::new(
                xir_optimize::DEFAULT_MAX_ALGEBRAIC_EXPRESSION_WORK,
            )
            .expect("xir_optimize's default expression work is positive"),
            max_workspace_bytes: None,
            keep_ptx: false,
            only_element_order: None,
            only_frontier_graph: None,
            record_dependencies_and_partitions: false,
        }
    }
}

impl CompileOptions {
    fn validate(self) -> Result<Self, String> {
        if let Some(target) = self.target {
            target.validate()?;
        }
        if self.search_budget == Some(Duration::ZERO) {
            return Err("loom search_budget must be positive".into());
        }
        Ok(self)
    }
}

/// Whether an operand holds one value per lane or one value shared by all lanes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lanes {
    /// A separate value for each of the program's `lane_count` lanes.
    PerLane,
    /// One value for every lane: an input broadcast across lanes, or an output
    /// summed over them.
    Shared,
}

#[derive(Clone, Debug)]
pub struct Operand {
    pub dtype: DType,
    /// The shape the function sees for one element.
    pub element_shape: Vec<i64>,
    pub lanes: Lanes,
    /// The domain indexing each axis, as an index into the program's
    /// `graph.domains`, or `None` for an axis indexed by no domain.
    pub axis_domains: Vec<Option<usize>>,
}

/// A scalar lane program placed over a domain graph: what [`compile`] turns
/// into kernels.
#[derive(Clone, Debug)]
pub struct Program {
    pub name: String,
    /// The function, written for one element. Its parameters are `inputs` and
    /// its results are `outputs`, in order.
    pub module: Module,
    /// Lanes each per-lane operand carries on its hidden last axis.
    pub lane_count: NonZeroUsize,
    /// The domains the operands are indexed by, and the maps between them.
    pub graph: domain_index::Graph,
    pub inputs: Vec<Operand>,
    pub outputs: Vec<Operand>,
}

pub use backend::ElementOrder;
pub use search::{compile, stream_candidates};

/// An operand in the backend's flat form: `lane_extent` is 1 or the program's
/// lane count, and each axis domain is an index or `-1`.
#[derive(Clone, Debug)]
pub(crate) struct FlatOperand {
    pub(crate) dtype: DType,
    pub(crate) element_shape: Vec<i64>,
    pub(crate) lane_extent: i64,
    pub(crate) axis_domains: Vec<i32>,
}

/// A [`Program`] in the backend's flat form: inputs and outputs in one list,
/// split at `num_inputs`, and the graph's arrows as separate lists.
#[derive(Clone, Debug)]
pub(crate) struct FlatProgram {
    pub(crate) module: Module,
    pub(crate) lane_count: i64,
    pub(crate) domains: Vec<Domain>,
    pub(crate) index_maps: Vec<IndexMap>,
    pub(crate) num_axes: usize,
    pub(crate) operands: Vec<FlatOperand>,
    pub(crate) num_inputs: usize,
    pub(crate) available_views: Vec<AlignedView>,
    pub(crate) name: String,
}

impl TryFrom<Program> for FlatProgram {
    type Error = String;

    fn try_from(program: Program) -> Result<Self, Self::Error> {
        let lane_count = i64::try_from(program.lane_count.get())
            .map_err(|_| format!("loom lane_count {} overflows i64", program.lane_count))?;
        let num_axes = program
            .inputs
            .iter()
            .chain(&program.outputs)
            .next()
            .map_or(0, |operand| operand.axis_domains.len());
        let num_inputs = program.inputs.len();
        let operands = program
            .inputs
            .into_iter()
            .chain(program.outputs)
            .map(|operand| {
                let axis_domains = operand
                    .axis_domains
                    .iter()
                    .map(|domain| match domain {
                        None => Ok(-1),
                        Some(domain) => i32::try_from(*domain)
                            .map_err(|_| format!("loom domain index {domain} overflows i32")),
                    })
                    .collect::<Result<_, String>>()?;
                Ok(FlatOperand {
                    dtype: operand.dtype,
                    element_shape: operand.element_shape,
                    lane_extent: match operand.lanes {
                        Lanes::PerLane => lane_count,
                        Lanes::Shared => 1,
                    },
                    axis_domains,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(FlatProgram {
            module: program.module,
            lane_count,
            domains: program.graph.domains,
            index_maps: program.graph.index_maps,
            num_axes,
            operands,
            num_inputs,
            available_views: program.graph.aligned_views,
            name: program.name,
        })
    }
}

/// Replace every operation that states a fixed number of applications of a
/// region — a sort, a loop whose count constant propagation reads — with the
/// circuit it denotes. This runs before anything else because every later stage
/// reads a program as a circuit, and it leaves a program holding no such
/// operation exactly as it was.
fn expand_regions(module: &mut Module) -> Result<(), String> {
    xir_optimize::PassManager::new([xir_optimize::Pass::UnrollStaticRegions])
        .run(module)
        .map(|_| ())
        .map_err(|error| format!("loom region expansion: {error}"))
}

/// The XIR optimization pipeline: the full reassociating pipeline, which
/// rewrites floating arithmetic exactly over the algebraic numbers a program's
/// coefficients live in. Loom scalarizes to fuse, and only exact rewriting
/// shrinks a scalarized cone back to a size a kernel can hold, so this is the
/// one pipeline rather than a choice among several.
fn optimization_pipeline(options: CompileOptions) -> xir_optimize::PassManager {
    xir_optimize::algebraic_pipeline()
        .with_max_algebraic_work(options.max_exact_work.get())
        .with_max_algebraic_expression_work(options.max_exact_work_per_expression.get())
}

#[cfg(test)]
fn lower_program(request: FlatProgram) -> Result<LiftedProgram, String> {
    lower_program_frontier(request)?
        .into_iter()
        .min_by_key(|program| program.finite_map_work().instantiate(1))
        .ok_or_else(|| "loom algebraic lowering produced no XIR alternative".into())
}

#[cfg(test)]
fn lower_program_frontier(request: FlatProgram) -> Result<Vec<LiftedProgram>, String> {
    frontier::lower_program_frontier_with_options(request, CompileOptions::default())
        .map(|(programs, _, _)| programs)
}

fn optimize_module(
    module: &mut Module,
    options: CompileOptions,
) -> Result<xir_optimize::PipelineReport, String> {
    optimization_pipeline(options)
        .run(module)
        .map_err(|error| format!("loom XIR optimization: {error}"))
}

/// Counts live selects with exactly one provably-zero arm: the XIR structure
/// from which Loom derives launch guards and its zero-skip kernel.
fn exact_zero_select_potential(module: &Module) -> usize {
    let func = module.entry_func();
    let facts = xir::analysis::FuncFacts::analyze(func);
    func.nodes
        .iter()
        .enumerate()
        .filter(|(index, node)| {
            facts.node(*index).live
                && node.op.as_any().is::<xir::ops::Select>()
                && node
                    .inputs
                    .get(1)
                    .zip(node.inputs.get(2))
                    .is_some_and(|(&yes, &no)| {
                        facts.value(yes).exact_zero ^ facts.value(no).exact_zero
                    })
        })
        .count()
}

fn recover_map_reduce_regions(module: &mut Module) -> Result<bool, String> {
    const MIN_SCALAR_STATEMENTS: u64 = 1 << 12;
    // A region only pays when its term count stays under the work it replaces,
    // and the scalar cone here is thousands of nodes, not millions. This bounds
    // both the expansion's memory and the time spent proving a region hopeless.
    const MAX_RECOVERED_TERMS: usize = 1 << 14;

    let func = module.entry_func();
    let facts = xir::analysis::FuncFacts::analyze(func);
    let scalar_statements = func
        .nodes
        .iter()
        .enumerate()
        .filter(|(node, _)| facts.node(*node).live)
        .flat_map(|(_, node)| node.results.iter())
        .fold(0u64, |work, &result| {
            work.saturating_add(facts.value(result).statements)
        });
    if scalar_statements < MIN_SCALAR_STATEMENTS {
        return Ok(false);
    }
    let zero_skip_potential = exact_zero_select_potential(module);
    let unrecovered = (zero_skip_potential > 0).then(|| module.clone());
    let report = xir_optimize::PassManager::new([
        xir_optimize::Pass::RecoverMapReduceRegions,
        xir_optimize::Pass::CommonSubexpressionElimination,
        xir_optimize::Pass::DeadCodeElimination,
    ])
    .with_max_polynomial_terms(MAX_RECOVERED_TERMS)
    .run(module)
    .map_err(|error| format!("loom XIR optimization: {error}"))?;
    let changed = report.passes.first().is_some_and(|pass| pass.changed);
    if changed
        && exact_zero_select_potential(module) == 0
        && let Some(unrecovered) = unrecovered
    {
        // Recovery's local work model does not price the execution-level
        // zero-skip optimization, so preserve the form that still exposes it.
        *module = unrecovered;
        return Ok(false);
    }
    Ok(changed)
}

#[cfg(test)]
#[path = "tests/float_coefficients.rs"]
mod float_coefficients;

#[cfg(test)]
mod tests {
    use algebraic_real::AlgebraicReal;
    use domain_index::Domain;
    use xir::builder::ModuleBuilder;
    use xir::dtype::DType as Dtype;
    use xir::dtype::DType;
    use xir::ops::{BroadcastInDim, Constant, Gather, Literal, Reshape, Scatter, Select, Slice};
    use xir::shape::Shape;

    use super::*;

    #[test]
    fn compile_options_reject_an_empty_search_budget_and_a_bad_target() {
        let options = CompileOptions {
            search_budget: Some(std::time::Duration::ZERO),
            ..Default::default()
        };
        let error = options.validate().unwrap_err();
        assert!(error.contains("search_budget"), "{error}");
        let options = CompileOptions {
            target: Some(GpuTarget {
                sm: 0,
                ..crate::backend::compiler::test_target()
            }),
            ..Default::default()
        };
        let error = options.validate().unwrap_err();
        assert!(error.contains("compute capability"), "{error}");
    }

    fn recovery_module(guard_terms: bool, keep_independent_guard: bool) -> Module {
        const TERMS: i64 = 4096;
        let mut builder = ModuleBuilder::new();
        let main = builder.func("recoverable", |func| {
            let left = func.param(Shape::new(DType::F32, [TERMS]));
            let right = func.param(Shape::new(DType::F32, [TERMS]));
            let predicates = guard_terms.then(|| func.param(Shape::new(DType::Bool, [TERMS])));
            let zero = func.constant(Literal::zeros(Shape::scalar(DType::F32)).unwrap());
            let scalar = |func: &mut ModuleBuilder, value, element| {
                let slice = func.op1(
                    Slice {
                        start_indices: [element].into_iter().collect(),
                        limit_indices: [element + 1].into_iter().collect(),
                        strides: [1].into_iter().collect(),
                    },
                    &[value],
                );
                func.op1(
                    Reshape {
                        out_dims: Default::default(),
                    },
                    &[slice],
                )
            };
            let mut products = (0..TERMS)
                .map(|term| {
                    let left = scalar(func, left, term);
                    let right = scalar(func, right, term);
                    let product = func.multiply(left, right);
                    predicates.map_or(product, |predicates| {
                        let predicate = scalar(func, predicates, term);
                        func.op1(Select, &[predicate, product, zero])
                    })
                })
                .collect::<Vec<_>>();
            while products.len() > 1 {
                products = products
                    .chunks(2)
                    .map(|pair| match pair {
                        [left, right] => func.add(*left, *right),
                        [value] => *value,
                        _ => unreachable!("chunks of two"),
                    })
                    .collect();
            }
            if keep_independent_guard {
                let predicate = func.param(Shape::scalar(DType::Bool));
                let value = func.param(Shape::scalar(DType::F32));
                products.push(func.op1(Select, &[predicate, value, zero]));
            }
            products
        });
        builder.finish(main).unwrap()
    }

    #[test]
    fn recovery_is_rejected_when_it_loses_exact_zero_selects() {
        let mut module = recovery_module(true, false);
        let potential = exact_zero_select_potential(&module);

        assert!(potential > 0);
        assert!(!recover_map_reduce_regions(&mut module).unwrap());
        assert_eq!(exact_zero_select_potential(&module), potential);
        assert!(
            !module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<Gather>() || node.op.as_any().is::<Scatter>())
        );
    }

    #[test]
    fn recovery_is_kept_when_it_preserves_exact_zero_selects() {
        let mut module = recovery_module(false, true);
        let potential = exact_zero_select_potential(&module);

        assert_eq!(potential, 1);
        assert!(recover_map_reduce_regions(&mut module).unwrap());
        assert_eq!(exact_zero_select_potential(&module), potential);
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<Gather>())
        );
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<Scatter>())
        );
    }

    #[test]
    fn recovery_is_kept_when_some_exact_zero_selects_survive() {
        let mut module = recovery_module(true, true);
        let potential = exact_zero_select_potential(&module);

        assert!(potential > 1);
        assert!(recover_map_reduce_regions(&mut module).unwrap());
        assert_eq!(exact_zero_select_potential(&module), 1);
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<Gather>())
        );
        assert!(
            module
                .entry_func()
                .nodes
                .iter()
                .any(|node| node.op.as_any().is::<Scatter>())
        );
    }

    #[test]
    fn lower_program_optimizes_xir_before_backend_planning() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let input = builder.param(Shape::new(DType::F32, [4]));
        let identity = builder.op1(
            Reshape {
                out_dims: [4].into_iter().collect(),
            },
            &[input],
        );
        let _dead = builder.add(input, input);
        let main = builder.end_func(vec![identity]);
        let module = builder.finish(main).unwrap();
        let binding = || FlatOperand {
            dtype: Dtype::F32,
            element_shape: vec![4],
            lane_extent: 8,
            axis_domains: vec![0],
        };
        let request = FlatProgram {
            module,
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding()],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "optimized".into(),
        };

        let program = lower_program(request).unwrap();

        let func = program.module.entry_func();
        let mnemonics = func
            .nodes
            .iter()
            .map(|node| node.op.mnemonic())
            .collect::<Vec<_>>();
        assert!(!mnemonics.contains(&"add"), "{mnemonics:?}");
        assert!(!mnemonics.contains(&"reshape"), "{mnemonics:?}");
    }

    #[test]
    fn multiplication_by_an_exact_zero_cancels() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let x = builder.param(Shape::new(DType::F32, [4]));
        let zero = builder.constant(Literal::new(Shape::new(DType::F32, [4]), vec![0u8; 16]));
        let product = builder.multiply(x, zero);
        let main = builder.end_func(vec![product]);
        let module = builder.finish(main).unwrap();
        let binding = || FlatOperand {
            dtype: Dtype::F32,
            element_shape: vec![4],
            lane_extent: 8,
            axis_domains: vec![0],
        };
        let request = FlatProgram {
            module,
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding()],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "cancelled".into(),
        };

        let program = lower_program(request).unwrap();

        let has_multiply = program
            .module
            .entry_func()
            .nodes
            .iter()
            .any(|node| node.op.mnemonic() == "multiply");
        assert!(!has_multiply);
    }

    #[test]
    fn algebraic_pipeline_combines_exact_commutative_monomials() {
        let root = AlgebraicReal::sqrt(2u32);
        let mut builder = ModuleBuilder::new();
        let main = builder.func("main", |func| {
            let shape = Shape::scalar(DType::F32);
            let x = func.param(shape.clone());
            let y = func.param(shape.clone());
            let a = func.constant(Literal::exact(shape.clone(), vec![root.clone()]).unwrap());
            let b = func.constant(Literal::exact(shape, vec![root.clone()]).unwrap());
            let ax = func.multiply(a, x);
            let by = func.multiply(b, y);
            let xy = func.multiply(ax, y);
            let yx = func.multiply(by, x);
            vec![func.add(xy, yx)]
        });
        let mut module = builder.finish(main).unwrap();

        optimization_pipeline(CompileOptions::default())
            .run(&mut module)
            .unwrap();

        let func = module.entry_func();
        assert_eq!(func.nodes.len(), 3);
        assert!(func.nodes.iter().all(|node| node.op.mnemonic() != "add"));
        let exact = func
            .nodes
            .iter()
            .find_map(|node| {
                node.op
                    .as_any()
                    .downcast_ref::<Constant>()
                    .and_then(|constant| constant.literal.exact.as_ref())
            })
            .unwrap();
        assert_eq!(exact, &vec![&root * &AlgebraicReal::integer(2)]);
    }

    #[test]
    fn a_large_result_is_scalarized_before_operation_packing() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let shape = Shape::new(DType::F32, [17]);
        let left = builder.param(shape.clone());
        let right = builder.param(shape);
        let output = builder.add(left, right);
        let main = builder.end_func(vec![output]);
        let module = builder.finish(main).unwrap();
        let binding = || FlatOperand {
            dtype: Dtype::F32,
            element_shape: vec![17],
            lane_extent: 8,
            axis_domains: vec![0],
        };
        let request = FlatProgram {
            module,
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding(), binding()],
            num_inputs: 2,
            available_views: Vec::new(),
            name: "bounded_frontier".into(),
        };

        let program = lower_program(request).unwrap();
        let func = program.module.entry_func();

        assert!(func.nodes.iter().any(|node| {
            node.op.mnemonic() == "add" && func.value_shape(node.results[0]).is_scalar()
        }));
        assert!(!func.nodes.iter().any(|node| {
            node.op.mnemonic() == "add" && func.value_shape(node.results[0]).dims.as_slice() == [17]
        }));
    }

    #[test]
    fn lower_program_folds_constant_tensor_operations() {
        let mut builder = ModuleBuilder::new();
        builder.begin_func("main");
        let _input = builder.param(Shape::new(DType::F32, [4]));
        let scalar = builder.constant(Literal::new(
            Shape::scalar(DType::F32),
            2.0f32.to_le_bytes().to_vec(),
        ));
        let output = builder.op1(
            BroadcastInDim {
                out_dims: [4].into_iter().collect(),
                broadcast_dimensions: Default::default(),
            },
            &[scalar],
        );
        let main = builder.end_func(vec![output]);
        let module = builder.finish(main).unwrap();
        let binding = || FlatOperand {
            dtype: Dtype::F32,
            element_shape: vec![4],
            lane_extent: 8,
            axis_domains: vec![0],
        };
        let request = FlatProgram {
            module,
            lane_count: 8,
            domains: vec![Domain { extent: 2 }],
            index_maps: Vec::new(),
            num_axes: 1,
            operands: vec![binding(), binding()],
            num_inputs: 1,
            available_views: Vec::new(),
            name: "constant_folded".into(),
        };

        let program = lower_program(request).unwrap();

        let nodes = &program.module.entry_func().nodes;
        assert_eq!(nodes.len(), 1);
        let constant = nodes[0].op.as_any().downcast_ref::<Constant>().unwrap();
        assert_eq!(constant.literal.data, 2.0f32.to_le_bytes().repeat(4));
    }
}
