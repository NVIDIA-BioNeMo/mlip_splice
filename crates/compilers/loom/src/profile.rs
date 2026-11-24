// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a compile reports about itself.
//!
//! Every record here describes a decision the compiler took, in terms a reader
//! outside the compiler can join: an arm by its position in the compiled choice,
//! a source unit by its entry, an output by its index. Nothing here steers a
//! compile; a field a decision reads belongs in the decision's own module.

use std::time::Duration;

/// One candidate element order, described entirely before ptxas sees it: the
/// key/value pairs are the summary a pre-compilation decision may read.
#[derive(Clone, Debug)]
pub struct ElementOrderSummary {
    pub graph: usize,
    pub order: &'static str,
    /// This order's share of describing every candidate. The orders are
    /// summarized together from one symbolic expansion, so a decision that
    /// reads all of them pays the sum of these.
    pub decision_ms: f64,
    pub summary: Vec<(&'static str, u64)>,
}

/// One source unit the compiler produced: which arm asked for it, what it is,
/// and how the recipe asks for its image. Joined to its arm by `order` and
/// `statement_cap`, never by parsing the entry name.
///
/// Registers and spill are the executor's to measure — it is the one that
/// assembles — so they are absent here; `assembly_mode` and `register_cap` are the
/// request this compile states.
#[derive(Clone, Debug)]
pub struct Kernel {
    pub graph: usize,
    pub order: &'static str,
    pub statement_cap: usize,
    pub placement: String,
    pub entry: String,
    pub flavor: &'static str,
    pub part: usize,
    pub slot: usize,
    pub register_cap: Option<u32>,
    pub assembly_mode: &'static str,
    pub statements: usize,
    /// Additions one term passes through inside one thread of this kernel, read
    /// off its arena; `None` when the analysis finds no bound.
    pub addition_chain_within_thread: Option<u64>,
    /// Whether this kernel then folds thread results into shared memory cells,
    /// so that its launch geometry extends the chain further.
    pub addition_chain_folds_across_threads: bool,
    pub ptx_bytes: usize,
    pub ptx_hash: String,
    /// The rendered PTX, kept only when the compile asked for it.
    pub ptx: Option<String>,
    pub admitted: bool,
}

/// One arm of the cap × order cross, in the position it takes in the compiled
/// whole-recipe choice. Entry names do not distinguish the arms — the same
/// kernel name appears under either order — so this list is the only honest
/// way to read an arm's identity back off a tune report.
#[derive(Clone, Debug)]
pub struct Arm {
    pub graph: usize,
    pub order: &'static str,
    pub statement_cap: usize,
    pub plan_clusters: bool,
}

/// One algebraic frontier graph and the pre-assembly quantities that describe
/// it. `recipe_arm` is its first emitted source index, before shortlisting;
/// `None` means no complete source configuration was emitted for this graph.
#[derive(Clone, Debug)]
pub struct FrontierGraph {
    pub graph: usize,
    pub frontier_size: usize,
    pub recipe_arm: Option<usize>,
    pub before_accumulation: u64,
    pub after_accumulation: u64,
    pub estimated_statements: u64,
    pub register_pressure: u64,
    pub hoistable_computed_values: usize,
}

/// One structural candidate after launch planning and PTX rendering, before
/// ptxas feedback. The record joins its algebraic graph and compile arm
/// explicitly; no kernel name parsing is required.
#[derive(Clone, Debug)]
pub struct PlannedCandidate {
    pub graph: usize,
    pub order: &'static str,
    pub statement_cap: usize,
    pub plan_clusters: bool,
    pub placement: String,
    pub candidate: usize,
    pub role: &'static str,
    pub origin: &'static str,
    pub producer_launches: usize,
    pub consumer_launches: usize,
    pub total_statements: usize,
    pub max_statements: usize,
    pub materialized_bytes: usize,
    pub max_closure_statements: Option<usize>,
    pub max_weighted_footprint: Option<usize>,
    pub duplicated_closure_statements: Option<usize>,
    pub duplicated_input_loads: Option<usize>,
    pub distinct_input_loads: Option<usize>,
    pub described_arrangements: usize,
    pub geometry_arrangements: usize,
    pub zero_skip_arrangements: usize,
    pub lane_hoist_arrangements: usize,
    pub register_capped_arrangements: usize,
    pub min_positions_per_block: usize,
    pub max_positions_per_block: usize,
    pub max_io_lanes: usize,
    pub max_warps: usize,
    pub max_shards: usize,
    pub max_passes: usize,
    pub fiber_bridge: bool,
}

/// One output's fold by key in one whole-recipe arm. `admitted` gives the
/// counts that arm's structural candidate could take: the baseline bucket
/// ladder, or `[1]` for an alternative plan that folds directly. `cap` and
/// `cap_bound` say what stops the baseline ladder, and the executor's report
/// says which arm won.
#[derive(Clone, Debug)]
pub struct FoldBuckets {
    pub graph: usize,
    pub placement: String,
    /// Position of the arm that takes this count in the recipe's choice.
    pub arm: usize,
    pub output: usize,
    /// `unique` when no two positions share an address, `colliding` otherwise.
    pub key: &'static str,
    /// Values one address receives over the whole operation.
    pub collisions: i64,
    /// Copies of the keyspace this arm folds into.
    pub buckets: u32,
    /// Bucket counts this arm's structural candidate admits.
    pub admitted: Vec<u32>,
    /// Why the ladder stops where it does, and at what bound.
    pub cap: &'static str,
    pub cap_bound: i64,
    pub elements: usize,
    pub workspace_bytes: usize,
    /// `none`, `convert`, or `fold` — what closes this arm's accumulation.
    pub projection: &'static str,
    /// Element-tile widths the executor can measure for this projection.
    pub tile_widths: Vec<u32>,
    /// The projection's entry symbol, empty when the arm needs none.
    pub entry: String,
}

/// How deep the float additions reaching one output cell go, for one whole
/// plan: the exact in-kernel chains composed with the launch geometry that
/// folds them. `total` is the figure to compare plans by, and the fields it is
/// built from are kept so a reader can audit it.
///
/// A `None` says the analysis found no bound — an unbounded loop, a callee with
/// no body, or a memory reduction that never settles — and never that the
/// program has no additions.
#[derive(Clone, Debug)]
pub struct AdditionDepth {
    pub graph: usize,
    pub placement: String,
    /// Position of the arm this plan takes in the compiled choice, which is what
    /// the executor's tune report names a winner by; `None` when the plan was
    /// materialized but no arm took it.
    pub arm: Option<usize>,
    /// The structural candidate this plan lowers.
    pub candidate: usize,
    pub output: usize,
    /// Deepest `in-kernel chain + atomic fan-in` over the launches writing this
    /// output.
    pub within_thread: Option<u64>,
    /// Roundings the deepest atomic fan-in adds on its own.
    pub across_blocks: u64,
    /// Roundings the projection that closes this output adds, `None` when the
    /// plan needs none. A projection sums `buckets` copies, so it never adds
    /// more than `buckets - 1`.
    pub projection: Option<u64>,
    /// The whole plan's figure for this output.
    pub total: Option<u64>,
    /// Blocks that may reach one bucket copy of one cell.
    pub blocks_per_bucket: i64,
    /// Values one address receives over the whole operation.
    pub collisions: i64,
    /// Copies of the keyspace this plan folds into.
    pub buckets: u32,
}

/// What one binding of an evaluation profile reads.
///
/// The profile's body is an ordered list of `Bind` operations, so a binding is
/// named by its position in that list and its dependencies are positions in the
/// same space; an entry at or past the binding count is an input the closure
/// reaches and no binding defines, interned in a program-fixed order. The
/// dependencies are the places the binding's operand cone reaches, taken from
/// the arena's use-def edges, so a value a region carries is the value it is
/// and never one an enclosing scope happens to spell the same way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingDependencies {
    pub graph: usize,
    /// The program these bindings belong to. A frontier graph and an
    /// output-family split of it are different programs with the same graph.
    pub program: String,
    /// Position of the binding among the evaluation profile's `Bind` operations.
    pub binding: usize,
    /// The entries this binding reads, ascending.
    pub dependencies: Vec<usize>,
}

/// Where one partition boundary falls, for one arm of the cap × order cross.
///
/// The cap decides how many kernels exist, so the boundary is a property of the
/// `(program, cap, order, placement)` tuple and not of the program alone.
/// `bisections` counts the halvings `refine` needed to bring an over-cap part
/// inside it, so a part it never touched is distinguishable from one it split
/// three times.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionPart {
    pub graph: usize,
    /// The program this partition cuts up. One arm of the cap × order cross
    /// plans several: the whole program, and each output-family split of it.
    pub program: String,
    pub order: &'static str,
    pub statement_cap: usize,
    pub plan_clusters: bool,
    pub placement: String,
    /// Position of this part among the bounded parts, in launch order.
    pub part: usize,
    pub bisections: usize,
    /// Still over the cap when index map stopped, so materialization owns it.
    pub oversized: bool,
    /// One `(output, element, terms)` per element slice this part writes, in
    /// the order it writes them.
    pub slices: Vec<(usize, usize, Vec<usize>)>,
}

/// Wall-clock time a compile spent in each of its phases.
#[derive(Clone, Copy, Debug, Default)]
pub struct Timings {
    /// Simplifying the `xir` module into Loom's program.
    pub simplify: Duration,
    /// Expanding, partitioning and scheduling the program into kernels.
    pub plan: Duration,
    /// Rendering each kernel as PTX.
    pub render_ptx: Duration,
    /// Stating the compiles the recipe asks for, sources included.
    pub state_requests: Duration,
    /// Assembling the recipe from the rendered kernels.
    pub build_recipe: Duration,
}

impl std::ops::AddAssign for Timings {
    fn add_assign(&mut self, other: Self) {
        self.simplify += other.simplify;
        self.plan += other.plan;
        self.render_ptx += other.render_ptx;
        self.state_requests += other.state_requests;
        self.build_recipe += other.build_recipe;
    }
}

/// Why source generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// Every candidate the frontier offers was produced.
    Exhausted,
    /// The caller of `stream_candidates` asked to stop.
    CallerStopped,
    /// `CompileOptions::search_budget` ran out.
    SearchBudget,
    /// `CompileOptions::search_size_estimate_limit` was reached.
    SearchSizeEstimateLimit,
}

impl StopReason {
    /// The name Python reports this reason by.
    pub fn name(self) -> &'static str {
        match self {
            StopReason::Exhausted => "exhausted",
            StopReason::CallerStopped => "caller_stopped",
            StopReason::SearchBudget => "search_budget",
            StopReason::SearchSizeEstimateLimit => "search_size_estimate_limit",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompileProfile {
    pub stop_reason: Option<StopReason>,
    /// Weighted search volume: twice reported PTX bytes plus 16 KiB per complete candidate.
    pub search_size_estimate: u64,
    /// Cumulative source time after each complete candidate was saved.
    pub candidate_elapsed: Vec<Duration>,
    /// Optional source attempts refused before a budget checkpoint stopped work.
    pub refused_attempts: usize,
    pub timings: Timings,
    /// Exact work consumed across the independently budgeted XIR executions.
    pub exact_work: usize,
    /// Expressions left unfolded because either exact-work limit was reached.
    pub abandoned_exact_expressions: usize,
    /// Description of every element order the compiler could take, before any
    /// kernel is rendered.
    pub element_orders: Vec<ElementOrderSummary>,
    /// Original source indices in the final recipe's choice order. Empty for
    /// streaming capture, which does not construct a shortlist.
    pub shortlisted_candidates: Vec<usize>,
    /// Predicted realization milliseconds: `max(assembly / workers, serial
    /// candidate tuning cost)`. Assembly/link estimates count unique units,
    /// including entry witnesses and guards. This is the quantity
    /// `CompileOptions::shortlist_budget` bounds.
    /// Not computed for streaming capture or an unbounded shortlist.
    pub shortlist_estimated_ms: Option<f64>,
    /// Distinct native compile units the shortlist requires, counted once each.
    pub shortlist_compile_units: Option<usize>,
    /// The arms emitted by source generation, in source order.
    pub arms: Vec<Arm>,
    /// Algebraic frontier graphs that reached compilation.
    pub frontier_graphs: Vec<FrontierGraph>,
    /// Structural candidates exposed by each planned recipe.
    pub planned_candidates: Vec<PlannedCandidate>,
    /// Every bucket count each output's fold offers, per arm that takes it.
    pub fold_buckets: Vec<FoldBuckets>,
    /// Every source unit produced, admitted or not.
    pub kernels: Vec<Kernel>,
    /// How deep each plan's accumulation of each output goes.
    pub addition_depths: Vec<AdditionDepth>,
    /// The structural baseline, when the compile was asked for it: what every
    /// definition reads, and where every partition boundary falls.
    pub binding_dependencies: Vec<BindingDependencies>,
    pub partition_parts: Vec<PartitionPart>,
}

impl CompileProfile {
    pub(crate) fn set_graph(&mut self, graph: usize) {
        for record in &mut self.element_orders {
            record.graph = graph;
        }
        for record in &mut self.arms {
            record.graph = graph;
        }
        for record in &mut self.planned_candidates {
            record.graph = graph;
        }
        for record in &mut self.fold_buckets {
            record.graph = graph;
        }
        for record in &mut self.kernels {
            record.graph = graph;
        }
        for record in &mut self.addition_depths {
            record.graph = graph;
        }
        for record in &mut self.binding_dependencies {
            record.graph = graph;
        }
        for record in &mut self.partition_parts {
            record.graph = graph;
        }
    }

    /// Absorb one arm's profile.
    ///
    /// The arms are merged in the order they are offered, and that order is the
    /// order of the compiled choice, so an arm's position is how many arms
    /// preceded it. Fold and accumulation records are stamped with it here, at
    /// the one place that knows it. A profile that carries no arm describes a
    /// rejected plan: its fold records are dropped, while its unattributed
    /// accumulation records remain available as compiler evidence.
    pub(crate) fn merge(&mut self, other: &Self) {
        let arm = (other.arms.len() == 1).then_some(self.arms.len());
        self.candidate_elapsed.extend(&other.candidate_elapsed);
        self.refused_attempts += other.refused_attempts;
        self.timings += other.timings;
        self.exact_work += other.exact_work;
        self.abandoned_exact_expressions += other.abandoned_exact_expressions;
        self.element_orders
            .extend(other.element_orders.iter().cloned());
        self.arms.extend(other.arms.iter().cloned());
        self.frontier_graphs
            .extend(other.frontier_graphs.iter().cloned());
        self.planned_candidates
            .extend(other.planned_candidates.iter().cloned());
        if let Some(arm) = arm {
            self.fold_buckets.extend(
                other
                    .fold_buckets
                    .iter()
                    .cloned()
                    .map(|record| crate::profile::FoldBuckets { arm, ..record }),
            );
        }
        self.kernels.extend(other.kernels.iter().cloned());
        self.addition_depths.extend(
            other
                .addition_depths
                .iter()
                .cloned()
                .map(|record| crate::profile::AdditionDepth { arm, ..record }),
        );
        // Every remaining record describes something one attempt did, so
        // merging them is concatenation. A dependency set describes a program,
        // and every arm that plans that program records the same one — so the
        // merged profile keeps the first and drops the repeats, which is what
        // makes a definition position mean a position in the program's binding
        // list.
        let described: std::collections::BTreeSet<(usize, &str)> = self
            .binding_dependencies
            .iter()
            .map(|record| (record.graph, record.program.as_str()))
            .collect();
        let fresh = other
            .binding_dependencies
            .iter()
            .filter(|record| !described.contains(&(record.graph, record.program.as_str())))
            .cloned()
            .collect::<Vec<_>>();
        self.binding_dependencies.extend(fresh);
        self.partition_parts
            .extend(other.partition_parts.iter().cloned());
    }
}
