// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What holds of every recipe, checked on recipes nobody wrote by hand.
//!
//! The generator is deterministic: a recipe is a value of its seed, so a law
//! that fails names the seed that broke it and the case is reproduced by
//! reading it back. Seeds cover multi-node graphs with real dependency edges,
//! sibling choices, and choices nested inside choices.
//!
//! The family here keeps its calls' effects in the call, which is what a family
//! that means to edit built recipes does — the laws about substitution and
//! replacement are laws about such a family. It is otherwise the executable
//! family: the same contract, the same contexts, the same launches.

use super::*;
use crate::algebra::computation::DeclaredEffects;
use crate::arg::Arg;
use crate::buffer::BufRef;
use crate::code::{
    CompileRequest, KernelModule, PtxCompileOptions, PtxTarget, ResourceBounds, SourceImage,
    SourceKind,
};
use crate::handles::{Buf, Writable};
use crate::signature::RecipeSignature;

/// Every port and workspace buffer in a generated recipe spans this, so an
/// access at offset zero of any width up to it is in range wherever it lands.
const EXTENT: usize = 256;

/// A launch that carries what it does, so a recipe holding it can still be
/// edited by effect after it was built.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DeclaringLaunch {
    launch: KernelLaunch,
    effects: Effects,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Declared;

impl Addresses for DeclaringLaunch {
    fn addresses(&self, visit: &mut dyn FnMut(crate::buffer::Address<BufRef>)) {
        self.launch.addresses(visit);
    }

    fn rebind(
        &mut self,
        bind: &mut dyn FnMut(crate::buffer::Address<BufRef>) -> crate::buffer::Address<BufRef>,
    ) {
        let mut moved = Vec::new();
        self.launch.rebind(&mut |address| {
            let landed = bind(address);
            moved.push((address.buf, landed.buf));
            landed
        });
        self.effects = Effects::of(self.effects.iter().map(|(buf, access)| {
            let landed = moved
                .iter()
                .find(|(from, _)| *from == buf)
                .map(|(_, to)| *to)
                .unwrap_or(buf);
            (landed, access)
        }));
    }
}

impl Computation for Declared {
    type Authorization = ();

    type Contract = RecipeSignature;
    type RootContext = CompileRequests;
    type BodyContext = Vec<KernelModule>;
    type Call = DeclaringLaunch;
    type RootRemap = RequestRemap;
    type BodyRemap = ModuleBase;
    type Refusal = ExecutableRefusal;

    fn merge_root_context(
        into: &mut CompileRequests,
        incoming: CompileRequests,
    ) -> Self::RootRemap {
        Executable::merge_root_context(into, incoming)
    }

    fn merge_body_context(
        into: &mut Vec<KernelModule>,
        incoming: Vec<KernelModule>,
    ) -> Self::BodyRemap {
        Executable::merge_body_context(into, incoming)
    }

    fn rebase_body_context(context: &mut Vec<KernelModule>, remap: &RequestRemap) {
        Executable::rebase_body_context(context, remap);
    }

    fn rebase_call(call: &mut DeclaringLaunch, remap: &ModuleBase) {
        Executable::rebase_call(&mut call.launch, remap);
    }

    fn validate_root_context(context: &CompileRequests) -> Result<(), ExecutableRefusal> {
        Executable::validate_root_context(context)
    }

    fn validate_body_context(
        context: &Vec<KernelModule>,
        root: &CompileRequests,
    ) -> Result<(), ExecutableRefusal> {
        Executable::validate_body_context(context, root)
    }

    fn validate_call(
        call: &DeclaringLaunch,
        context: &Vec<KernelModule>,
    ) -> Result<(), ExecutableRefusal> {
        Executable::validate_call(&call.launch, context)
    }

    fn launch_arity(call: &DeclaringLaunch) -> usize {
        Executable::launch_arity(&call.launch)
    }

    fn validate_geometry(
        call: &DeclaringLaunch,
        launch: usize,
        geometry: &Geometry,
    ) -> Result<(), ExecutableRefusal> {
        Executable::validate_geometry(&call.launch, launch, geometry)
    }
}

impl DeclaredEffects for Declared {
    fn call_effects(call: &DeclaringLaunch) -> Effects {
        call.effects.clone()
    }
}

/// A deterministic stream of choices.
struct Seed(u64);

impl Seed {
    fn next(&mut self) -> u64 {
        // xorshift64*: one multiply and three shifts, and the same sequence
        // everywhere, which is the only property a generator needs here.
        let mut state = self.0;
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        self.0 = state;
        state.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

fn request(ptx: &str) -> CompileRequest {
    CompileRequest::unguarded(
        vec![SourceImage {
            kind: SourceKind::Ptx,
            target: PtxTarget {
                isa: [8, 7],
                minimum_sm: 90,
            },
            blob: crate::Blob::from_vec(ptx.as_bytes().to_vec()),
        }],
        PtxCompileOptions::default(),
    )
}

fn add_module(builder: &mut RecipeBuilder<Declared>, ptx: &str) -> usize {
    let compiled = builder.root_context_mut().intern(request(ptx));
    let modules = builder.body_context_mut();
    modules.push(KernelModule::from_request(
        compiled,
        Vec::new(),
        ResourceBounds::default(),
    ));
    modules.len() - 1
}

fn declaring(
    module: usize,
    entry: &str,
    args: Vec<Arg>,
    effects: impl IntoIterator<Item = (BufRef, Access)>,
) -> DeclaringLaunch {
    DeclaringLaunch {
        launch: KernelLaunch {
            module,
            entry: entry.to_string(),
            args,
        },
        effects: Effects::of(effects),
    }
}

/// Work laid into `builder`, over one input and one output it is bound
/// through. `depth` is how many more levels of choice may still open.
fn lay_out(builder: &mut RecipeBuilder<Declared>, seed: &mut Seed, depth: usize) {
    let scratch = builder.scratch(EXTENT);
    for step in 0..2 + seed.below(3) {
        match seed.below(if depth == 0 { 3 } else { 4 }) {
            0 => {
                let module = add_module(builder, &format!("ptx {}", seed.below(3)));
                let geometry = Geometry::new([1 + seed.below(4) as u32, 1, 1], [32, 1, 1], 0);
                let call = declaring(
                    module,
                    &format!("entry_{step}"),
                    vec![
                        Arg::input(0),
                        Arg::workspace(scratch.index()),
                        Arg::output(0),
                    ],
                    [
                        (BufRef::Input(0), Access::Read),
                        (BufRef::Output(0), Access::Write),
                        (BufRef::Workspace(scratch.index()), Access::Write),
                    ],
                );
                builder
                    .call(call, vec![geometry])
                    .reads([Buf::Input(0)])
                    .writes([Buf::Output(0), Buf::Workspace(scratch)])
                    .add()
                    .expect("the statement matches what the launch is handed");
            }
            1 => {
                builder.memset(Writable::Workspace(scratch).at(0), 0, EXTENT);
            }
            2 => {
                builder.memcpy(Buf::Input(0).at(0), Writable::Output(0).at(0), EXTENT / 2);
            }
            _ => {
                let candidates = (0..1 + seed.below(2))
                    .map(|_| {
                        let mut inner = RecipeBuilder::<Declared>::new(1, 1);
                        lay_out(&mut inner, seed, depth - 1);
                        inner.finish_fragment()
                    })
                    .collect();
                builder
                    .choice(candidates)
                    .inputs([Buf::Input(0)])
                    .does([Buf::Input(0)], Access::Read)
                    .outputs([Buf::Output(0)])
                    .does([Buf::Output(0)], Access::Write)
                    .add()
                    .expect("the choice answers for what its candidates do");
            }
        }
    }
}

fn generate(seed: u64) -> Recipe<Declared> {
    proved(seed).into_recipe()
}

/// The same, with what it does at its ports still on it.
fn proved(seed: u64) -> ProvedRecipe<Declared> {
    let mut seed = Seed(seed | 1);
    let mut builder = RecipeBuilder::<Declared>::new(1, 1);
    lay_out(&mut builder, &mut seed, 2);
    builder
        .finish(RecipeSignature::new(vec![EXTENT], vec![EXTENT]))
        .expect("a generated recipe is built inside the ports it states")
}

/// Enough seeds to reach every shape the generator can produce, and few enough
/// that the laws stay a fast check.
fn generated() -> impl Iterator<Item = (u64, Recipe<Declared>)> {
    (1..=64u64).map(|seed| (seed, generate(seed)))
}

/// The first generated recipe that holds a computation at all, for a law about
/// what happens to one. Some seeds lay out only memory work.
fn with_a_call() -> (Recipe<Declared>, NodePath) {
    let (_, recipe) = generated()
        .find(|(_, recipe)| !recipe.call_paths().is_empty())
        .expect("some seed lays out a computation");
    let path = recipe.call_paths().remove(0);
    (recipe, path)
}

/// The same place, named in another recipe's shape.
fn same_place(recipe: &Recipe<Declared>, path: &NodePath) -> NodePath {
    let mut at = recipe.root_path();
    for step in path.body().steps() {
        at = at.descend(step.node, step.candidate);
    }
    at.node(path.node())
}

/// What every choice this generator places declares, stated once here because
/// [`lay_out`] states it once there. A recipe keeps the edges a choice's
/// statement justified and not the statement, so a law about those edges has to
/// hold the evidence the builder held before it discharged it.
fn declared_choice_effects() -> Effects {
    Effects::of([
        (BufRef::Input(0), Access::Read),
        (BufRef::Output(0), Access::Write),
    ])
}

/// What a node does: memory work says it by being what it is, this family's
/// calls carry it, and a choice's is the evidence its placement was given.
fn stated_effects(work: &Work<Declared>) -> Effects {
    match work {
        Work::Call { call, .. } => Declared::call_effects(call),
        Work::Choice { .. } => declared_choice_effects(),
        other => other
            .memory_effects()
            .expect("every other work states its own"),
    }
}

/// One node as structure sees it: where it sits, what it waits for, how it is
/// launched, what it does, and what it binds when it offers a choice.
#[derive(Debug, PartialEq, Eq)]
struct Placed {
    depth: usize,
    node: usize,
    waits_for: Vec<usize>,
    geometry: Option<Vec<Geometry>>,
    effects: Effects,
    binding: Option<(Vec<BufRef>, Vec<BufRef>)>,
}

/// The shape of a recipe, in the terms structure is written in — everything a
/// transformation over computations must leave exactly as it found it.
fn shape(recipe: &Recipe<Declared>) -> Vec<Placed> {
    recipe
        .nodes_with_paths()
        .into_iter()
        .map(|(path, node)| Placed {
            depth: path.body().steps().len(),
            node: path.node(),
            waits_for: node.deps().iter().copied().collect(),
            geometry: match node.work() {
                Work::Call { geometry, .. } => Some(geometry.clone()),
                _ => None,
            },
            effects: stated_effects(node.work()),
            binding: match node.work() {
                Work::Choice {
                    input_binding,
                    output_binding,
                    ..
                } => Some((input_binding.clone(), output_binding.clone())),
                _ => None,
            },
        })
        .collect()
}

/// What a mapping over this family can refuse: its own reason, or a structural
/// one about the recipe it produced.
#[derive(Debug, PartialEq, Eq)]
enum Refused {
    Stated(&'static str),
    Structure(Box<RecipeError<ExecutableRefusal>>),
}

impl From<RecipeError<ExecutableRefusal>> for Refused {
    fn from(error: RecipeError<ExecutableRefusal>) -> Refused {
        Refused::Structure(Box::new(error))
    }
}

/// A mapping that changes one computation and leaves everything else exactly
/// as it stands.
struct Suffix(&'static str);

impl Mapping<Declared, Declared> for Suffix {
    type Error = Refused;

    const AUTHORIZATION: () = ();

    fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
        Ok(contract.clone())
    }

    fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
        Ok(context.clone())
    }

    fn body_context(&mut self, context: &Vec<KernelModule>) -> Result<Vec<KernelModule>, Refused> {
        Ok(context.clone())
    }

    fn call(&mut self, call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
        let mut call = call.clone();
        call.launch.entry.push_str(self.0);
        Ok(call)
    }
}

/// The two mappings applied in order, as one mapping.
struct Then<A, B>(A, B);

impl<A, B> Mapping<Declared, Declared> for Then<A, B>
where
    A: Mapping<Declared, Declared, Error = Refused>,
    B: Mapping<Declared, Declared, Error = Refused>,
{
    type Error = Refused;

    const AUTHORIZATION: () = ();

    fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
        let contract = self.0.contract(contract)?;
        self.1.contract(&contract)
    }

    fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
        let context = self.0.root_context(context)?;
        self.1.root_context(&context)
    }

    fn body_context(&mut self, context: &Vec<KernelModule>) -> Result<Vec<KernelModule>, Refused> {
        let context = self.0.body_context(context)?;
        self.1.body_context(&context)
    }

    fn call(&mut self, call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
        let call = self.0.call(call)?;
        self.1.call(&call)
    }
}

/// Mapping every part to itself is the identity on the whole recipe.
#[test]
fn mapping_nothing_changes_nothing() {
    for (seed, recipe) in generated() {
        let mapped = recipe.map(&mut Suffix("")).unwrap();
        assert_eq!(mapped, recipe, "seed {seed}");
    }
}

/// Mapping twice is mapping once with the composition. Structure is untouched
/// either way, so the two ends agree on every node, edge, geometry, effect and
/// binding — and a path taken from one names the same place in the other.
#[test]
fn mapping_composes() {
    for (seed, recipe) in generated() {
        let stepwise = recipe
            .map(&mut Suffix("_f"))
            .unwrap()
            .map(&mut Suffix("_g"))
            .unwrap();
        let composed = recipe.map(&mut Then(Suffix("_f"), Suffix("_g"))).unwrap();
        assert_eq!(stepwise, composed, "seed {seed}");
        assert_eq!(shape(&composed), shape(&recipe), "seed {seed}");
        assert_eq!(composed.shape(), recipe.shape(), "seed {seed}");
        for path in recipe.node_paths() {
            composed
                .node_at(&path)
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        }
    }
}

/// A mapping that refuses stops the whole transformation: there is no half
/// mapped recipe.
#[test]
fn a_refused_mapping_produces_no_recipe() {
    struct Refuse;

    impl Mapping<Declared, Declared> for Refuse {
        type Error = Refused;

        const AUTHORIZATION: () = ();

        fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
            Ok(contract.clone())
        }
        fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
            Ok(context.clone())
        }
        fn body_context(
            &mut self,
            context: &Vec<KernelModule>,
        ) -> Result<Vec<KernelModule>, Refused> {
            Ok(context.clone())
        }
        fn call(&mut self, _call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
            Err(Refused::Stated("no"))
        }
    }

    let (recipe, _) = with_a_call();
    assert_eq!(recipe.map(&mut Refuse).err(), Some(Refused::Stated("no")));
}

/// A proof crosses a mapping: the boundary is a fact about the structure and
/// the bindings, and mapping changes neither. What comes back can be offered as
/// a candidate, which is the whole of what a proof is for.
#[test]
fn a_proof_crosses_a_mapping_that_keeps_what_the_calls_do() {
    for seed in 1..=64u64 {
        let held = proved(seed);
        let mapped = held.map(&mut Suffix("_f")).unwrap();
        assert_eq!(mapped.boundary(), held.boundary(), "seed {seed}");
        assert_eq!(
            mapped.recipe(),
            &held.recipe().map(&mut Suffix("_f")).unwrap(),
            "seed {seed}"
        );
    }
}

/// A mapping whose calls do more than the ones they replace does not carry the
/// proof: the edges around them were derived from the statement that came
/// across, and work the statement does not answer for would be left unordered.
#[test]
fn a_mapping_whose_calls_do_more_is_refused() {
    struct AlsoWrites;

    impl Mapping<Declared, Declared> for AlsoWrites {
        type Error = Refused;

        const AUTHORIZATION: () = ();

        fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
            Ok(contract.clone())
        }
        fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
            Ok(context.clone())
        }
        fn body_context(
            &mut self,
            context: &Vec<KernelModule>,
        ) -> Result<Vec<KernelModule>, Refused> {
            Ok(context.clone())
        }

        /// The same launch, over the same addresses, now writing the output it
        /// only read before.
        fn call(&mut self, call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
            let mut call = call.clone();
            call.effects = Effects::of(
                call.effects
                    .iter()
                    .map(|(buf, access)| match buf {
                        BufRef::Output(_) => (buf, Access::Write),
                        _ => (buf, access),
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(call)
        }
    }

    // One call that reads both buffers it is handed — so the boundary that
    // comes across says exactly that, and a call that writes one of them does
    // more than the edges around it were derived from.
    let mut builder = RecipeBuilder::<Declared>::new(1, 1);
    let module = add_module(&mut builder, ".version 7.0");
    let call = declaring(
        module,
        "reads",
        vec![Arg::input(0), Arg::output(0)],
        [
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::Read),
        ],
    );
    builder
        .call(call, vec![Geometry::single()])
        .reads([Buf::Input(0), Buf::Output(0)])
        .add()
        .unwrap();
    let held = builder
        .finish(RecipeSignature::new(vec![EXTENT], vec![EXTENT]))
        .unwrap();
    assert!(matches!(
        held.map(&mut AlsoWrites),
        Err(Refused::Structure(error)) if matches!(*error, RecipeError::UndeclaredEffect { .. })
    ));
    // The recipe itself still maps: what is refused is carrying the proof.
    assert!(held.recipe().map(&mut AlsoWrites).is_ok());
}

/// The same rule holds for a fragment: a proof crosses a mapping only if the
/// work that arrives does no more, and what a refused mapping leaves behind is
/// nothing a choice or a substitution will take.
#[test]
fn a_fragments_proof_does_not_cross_a_mapping_that_does_more() {
    struct AlsoWrites;

    impl Mapping<Declared, Declared> for AlsoWrites {
        type Error = Refused;

        const AUTHORIZATION: () = ();

        fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
            Ok(contract.clone())
        }
        fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
            Ok(context.clone())
        }
        fn body_context(
            &mut self,
            context: &Vec<KernelModule>,
        ) -> Result<Vec<KernelModule>, Refused> {
            Ok(context.clone())
        }
        fn call(&mut self, call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
            let mut call = call.clone();
            call.effects = Effects::of(
                call.effects
                    .iter()
                    .map(|(buf, access)| match buf {
                        BufRef::Output(_) => (buf, Access::Write),
                        _ => (buf, access),
                    })
                    .collect::<Vec<_>>(),
            );
            Ok(call)
        }
    }

    // One call that reads both buffers it is handed, proved as such.
    let mut builder = RecipeBuilder::<Declared>::new(1, 1);
    let module = add_module(&mut builder, ".version 7.0");
    let call = declaring(
        module,
        "reads",
        vec![Arg::input(0), Arg::output(0)],
        [
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::Read),
        ],
    );
    builder
        .call(call, vec![Geometry::single()])
        .reads([Buf::Input(0), Buf::Output(0)])
        .add()
        .unwrap();
    let held = builder.finish_fragment();
    assert!(held.boundary().is_some());
    assert!(matches!(
        held.map(&mut AlsoWrites),
        Err(Refused::Structure(error)) if matches!(*error, RecipeError::UndeclaredEffect { .. })
    ));

    // And the work itself, mapped without a proof, reaches neither a choice
    // nor a substitution: what nobody proved is refused where it is offered.
    let unproved = || {
        RecipeFragment::unproved(
            held.context().clone(),
            held.body().map(&mut AlsoWrites).unwrap(),
        )
    };
    let mut host = RecipeBuilder::<Declared>::new(1, 1);
    assert!(matches!(
        host.choice(vec![unproved()])
            .inputs([Buf::Input(0)])
            .outputs([Buf::Output(0)])
            .does_what_its_candidates_do()
            .add(),
        Err(RecipeError::UnprovedFragment)
    ));
    let (recipe, path) = with_a_call();
    assert!(matches!(
        recipe.substitute(&path, unproved(), &[BufRef::Input(0)], &[BufRef::Output(0)]),
        Err(RecipeError::UnprovedFragment)
    ));
}

/// A mapping that produces work the new family refuses is refused, structurally
/// — a recipe that was sound over one family is not evidence about another.
#[test]
fn a_mapping_that_breaks_the_result_is_refused() {
    struct Unmoored;

    impl Mapping<Declared, Declared> for Unmoored {
        type Error = Refused;

        const AUTHORIZATION: () = ();

        fn contract(&mut self, contract: &RecipeSignature) -> Result<RecipeSignature, Refused> {
            Ok(contract.clone())
        }
        fn root_context(&mut self, context: &CompileRequests) -> Result<CompileRequests, Refused> {
            Ok(context.clone())
        }
        fn body_context(
            &mut self,
            _context: &Vec<KernelModule>,
        ) -> Result<Vec<KernelModule>, Refused> {
            Ok(Vec::new())
        }
        fn call(&mut self, call: &DeclaringLaunch) -> Result<DeclaringLaunch, Refused> {
            Ok(call.clone())
        }
    }

    let (recipe, _) = with_a_call();
    assert!(matches!(
        recipe.map(&mut Unmoored),
        Err(Refused::Structure(_))
    ));
}

/// A one-call fragment, and the compile it draws from.
fn one_call_fragment(ptx: &str, entry: &str) -> RecipeFragment<Declared> {
    let mut builder = RecipeBuilder::<Declared>::new(1, 1);
    let module = add_module(&mut builder, ptx);
    let call = declaring(
        module,
        entry,
        vec![Arg::input(0), Arg::output(0)],
        [
            (BufRef::Input(0), Access::Read),
            (BufRef::Output(0), Access::Write),
        ],
    );
    builder
        .call(call, vec![Geometry::single()])
        .reads([Buf::Input(0)])
        .writes([Buf::Output(0)])
        .add()
        .unwrap();
    builder.finish_fragment()
}

/// Substituting a call binds the incoming work to the buffers the host names,
/// keeps the graph around it, and leaves the recipe it came from alone.
#[test]
fn substitution_binds_and_preserves() {
    for (seed, recipe) in generated() {
        let Some(path) = recipe.call_paths().into_iter().next() else {
            continue;
        };
        let before = recipe.clone();
        let host = recipe.body_at(path.body()).unwrap().clone();
        let nodes_before = host.nodes().len();
        let waited_for: Vec<usize> = host.nodes()[path.node()].deps().iter().copied().collect();
        let dependents: Vec<usize> = (path.node() + 1..nodes_before)
            .filter(|&node| host.nodes()[node].deps().contains(&path.node()))
            .collect();

        let substituted = recipe
            .substitute(
                &path,
                one_call_fragment("substituted", "landed"),
                &[BufRef::Input(0)],
                &[BufRef::Output(0)],
            )
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));

        assert_eq!(recipe, before, "seed {seed}: the source is untouched");

        // The result names a shape of its own, so paths taken before it refuse.
        assert_eq!(
            substituted.node_at(&path),
            Err(RecipeError::PathNotOwned),
            "seed {seed}"
        );

        let landed = substituted
            .body_at(same_place(&substituted, &path).body())
            .unwrap();
        assert_eq!(
            landed.nodes().len(),
            nodes_before,
            "seed {seed}: one call became one call"
        );

        // The incoming call is bound to the host buffers the binding named, and
        // waits for exactly what the call it replaced waited for.
        let Work::Call { call, .. } = landed.nodes()[path.node()].work() else {
            panic!("seed {seed}: the substituted node is a call")
        };
        assert_eq!(call.launch.entry, "landed");
        assert_eq!(
            call.launch.args,
            vec![Arg::input(0), Arg::output(0)],
            "seed {seed}"
        );
        assert_eq!(
            call.effects,
            Effects::of([
                (BufRef::Input(0), Access::Read),
                (BufRef::Output(0), Access::Write)
            ]),
            "seed {seed}: the effects landed with the work"
        );
        assert_eq!(
            landed.nodes()[path.node()]
                .deps()
                .iter()
                .copied()
                .collect::<Vec<_>>(),
            waited_for,
            "seed {seed}"
        );
        for dependent in dependents {
            assert!(
                landed.nodes()[dependent].deps().contains(&path.node()),
                "seed {seed}: node {dependent} still waits on the region"
            );
        }
        // The incoming module landed in the host body's own table, and the
        // launch names it there.
        assert!(call.launch.module < landed.context().len(), "seed {seed}");
    }
}

/// Two substituted fragments compiling the same source hold one compile
/// request between them: merging a root context is the family's own interning,
/// and it leaves what was already there alone.
#[test]
fn substituted_fragments_sharing_a_source_share_one_request() {
    let (recipe, first) = with_a_call();
    let requests_before = recipe.context().len();

    let once = recipe
        .substitute(
            &first,
            one_call_fragment("shared", "a"),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
        )
        .unwrap();
    assert_eq!(once.context().len(), requests_before + 1);
    for (index, request) in recipe.context().as_slice().iter().enumerate() {
        assert_eq!(
            &once.context().as_slice()[index],
            request,
            "a merge moved an entry that was already there"
        );
    }

    let second = once.call_paths().remove(0);
    let twice = once
        .substitute(
            &second,
            one_call_fragment("shared", "b"),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
        )
        .unwrap();
    assert_eq!(twice.context().len(), requests_before + 1);
}

/// A fragment with no work at all removes the call and joins whatever waited
/// on it to whatever it waited for.
#[test]
fn substituting_nothing_removes_the_call() {
    let (recipe, path) = with_a_call();
    let nodes_before = recipe.body_at(path.body()).unwrap().nodes().len();

    let emptied = recipe
        .substitute(
            &path,
            RecipeBuilder::<Declared>::new(1, 1).finish_fragment(),
            &[BufRef::Input(0)],
            &[BufRef::Output(0)],
        )
        .unwrap();
    let landed = emptied.body_at(same_place(&emptied, &path).body()).unwrap();
    assert_eq!(landed.nodes().len(), nodes_before - 1);
}

/// Replacing one node's computation touches that node and nothing else — and
/// the recipe replaced from still holds what it held.
#[test]
fn replacement_never_mutates_the_source() {
    for (seed, recipe) in generated() {
        let Some(path) = recipe.call_paths().into_iter().next() else {
            continue;
        };
        let before = recipe.clone();
        let mut replacement = recipe.call_at(&path).unwrap().clone();
        replacement.launch.entry = "replaced".to_string();

        let replaced = recipe
            .replace_call(&path, replacement)
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));

        assert_eq!(recipe, before, "seed {seed}: the source is untouched");
        assert_eq!(shape(&replaced), shape(&recipe), "seed {seed}");
        assert_eq!(
            replaced.call_at(&path).unwrap().launch.entry,
            "replaced",
            "seed {seed}"
        );
    }
}

/// Every generated recipe is valid, its paths all resolve, and the workspace it
/// declares covers every way its choices can go.
#[test]
fn a_generated_recipe_proves_itself() {
    for (seed, recipe) in generated() {
        recipe
            .validate()
            .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        for path in recipe.node_paths() {
            recipe
                .node_at(&path)
                .unwrap_or_else(|error| panic!("seed {seed}: {error}"));
        }
        assert!(workspace_upper_bound(&recipe) >= EXTENT, "seed {seed}");
        for (earlier, later) in recipe.concurrent() {
            assert!(!recipe.waits_for(later).unwrap().contains(&earlier));
        }
    }
}

/// Every dependency a generated recipe states is one the two nodes' effects
/// justify: they touch a buffer that can change, and at least one of them
/// writes it. Choices included — the evidence they were placed with is the
/// evidence the edges came from.
#[test]
fn effects_justify_every_edge() {
    for (seed, recipe) in generated() {
        let nodes = recipe.body().nodes();
        for (index, node) in nodes.iter().enumerate() {
            let mine = stated_effects(node.work());
            for &dep in node.deps() {
                let theirs = stated_effects(nodes[dep].work());
                let conflicts = mine.iter().any(|(buf, access)| {
                    !matches!(buf, BufRef::Input(_))
                        && theirs
                            .access(buf)
                            .is_some_and(|other| access.writes() || other.writes())
                });
                assert!(
                    conflicts,
                    "seed {seed}: node {index} waits for {dep} with nothing between them"
                );
            }
        }
    }
}

/// The generator reaches the shapes these laws are about: several nodes, real
/// edges, sibling choices and nested ones.
#[test]
fn the_generator_reaches_nested_choices() {
    let deepest = generated()
        .map(|(_, recipe)| {
            recipe
                .node_paths()
                .iter()
                .map(|path| path.body().steps().len())
                .max()
                .unwrap_or(0)
        })
        .max()
        .unwrap();
    assert!(deepest >= 2, "a choice inside a choice, reached: {deepest}");

    let widest = generated()
        .map(|(_, recipe)| recipe.body().nodes().len())
        .max()
        .unwrap();
    assert!(widest >= 3, "several nodes in one body: {widest}");

    assert!(
        generated().any(|(_, recipe)| recipe
            .body()
            .nodes()
            .iter()
            .any(|node| !node.deps().is_empty())),
        "real dependency edges"
    );
}
