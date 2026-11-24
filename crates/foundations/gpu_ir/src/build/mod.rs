// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Building a function: the arena, owned by the thing that writes it.
//!
//! A producer holds one [`Builder`], binds values through it, and finishes with
//! the body it built. Nothing else owns the arena, so where a value was bound is
//! never in question.
//!
//! **The builder owns what it builds, and `finish_owned` moves it out.** It does
//! not borrow the thing it appends to. That is a decision about shape: a
//! borrowing builder would carry a lifetime, and every caller that pairs one
//! with a read-only context — an emitter holding a layout, say — would have to
//! reconcile two unrelated borrows at the type level, or reach for a generic
//! associated type to keep them apart. Owning costs one move at the end and
//! keeps every signature lifetime-free.

mod import;
mod owner;
mod placement;
#[cfg(test)]
mod tests;
mod value;

pub(crate) use import::Calls;
pub use import::Import;
pub use owner::{Arm, Body, Carry, Counted, Fragment, LoopBody, OpenLoop, Value};

use owner::ScopeId;

use std::sync::Arc;

use crate::arena::{ArenaTag, Declaration, Function, OpId, ValueId};
use crate::func::DeviceFunction;
use crate::op::{OpKind, Trip, WriteMode};
use crate::{Evict, Param, PtrTy, Scalar, SharedArr, Ty, ValTy};

/// One list of operations under construction, and the scope its values belong
/// to.
struct Frame {
    scope: ScopeId,
    ops: Vec<OpId>,
}

/// Writes one arena and hands back a reference to each value it defines.
///
/// **Non-normalizing, and deliberately so.** It does not fold constants,
/// reassociate, reorder commutative operands, contract or expand FMA, or share
/// a repeated subexpression. Float association is a contract in this workspace,
/// not a preference — the rounding sequence a caller wrote is the one that must
/// reach the device — so the operations this records are the ones it was handed,
/// and any rewrite is a caller's explicit decision made somewhere it can be
/// read.
///
/// The backend numbers registers per class in definition order and labels from a
/// single ascending counter, so the emitted text is a function of the order
/// operations are placed here. Placing in a different order emits different text
/// for the same program; that is a property to preserve, not one to optimize.
pub struct Builder {
    function: Function,
    /// The lists being built, innermost last.
    frames: Vec<Frame>,
    /// How many scopes this builder has opened, counting up from zero.
    scopes: u32,
    /// What the function hands back. A kernel hands back nothing; a function
    /// meant to be called states its results before it finishes.
    returns: Vec<ValueId>,
    /// The constants this builder has already built, by the scope they were
    /// built in and the bits deciding them.
    ///
    /// The scope is part of the key because a value is readable only where it
    /// is bound: one built inside a region and read outside it would be
    /// defined in a scope that closes.
    constants: std::collections::HashMap<(ScopeId, Scalar, u64), OpId>,
    /// The operand walks placement asks for, kept between placements.
    ///
    /// A walk holds one bit per operation, so allocating one costs the size of
    /// the arena; placing operations one at a time asks for one per placement,
    /// and a compile places a million times.
    walks: Walks,
}

/// Scratch a builder reuses rather than allocating per placement.
#[derive(Default)]
struct Walks {
    regions: weave::Reached,
    /// What the operations being placed read, which is what may travel.
    ancestors: weave::Reached,
    /// What one operation being placed reads, asked once per operation.
    reaches: weave::Reached,
}

impl Default for Builder {
    fn default() -> Builder {
        Builder::new()
    }
}

impl Builder {
    pub fn new() -> Builder {
        Builder {
            function: Function::new(),
            frames: vec![Frame {
                scope: ScopeId::root(),
                ops: Vec::new(),
            }],
            scopes: 1,
            returns: Vec::new(),
            constants: std::collections::HashMap::new(),
            walks: Walks::default(),
        }
    }

    /// Continue the body a builder finished, keeping who owns its values and
    /// where its naming had got to.
    pub fn resuming(body: Body) -> Builder {
        let function = body.function;
        let entry = function.entry();
        let ops = function.block(entry).ops.clone();
        let returns = function.returned().to_vec();
        Builder {
            function,
            frames: vec![Frame {
                scope: ScopeId::root(),
                ops,
            }],
            scopes: 1,
            returns,
            constants: std::collections::HashMap::new(),
            walks: Walks::default(),
        }
    }

    pub fn function(&self) -> &Function {
        &self.function
    }

    pub(crate) fn tag(&self) -> ArenaTag {
        self.function.tag()
    }

    pub(super) fn scope(&self) -> ScopeId {
        self.frame().scope
    }

    /// The list being built. A builder always has one open, so this is the
    /// frame every method below means when it says "this frame".
    fn frame(&self) -> &Frame {
        self.frames.last().expect("a frame is always open")
    }

    fn frame_mut(&mut self) -> &mut Frame {
        self.frames.last_mut().expect("a frame is always open")
    }

    fn open(&self) -> Vec<ScopeId> {
        self.frames.iter().map(|frame| frame.scope).collect()
    }

    pub(super) fn read(&self, value: Value) -> ValueId {
        value.read(self.tag(), &self.open())
    }

    pub(super) fn held(&self, id: ValueId) -> Value {
        Value::new(self.tag(), self.scope(), id)
    }

    /// The value at `index` of the arena being written, for a caller holding an
    /// index rather than a value.
    ///
    /// An editor is such a caller: its handles outlive any one build, so it
    /// cannot be handed a [`Value`] and cannot carry the scope one records. What
    /// it gets is a value of the scope being written now, so the builder's own
    /// reachability assertion says nothing about it — where a value may be read
    /// is decided by [`crate::verify()`], which is where an editor's programs are
    /// decided anyway.
    pub fn value_at(&self, index: usize) -> Value {
        self.held(self.function.value_at(index))
    }

    fn read_all(&self, values: impl IntoIterator<Item = Value>) -> Vec<ValueId> {
        values.into_iter().map(|value| self.read(value)).collect()
    }

    /// Records an operation and places it in the list being built.
    fn record(
        &mut self,
        kind: OpKind,
        materialized: bool,
        operands: Vec<ValueId>,
        result_tys: &[ValTy],
        into: Vec<Option<ValueId>>,
        regions: Vec<crate::arena::RegionId>,
    ) -> OpId {
        let op = self
            .function
            .add_op(kind, materialized, operands, result_tys, into, regions);
        self.frame_mut().ops.push(op);
        op
    }

    /// Records an operation reading nothing, sharing one already built.
    ///
    /// A constant is decided by its kind alone — it reads nothing, so two of
    /// the same kind compute the same value, and one arena needs one of each.
    /// Building a second is a second thing to place, to walk and to render.
    pub(super) fn constant(&mut self, kind: OpKind, ty: ValTy) -> Value {
        // The bits, not the number: two float constants are the same operation
        // exactly when they render the same, which is what their bits say and
        // what equality on `f64` does not.
        let at = match kind {
            OpKind::IntConst { scalar, value } => (self.scope(), scalar, value as u64),
            OpKind::FloatConst { scalar, value } => (self.scope(), scalar, value.to_bits()),
            _ => panic!("{kind:?} is not a constant"),
        };
        // Placement moves operations between frames, so one remembered here
        // may since have travelled: it is shared only while this frame still
        // holds it, and rebuilt otherwise.
        if let Some(&op) = self.constants.get(&at)
            && self.frame().ops.contains(&op)
        {
            let id = self.function.op(op).results[0];
            return self.held(id);
        }
        let op = self.record(kind, false, Vec::new(), &[ty], Vec::new(), Vec::new());
        self.constants.insert(at, op);
        let id = self.function.op(op).results[0];
        self.held(id)
    }

    /// Records an operation that computes storage once at its block position.
    pub(super) fn place(
        &mut self,
        kind: OpKind,
        operands: Vec<ValueId>,
        result_tys: &[ValTy],
        into: Vec<Option<ValueId>>,
        regions: Vec<crate::arena::RegionId>,
    ) -> OpId {
        self.record(kind, true, operands, result_tys, into, regions)
    }

    /// Records an operation in the block being built. Its result is evaluated
    /// at each use rather than stored once.
    pub(super) fn operand(&mut self, kind: OpKind, operands: Vec<ValueId>, ty: ValTy) -> Value {
        let op = self.record(kind, false, operands, &[ty], Vec::new(), Vec::new());
        let id = self.function.op(op).results[0];
        self.held(id)
    }

    // ---- what a kernel declares ------------------------------------------

    /// Declares a parameter of the kernel this body belongs to.
    ///
    /// A kernel declares what it takes, not a region of it: a declaration made
    /// under a guard would be the kernel's all the same, which is not what
    /// writing it there says. Each declaration is its own, and the list keeps
    /// the order they were made in.
    pub fn parameter(&mut self, parameter: Param) -> Value {
        assert_eq!(
            self.scope(),
            ScopeId::root(),
            "a kernel parameter was declared inside a region"
        );
        self.refuse_redeclaration(&parameter.name);
        let ty = match parameter.ty {
            crate::ParamTy::Ptr(elem) => ValTy::Ptr(PtrTy {
                elem,
                space: crate::Space::Global,
            }),
            crate::ParamTy::Scalar(scalar) => ValTy::Data(Ty::scalar(scalar)),
        };
        let name = parameter.name.clone();
        let id = self.function.declare(Declaration::Param(parameter), ty);
        self.function.set_name(id, name);
        Value::new(self.tag(), ScopeId::root(), id)
    }

    /// Declares a shared array of the kernel this body belongs to, under the
    /// same rule as a parameter.
    pub fn shared_array(&mut self, array: SharedArr) -> Value {
        assert_eq!(
            self.scope(),
            ScopeId::root(),
            "a shared array was declared inside a region"
        );
        self.refuse_redeclaration(&array.name);
        let ty = ValTy::Ptr(PtrTy {
            elem: array.elem,
            space: crate::Space::Shared,
        });
        let name = array.name.clone();
        let id = self.function.declare(Declaration::Shared(array), ty);
        self.function.set_name(id, name);
        Value::new(self.tag(), ScopeId::root(), id)
    }

    pub fn parameters(&self) -> Vec<Param> {
        self.function.parameters()
    }

    pub fn shared_arrays(&self) -> Vec<SharedArr> {
        self.function.shared_arrays()
    }

    /// Refuses a second declaration under a name this kernel already declares.
    ///
    /// One name is one declaration. Two that agree are still two, and the
    /// second says something the first already said — a mistake about what the
    /// kernel takes, not a restatement of it.
    fn refuse_redeclaration(&self, name: &str) {
        for declared in self.function.declarations() {
            let (held, kind) = match declared {
                Declaration::Param(param) => (param.name.as_str(), "parameter"),
                Declaration::Shared(array) => (array.name.as_str(), "shared array"),
            };
            assert!(held != name, "{name} is already a {kind} of this kernel");
        }
    }

    // ---- statements ------------------------------------------------------

    pub fn barrier(&mut self) {
        self.place(OpKind::Bar, Vec::new(), &[], Vec::new(), Vec::new());
    }

    /// Leaves the kernel.
    pub fn ret(&mut self) {
        self.place(OpKind::Ret, Vec::new(), &[], Vec::new(), Vec::new());
    }

    /// Leaves the innermost loop.
    pub fn break_(&mut self) {
        self.place(OpKind::Break, Vec::new(), &[], Vec::new(), Vec::new());
    }

    /// Replaces what the address holds.
    pub fn store(&mut self, ptr: Value, value: Value) {
        self.store_evict(ptr, value, Evict::Normal);
    }

    pub fn store_evict(&mut self, ptr: Value, value: Value, evict: Evict) {
        self.write(WriteMode::Set, ptr, value, evict);
    }

    /// Adds to what the address holds, indivisibly, so several blocks may fold
    /// into one cell.
    pub fn red_add(&mut self, ptr: Value, value: Value) {
        self.write(WriteMode::Add, ptr, value, Evict::Normal);
    }

    /// Adds `value` to the scalar at `ptr` indivisibly, and gives back what was
    /// there before. The addition is a step of this body whether or not the
    /// result is read; leaving it unread is what a caller does when it wants
    /// the sum and not the order.
    pub fn fetch_add(&mut self, ptr: Value, value: Value) -> Value {
        let ty = self.value_ty(&value);
        let operands = vec![self.read(ptr), self.read(value)];
        let op = self.place(OpKind::FetchAdd, operands, &[ty], Vec::new(), Vec::new());
        let id = self.function.op(op).results[0];
        self.held(id)
    }

    pub fn write(&mut self, mode: WriteMode, ptr: Value, value: Value, evict: Evict) {
        let operands = vec![self.read(ptr), self.read(value)];
        self.place(
            OpKind::Write { mode, evict },
            operands,
            &[],
            Vec::new(),
            Vec::new(),
        );
    }

    /// Runs `callee` on `arguments`, producing what it hands back.
    ///
    /// The callee is the function itself, not a name resolved against a table
    /// beside the program, so a call keeps its meaning wherever the operation
    /// is carried. [`crate::inline_calls`] is what runs the body here.
    pub fn call(&mut self, callee: Arc<DeviceFunction>, arguments: Vec<Value>) -> Vec<Value> {
        let operands = self.read_all(arguments);
        let tys = callee.result_tys();
        let op = self.place(OpKind::Call(callee), operands, &tys, Vec::new(), Vec::new());
        self.results_of(op)
    }

    // ---- naming ----------------------------------------------------------

    /// Names a value, for a caller whose name carries meaning the arena does
    /// not — which stage a value belongs to, which output it addresses.
    pub fn name_value(&mut self, value: &Value, name: impl Into<String>) {
        let name = name.into();
        let id = value.id_in(&self.function);
        self.function.set_name(id, name);
    }

    // ---- lists built aside -----------------------------------------------

    fn enter(&mut self, scope: ScopeId) {
        self.frames.push(Frame {
            scope,
            ops: Vec::new(),
        });
    }

    fn inner_scope(&mut self) -> ScopeId {
        let scope = ScopeId(self.scopes);
        self.scopes = self
            .scopes
            .checked_add(1)
            .expect("builder scopes exhausted");
        scope
    }

    fn leave(&mut self) -> Vec<OpId> {
        self.frames.pop().expect("a frame was opened").ops
    }

    /// Operations built now and placed later, by this builder, together with
    /// whatever building them produced.
    pub fn build_fragment<T>(&mut self, build: impl FnOnce(&mut Builder) -> T) -> (Fragment, T) {
        let scope = self.scope();
        self.enter(scope);
        let built = build(self);
        let ops = self.leave();
        (
            Fragment {
                tag: self.tag(),
                scope,
                ops,
            },
            built,
        )
    }

    /// As [`Builder::build_fragment`], for a region that runs under a scope of
    /// its own rather than in sequence here.
    fn build_region<T>(&mut self, build: impl FnOnce(&mut Builder) -> T) -> (Vec<OpId>, T) {
        let scope = self.inner_scope();
        self.enter(scope);
        let built = build(self);
        (self.leave(), built)
    }

    pub fn append_fragment(&mut self, fragment: Fragment) {
        assert_eq!(
            fragment.tag,
            self.tag(),
            "a fragment of another builder was appended here"
        );
        assert_eq!(
            fragment.scope,
            self.scope(),
            "a fragment built in another scope was appended here"
        );
        self.place_all(fragment.into_ops());
    }

    /// Places a fragment here and says which of its operations run.
    ///
    /// A caller that means to read the fragment back as a list of statements
    /// wants both answers at once, and taking them apart is what loses an
    /// operand: it is placed by this, and named by neither the steps returned
    /// nor any later slice of them.
    pub fn place_fragment(&mut self, fragment: Fragment) -> Vec<OpId> {
        let steps = fragment.steps(&self.function);
        self.append_fragment(fragment);
        steps
    }

    /// Abandons a fragment: what it built is not part of the program.
    ///
    /// A caller that builds speculatively and then finds it does not need the
    /// result says so here. Every operation belongs to a block, and a fragment
    /// is what a builder holds *before* deciding where its operations go, so
    /// one simply let go of would leave them held by nobody.
    pub fn discard_fragment(&mut self, fragment: Fragment) {
        assert_eq!(
            fragment.tag,
            self.tag(),
            "a fragment of another builder was discarded here"
        );
        let ops = fragment.into_ops();
        self.function.forget_unheld(&ops);
    }

    pub fn append_fragments(&mut self, fragments: impl IntoIterator<Item = Fragment>) {
        for fragment in fragments {
            self.append_fragment(fragment);
        }
    }

    /// The operation that defines a value this builder placed.
    pub fn defining_op(&self, value: &Value) -> Option<OpId> {
        self.function.defining_op(value.id_in(&self.function))
    }

    /// The operations the list being built holds so far, steps and operands
    /// alike. Every operation belongs to a block, so an operand appears here
    /// too, and is evaluated wherever it is read rather than at this position.
    pub fn owned(&self) -> &[OpId] {
        &self.frame().ops
    }

    /// Those of them that execute where they stand, which is what a caller
    /// reading the list back as a body means by it.
    pub fn steps(&self) -> Vec<OpId> {
        self.owned()
            .iter()
            .copied()
            .filter(|&op| self.function.is_step(op))
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.owned().is_empty()
    }

    // ---- regions ---------------------------------------------------------

    fn region_of(&mut self, ops: Vec<OpId>, yields: Vec<ValueId>) -> crate::arena::RegionId {
        let block = self.function.add_block();
        self.function.set_block_ops(block, ops);
        self.function.set_terminator(block, yields);
        self.function.add_region(vec![block])
    }

    /// A region that runs whatever it holds, opening a scope of its own.
    pub fn scoped(&mut self, body: impl FnOnce(&mut Builder)) {
        let (ops, ()) = self.build_region(body);
        let region = self.region_of(ops, Vec::new());
        self.place(OpKind::Scope, Vec::new(), &[], Vec::new(), vec![region]);
    }

    /// A guarded region that produces nothing.
    pub fn if_(&mut self, pred: Value, body: impl FnOnce(&mut Builder)) {
        self.if_else(pred, body, |_| {});
    }

    /// Two guarded regions, neither producing a value.
    pub fn if_else(
        &mut self,
        pred: Value,
        then_: impl FnOnce(&mut Builder),
        else_: impl FnOnce(&mut Builder),
    ) {
        let (then_ops, ()) = self.build_region(then_);
        let (else_ops, ()) = self.build_region(else_);
        let then_region = self.region_of(then_ops, Vec::new());
        let else_region = self.region_of(else_ops, Vec::new());
        let pred = self.read(pred);
        self.place(
            OpKind::If,
            vec![pred],
            &[],
            Vec::new(),
            vec![then_region, else_region],
        );
    }

    /// Two guarded regions producing one value per entry of `results`, each
    /// stated as what it holds and the place it takes over.
    ///
    /// A result is a fresh value: the arms read what the enclosing scope bound,
    /// and the result becomes visible only after the branch. Where it names a
    /// place, it is produced into that place's storage, so the join copies
    /// nothing.
    pub fn if_producing(
        &mut self,
        pred: Value,
        results: &[(Ty, Option<Value>)],
        then_: impl FnOnce(&mut Builder) -> Vec<Value>,
        else_: impl FnOnce(&mut Builder) -> Vec<Value>,
    ) -> Vec<Value> {
        let into = results
            .iter()
            .map(|(_, place)| place.clone().map(|place| self.read(place)))
            .collect::<Vec<_>>();
        let tys = results
            .iter()
            .map(|(ty, _)| ValTy::Data(*ty))
            .collect::<Vec<_>>();
        let then_ = self.build_region_yielding(results.len(), then_);
        let else_ = self.build_region_yielding(results.len(), else_);
        assert_eq!(
            then_.yields.len(),
            results.len(),
            "a then arm yielded the wrong arity"
        );
        let then_region = self.region_of(then_.ops, then_.yields);
        let else_region = self.region_of(else_.ops, else_.yields);
        let pred = self.read(pred);
        let op = self.place(
            OpKind::If,
            vec![pred],
            &tys,
            into,
            vec![then_region, else_region],
        );
        self.results_of(op)
    }

    fn build_region_yielding(
        &mut self,
        arity: usize,
        build: impl FnOnce(&mut Builder) -> Vec<Value>,
    ) -> Arm {
        self.open_arm();
        let yields = build(self);
        assert_eq!(yields.len(), arity, "a region yielded the wrong arity");
        self.close_arm(yields)
    }

    /// Opens a region of a construct this builder will place, for a caller that
    /// builds its body by calling back into itself.
    ///
    /// The closure forms above are the same thing with the two ends tied
    /// together. They cannot serve a producer whose body is built by a method
    /// of its own — the closure would need the builder while the producer still
    /// holds it — so the two ends are also available apart.
    pub fn open_arm(&mut self) {
        let scope = self.inner_scope();
        self.enter(scope);
    }

    pub fn close_arm(&mut self, yields: Vec<Value>) -> Arm {
        let yields = self.read_all(yields);
        Arm {
            ops: self.leave(),
            yields,
        }
    }

    /// Two arms already built, producing one value per entry of `results`.
    pub fn if_arms(
        &mut self,
        pred: Value,
        results: &[(Ty, Option<Value>)],
        then_: Arm,
        else_: Arm,
    ) -> Vec<Value> {
        let into = results
            .iter()
            .map(|(_, place)| place.clone().map(|place| self.read(place)))
            .collect::<Vec<_>>();
        let tys = results
            .iter()
            .map(|(ty, _)| ValTy::Data(*ty))
            .collect::<Vec<_>>();
        assert_eq!(
            then_.yields.len(),
            results.len(),
            "a then arm yielded the wrong arity"
        );
        assert_eq!(
            else_.yields.len(),
            results.len(),
            "an else arm yielded the wrong arity"
        );
        let then_region = self.region_of(then_.ops, then_.yields);
        let else_region = self.region_of(else_.ops, else_.yields);
        let pred = self.read(pred);
        let op = self.place(
            OpKind::If,
            vec![pred],
            &tys,
            into,
            vec![then_region, else_region],
        );
        self.results_of(op)
    }

    /// Opens the body of a loop this builder will place: the index and each
    /// carried value are bound on the way in.
    pub fn open_loop_body(&mut self, index: Scalar, carried: Vec<Carry>) -> OpenLoop {
        let placement = self.scope();
        let block = self.function.add_block();
        let index_value = self
            .function
            .add_block_param(block, ValTy::Data(Ty::scalar(index)));
        // A carried value continues the place it starts from, so where that
        // place carries a name it carries it too — inside the body and in what
        // the loop produces.
        let carried_values = carried
            .iter()
            .map(|carry| {
                let param = self.function.add_block_param(block, ValTy::Data(carry.ty));
                if let Some(place) = carry.into.as_ref() {
                    let place = place.clone().id_in(&self.function);
                    if let Some(name) = self.function.name(place).map(str::to_owned) {
                        self.function.set_name(param, name);
                    }
                }
                param
            })
            .collect::<Vec<_>>();
        let scope = self.inner_scope();
        self.enter(scope);
        OpenLoop {
            placement,
            block,
            index_scalar: index,
            index: Value::new(self.tag(), scope, index_value),
            given: carried_values
                .into_iter()
                .map(|id| Value::new(self.tag(), scope, id))
                .collect(),
            carried,
        }
    }

    pub fn close_loop_body(&mut self, open: OpenLoop, yields: Vec<Value>) -> LoopBody {
        assert_eq!(
            yields.len(),
            open.carried.len(),
            "a loop body yielded the wrong arity"
        );
        let yields = self.read_all(yields);
        let ops = self.leave();
        self.function.set_block_ops(open.block, ops);
        self.function.set_terminator(open.block, yields);
        LoopBody {
            tag: self.tag(),
            scope: open.placement,
            block: open.block,
            index: open.index_scalar,
            carried: open.carried,
        }
    }

    fn results_of(&self, op: OpId) -> Vec<Value> {
        self.function
            .op(op)
            .results
            .iter()
            .map(|&id| self.held(id))
            .collect()
    }

    /// One value, chosen by `pred`, that each arm builds.
    pub fn if_value(
        &mut self,
        pred: Value,
        ty: Ty,
        then_: impl FnOnce(&mut Builder) -> Value,
        else_: impl FnOnce(&mut Builder) -> Value,
    ) -> Value {
        let mut produced =
            self.if_producing(pred, &[(ty, None)], |b| vec![then_(b)], |b| vec![else_(b)]);
        produced.pop().expect("one result")
    }

    /// `N` values, chosen by `pred`, that each arm builds.
    pub fn if_values<const N: usize>(
        &mut self,
        pred: Value,
        ty: [Ty; N],
        then_: impl FnOnce(&mut Builder) -> [Value; N],
        else_: impl FnOnce(&mut Builder) -> [Value; N],
    ) -> [Value; N] {
        let results = ty.map(|ty| (ty, None));
        let produced = self.if_producing(pred, &results, |b| then_(b).into(), |b| else_(b).into());
        into_array(produced)
    }

    // ---- counted loops ----------------------------------------------------

    /// A loop body built for a loop this builder will place: the index and each
    /// carried value are bound on the way in, so the body sees them and nothing
    /// outside can.
    pub fn build_loop_body<T>(
        &mut self,
        index: Scalar,
        carried: Vec<Carry>,
        build: impl FnOnce(&mut Builder, Value, &[Value]) -> (Vec<Value>, T),
    ) -> (LoopBody, T) {
        let open = self.open_loop_body(index, carried);
        let (this, given) = (open.index.clone(), open.given.clone());
        let (yields, built) = build(self, this, &given);
        (self.close_loop_body(open, yields), built)
    }

    /// A counted loop over the body it was entered with, placed here.
    ///
    /// The index and the carried values come from the body and nowhere else, so
    /// a body built to be entered one way cannot be placed another. The bounds
    /// are values this builder owns and are read here like any other operand.
    /// `trip` is what the caller knows about how often the block's threads go
    /// round: a fact about the bounds, which are values, so the arena cannot
    /// read it off the program. Stating [`Trip::BlockUniform`] is what admits a
    /// barrier in the body, and a barrier reached a different number of times
    /// per thread is one no thread gets past.
    pub fn place_counted_loop(
        &mut self,
        trip: Trip,
        start: Value,
        stop: Value,
        step: Value,
        body: LoopBody,
    ) -> Vec<Value> {
        self.counted_loop_op(start, stop, step, body, true, trip)
    }

    /// A counted loop standing for the values it reduces to rather than being a
    /// step of the body: it is evaluated where the expression it was written in
    /// is, so it nests inside one like any other operand.
    pub fn fold_loop(
        &mut self,
        trip: Trip,
        start: Value,
        stop: Value,
        step: Value,
        body: LoopBody,
    ) -> Vec<Value> {
        self.counted_loop_op(start, stop, step, body, false, trip)
    }

    fn counted_loop_op(
        &mut self,
        start: Value,
        stop: Value,
        step: Value,
        body: LoopBody,
        placed: bool,
        trip: Trip,
    ) -> Vec<Value> {
        assert_eq!(
            body.tag,
            self.tag(),
            "a loop body of another builder was placed here"
        );
        assert_eq!(
            body.scope,
            self.scope(),
            "a loop body built in another scope was placed here"
        );
        let mut operands = vec![self.read(start), self.read(stop), self.read(step)];
        let mut into = Vec::with_capacity(body.carried.len());
        let mut tys = Vec::with_capacity(body.carried.len());
        for carry in body.carried {
            operands.push(self.read(carry.init));
            into.push(carry.into.map(|place| self.read(place)));
            tys.push(ValTy::Data(carry.ty));
        }
        let region = self.function.add_region(vec![body.block]);
        let kind = OpKind::Loop {
            index: body.index,
            trip,
        };
        let op = self.record(kind, placed, operands, &tys, into, vec![region]);
        self.results_of(op)
    }

    /// A counted loop carrying one value per entry of `carried`, whose body
    /// receives the index and what this iteration was given, and yields what the
    /// next receives.
    pub fn counted_loop_producing(
        &mut self,
        counted: Counted,
        carried: Vec<Carry>,
        body: impl FnOnce(&mut Builder, Value, &[Value]) -> Vec<Value>,
    ) -> Vec<Value> {
        let Counted {
            index,
            start,
            stop,
            step,
            trip,
        } = counted;
        let (built, ()) = self.build_loop_body(index, carried, |b, index, given| {
            (body(b, index, given), ())
        });
        self.place_counted_loop(trip, start, stop, step, built)
    }

    /// [`Builder::counted_loop_producing`] over a fixed number of fresh carried
    /// values.
    pub fn counted_loop<const N: usize>(
        &mut self,
        counted: Counted,
        carried: [(Ty, Value); N],
        body: impl FnOnce(&mut Builder, Value, [Value; N]) -> [Value; N],
    ) -> [Value; N] {
        let carried = carried
            .into_iter()
            .map(|(ty, init)| Carry {
                ty,
                init,
                into: None,
            })
            .collect::<Vec<_>>();
        let produced = self.counted_loop_producing(counted, carried, |b, i, g| {
            let given: [Value; N] = into_array(g.to_vec());
            body(b, i, given).into()
        });
        into_array(produced)
    }

    // ---- finishing --------------------------------------------------------

    /// What this function hands back, for one meant to be called. A kernel
    /// hands back nothing, which is what it starts out doing.
    pub fn returns(&mut self, values: Vec<Value>) {
        assert_eq!(
            self.scope(),
            ScopeId::root(),
            "a function's results were stated inside a region"
        );
        self.returns = self.read_all(values);
    }

    pub fn finish_owned(mut self) -> Body {
        assert_eq!(self.frames.len(), 1, "a region was left open");
        let entry = self.function.entry();
        let ops = self.leave();
        self.function.set_block_ops(entry, ops);
        let returns = std::mem::take(&mut self.returns);
        self.function.set_terminator(entry, returns);
        Body {
            function: self.function,
        }
    }

    pub fn into_function(self) -> Function {
        self.finish_owned().into_function()
    }
}

fn into_array<T: std::fmt::Debug, const N: usize>(values: Vec<T>) -> [T; N] {
    values.try_into().expect("the arity was checked")
}
