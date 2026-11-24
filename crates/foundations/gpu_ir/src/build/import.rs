// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carrying a program from one arena into another.
//!
//! Values of two arenas can never meet, so the two things a caller does with a
//! finished body are different operations. **Analysis is a subset**: a part is a
//! list of operation ids of the one arena, with nothing copied and nothing
//! renumbered. **Materialization imports**: a generated kernel is not a view of
//! the body it came from — the same value is computed in one kernel and loaded
//! in another, candidates cut differently, and each kernel adds parameters and
//! writes that have no counterpart in the source.
//!
//! Importing copies in the **source's own order**. The emitted text is a
//! function of the order values are defined, so choosing a fresh topological
//! order here would renumber every register for the same program.

use std::collections::BTreeSet;

use super::owner::{ScopeId, Value};
use super::{Body, Builder};
use crate::arena::{ArenaTag, Def, Function, OpId, ValueId};
use crate::op::OpKind;

/// What a copy does with a call it meets: carries it as it stands, or runs the
/// callee's body here instead.
///
/// Inlining is this and nothing more, which is the point: a call is expanded by
/// the same copy that carries a program between arenas, so there is no second
/// renumbering path to get wrong.
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Calls {
    Carried,
    Inlined,
}

/// What one import produced: the operations it copied, in the source's order,
/// and the destination value standing for each source value it reached.
pub struct Import {
    source: ArenaTag,
    tag: ArenaTag,
    scope: ScopeId,
    ops: Vec<OpId>,
    map: Vec<Option<ValueId>>,
}

impl Import {
    /// The stored operations this import copied, in the source's order. The
    /// caller decides which of them the body runs.
    pub fn ops(&self) -> &[OpId] {
        &self.ops
    }

    pub fn into_ops(self) -> Vec<OpId> {
        self.ops
    }

    /// Those of them that execute where they stand, for a caller reading the
    /// import as a list of steps rather than placing it.
    pub fn steps(&self, f: &Function) -> Vec<OpId> {
        self.ops
            .iter()
            .copied()
            .filter(|&op| f.is_step(op))
            .collect()
    }

    /// The destination value standing for a source value this import reached.
    pub fn value(&self, source: ValueId) -> Value {
        let source = source.at(self.source);
        let id = self.map[source].expect("that value was not reached by this import");
        Value::new(self.tag, self.scope, id)
    }

    pub fn reached(&self, source: ValueId) -> bool {
        self.map[source.at(self.source)].is_some()
    }
}

impl Builder {
    /// The destination value standing for a source value, copying the operand
    /// cone above it where the import has not already reached it.
    ///
    /// Only a place arrives by being copied as a step. An expression the source
    /// never bound — an output element, a guard predicate, a reduction term —
    /// is an operand, so it arrives when a caller asks for it, once per import
    /// and shared with every later ask, as the source shares it.
    ///
    /// A value the import cut away is not there to copy: what it used to read
    /// did not arrive, and this says so rather than rebuilding it.
    pub fn imported(
        &mut self,
        import: &mut Import,
        source: &Function,
        value: ValueId,
    ) -> Result<Value, String> {
        source.value(value);
        let id = self.copy_value(source, value, &mut import.map)?;
        Ok(Value::new(import.tag, import.scope, id))
    }
}

impl Builder {
    /// Copies the whole body of `from` into this arena, with `substitutions`
    /// supplying a value of this arena for each value `from` reads from outside
    /// itself — its own parameters.
    ///
    /// The operations come back in the source's order and are **not** placed:
    /// what the body runs is the caller's decision, made once the analyses that
    /// read the copy have finished with it.
    pub fn import(
        &mut self,
        from: &Body,
        substitutions: &[(ValueId, Value)],
    ) -> Result<Import, String> {
        self.import_resolving(from, substitutions, Calls::Carried)
    }

    /// As [`Builder::import`], deciding what happens to a call on the way.
    pub fn import_resolving(
        &mut self,
        from: &Body,
        substitutions: &[(ValueId, Value)],
        calls: Calls,
    ) -> Result<Import, String> {
        let source = from.function();
        let entry = source.entry();
        for &param in &source.block(entry).params {
            if !substitutions.iter().any(|(replaced, _)| *replaced == param) {
                return Err("importing left a parameter of the source unsubstituted".into());
            }
        }
        let ops = source.block(entry).ops.clone();
        self.import_steps(source, &ops, substitutions, calls)
    }

    /// Copies a list of steps of `source` into this arena, with `substitutions`
    /// supplying a value of this arena for everything they read from outside
    /// themselves.
    pub fn import_ops(
        &mut self,
        source: &Function,
        steps: &[OpId],
        substitutions: &[(ValueId, Value)],
    ) -> Result<Import, String> {
        self.import_steps(source, steps, substitutions, Calls::Carried)
    }

    fn import_steps(
        &mut self,
        source: &Function,
        steps: &[OpId],
        substitutions: &[(ValueId, Value)],
        calls: Calls,
    ) -> Result<Import, String> {
        let mut map = vec![None; source.num_values()];
        for (replaced, replacement) in substitutions {
            source.value(*replaced);
            let replacement = self.read(replacement.clone());
            map[replaced.index()] = Some(replacement);
        }
        let before = self.frame_len();
        let mut ops = Vec::with_capacity(steps.len());
        for &op in steps {
            // A substituted definition is a cut: it is read, not recomputed, so
            // neither it nor what it used to read arrives.
            if source
                .op(op)
                .results
                .iter()
                .any(|result| map[result.index()].is_some())
            {
                continue;
            }
            ops.extend(self.copy_op(source, op, &mut map, calls)?);
        }
        // Every operation belongs to a block, so what an import copies is held
        // here from the moment it arrives, in the order it was created: an
        // operand is copied through the operation reading it, so it is created
        // before that reader. The caller decides which of them the body *runs*
        // — that is what the step list it takes back is for.
        self.reorder_fresh(before, &ops);
        Ok(Import {
            source: source.tag(),
            tag: self.tag(),
            scope: self.scope(),
            ops,
            map,
        })
    }

    /// Copies the operations of `roots`' closure in `from`, in the source's own
    /// order, with `substitutions` supplying the cut points, and places them
    /// here.
    ///
    /// A substituted value is read rather than recomputed: what it used to read
    /// is not imported, which is what makes a substitution a cut in the graph
    /// rather than an extra definition. That is how a value written out in one
    /// kernel and read back in another arrives.
    pub fn import_closure(
        &mut self,
        from: &Body,
        roots: &[Value],
        substitutions: &[(ValueId, Value)],
    ) -> Result<Vec<Value>, String> {
        let source = from.function();
        let entry = source.entry();
        let roots = roots
            .iter()
            .map(|root| root.id_in(source))
            .collect::<Vec<_>>();

        let cut = substitutions
            .iter()
            .map(|(replaced, _)| replaced.at(source.tag()))
            .collect::<BTreeSet<_>>();
        // Backwards, to learn which definitions the roots reach; a substituted
        // one is a cut, so what it used to read is not reached through it.
        let mut needed = BTreeSet::new();
        for &root in &roots {
            for read in source.reads(root) {
                needed.insert(read.index());
            }
        }
        let mut reached = BTreeSet::new();
        for index in (0..source.block(entry).ops.len()).rev() {
            let op = source.block(entry).ops[index];
            if !source.op(op).placed {
                continue;
            }
            let defined = match source.op(op).kind {
                OpKind::Bind => source.op(op).results[0],
                _ => return Err("importing expects flat bindings".into()),
            };
            if !needed.remove(&defined.index()) {
                continue;
            }
            reached.insert(defined.index());
            if !cut.contains(&defined.index()) {
                for read in source.reads(source.op(op).operands[0]) {
                    needed.insert(read.index());
                }
            }
        }

        let mut map = vec![None; source.num_values()];
        for (replaced, replacement) in substitutions {
            source.value(*replaced);
            let replacement = self.read(replacement.clone());
            map[replaced.index()] = Some(replacement);
        }
        // Forwards, so what arrives is in the order the source holds it.
        for index in 0..source.block(entry).ops.len() {
            let op = source.block(entry).ops[index];
            if !source.op(op).placed {
                continue;
            }
            let defined = source.op(op).results[0];
            if !reached.contains(&defined.index()) || map[defined.index()].is_some() {
                continue;
            }
            let copied = self.copy_op(source, op, &mut map, Calls::Carried)?;
            self.place_all(copied);
        }
        roots
            .into_iter()
            .map(|root| {
                self.copy_value(source, root, &mut map)
                    .map(|id| self.held(id))
            })
            .collect()
    }

    /// Carries the name a source value had onto the value standing for it here.
    ///
    /// A name is provenance — which stage a value belongs to, which output it
    /// addresses — and that is a property of the value, not of the arena it
    /// happens to sit in, so it follows the copy. Two copies of one body may
    /// then carry one name twice; nothing shadows, because a value is its id.
    fn carry_name(&mut self, source: &Function, from: ValueId, to: ValueId) {
        if let Some(name) = source.name(from) {
            let name = name.to_owned();
            self.function.set_name(to, name);
        }
    }

    /// Copies one placed operation, its regions included, and gives back what
    /// arrived — one operation, or the callee's whole body where a call was
    /// inlined.
    fn copy_op(
        &mut self,
        source: &Function,
        op: OpId,
        map: &mut [Option<ValueId>],
        calls: Calls,
    ) -> Result<Vec<OpId>, String> {
        if calls == Calls::Inlined
            && let OpKind::Call(callee) = source.op(op).kind.clone()
            && let Some(body) = callee.body()
        {
            return self.inline_call(source, op, &callee.name, body, map, calls);
        }
        let held = source.op(op);
        let mut operands = Vec::with_capacity(held.operands.len());
        for index in 0..held.operands.len() {
            let operand = source.op(op).operands[index];
            operands.push(self.copy_value(source, operand, map)?);
        }
        let mut regions = Vec::with_capacity(source.op(op).regions.len());
        for index in 0..source.op(op).regions.len() {
            let region = source.op(op).regions[index];
            let block = source.region_block(region);
            let copied = self.function.add_block();
            for slot in 0..source.block(block).params.len() {
                let param = source.block(block).params[slot];
                let ty = source.ty(param);
                let value = self.function.add_block_param(copied, ty);
                self.carry_name(source, param, value);
                map[param.index()] = Some(value);
            }
            let mut placed = Vec::with_capacity(source.block(block).ops.len());
            for slot in 0..source.block(block).ops.len() {
                let inner = source.block(block).ops[slot];
                placed.extend(self.copy_op(source, inner, map, calls)?);
            }
            self.function.set_block_ops(copied, placed);
            let mut yields = Vec::with_capacity(source.block(block).terminator.yields.len());
            for slot in 0..source.block(block).terminator.yields.len() {
                let yielded = source.block(block).terminator.yields[slot];
                yields.push(self.copy_value(source, yielded, map)?);
            }
            self.function.set_terminator(copied, yields);
            regions.push(self.function.add_region(vec![copied]));
        }
        let mut into = Vec::with_capacity(source.op(op).into.len());
        for index in 0..source.op(op).into.len() {
            let destination = source.op(op).into[index];
            into.push(match destination {
                Some(destination) => Some(self.copy_value(source, destination, map)?),
                None => None,
            });
        }
        let result_tys = source
            .op(op)
            .results
            .iter()
            .map(|&result| source.ty(result))
            .collect::<Vec<_>>();
        let kind = source.op(op).kind.clone();
        let placed = source.op(op).placed;
        let copied = self
            .function
            .add_op(kind, placed, operands, &result_tys, into, regions);
        for index in 0..source.op(op).results.len() {
            let result = source.op(op).results[index];
            let here = self.function.op(copied).results[index];
            self.carry_name(source, result, here);
            map[result.index()] = Some(here);
        }
        Ok(vec![copied])
    }

    /// Runs a callee's body here in place of the call, and gives the call's
    /// results the values that body handed back.
    ///
    /// The callee is a second arena, so this is the ordinary two-arena copy
    /// with the arguments standing in for the callee's parameters: exactly the
    /// substitution a cut point is, made at the boundary a call already draws.
    /// A call inside the callee is expanded the same way, and there can be no
    /// cycle to fall into — a callee is a whole function, so it existed before
    /// anything could call it.
    fn inline_call(
        &mut self,
        source: &Function,
        op: OpId,
        callee: &str,
        body: &Function,
        map: &mut [Option<ValueId>],
        calls: Calls,
    ) -> Result<Vec<OpId>, String> {
        if !body.shared_arrays().is_empty() {
            return Err(format!(
                "callee {callee} declares shared memory, which belongs to the kernel it is \
                 inlined into"
            ));
        }
        let declarations = body.declarations().len();
        if declarations != source.op(op).operands.len() {
            return Err(format!(
                "callee {callee} takes {declarations} arguments and was given {}",
                source.op(op).operands.len()
            ));
        }
        let mut inner = vec![None; body.num_values()];
        for slot in 0..declarations {
            let argument = source.op(op).operands[slot];
            let here = self.copy_value(source, argument, map)?;
            inner[body.declared(slot).index()] = Some(here);
        }
        let mut placed = Vec::new();
        for slot in 0..body.block(body.entry()).ops.len() {
            let step = body.block(body.entry()).ops[slot];
            placed.extend(self.copy_op(body, step, &mut inner, calls)?);
        }
        let returned = body.returned().to_vec();
        if returned.len() != source.op(op).results.len() {
            return Err(format!(
                "callee {callee} hands back {} values and the call produces {}",
                returned.len(),
                source.op(op).results.len()
            ));
        }
        for (slot, &handed) in returned.iter().enumerate() {
            let here = self.copy_value(body, handed, &mut inner)?;
            let result = source.op(op).results[slot];
            self.carry_name(source, result, here);
            map[result.index()] = Some(here);
        }
        Ok(placed)
    }

    /// Copies one value: a place must already have arrived, an operand is copied
    /// here through the operation defining it and shared with every other read
    /// of it, as the source shares it.
    fn copy_value(
        &mut self,
        source: &Function,
        value: ValueId,
        map: &mut [Option<ValueId>],
    ) -> Result<ValueId, String> {
        source.value(value);
        if let Some(copied) = map[value.index()] {
            return Ok(copied);
        }
        let mut made = Vec::new();
        let copied = self.copy_value_recording(source, value, map, &mut made)?;
        // Nothing asked for these by name, so they belong where the operation
        // reading them is being built.
        self.place_all(made);
        Ok(copied)
    }

    /// [`Builder::copy_value`], collecting the operations it had to create.
    fn copy_value_recording(
        &mut self,
        source: &Function,
        value: ValueId,
        map: &mut [Option<ValueId>],
        made: &mut Vec<OpId>,
    ) -> Result<ValueId, String> {
        source.value(value);
        if let Some(copied) = map[value.index()] {
            return Ok(copied);
        }
        match source.value(value).def {
            Def::Param { .. } => {
                Err("importing reached a parameter it was given no value for".into())
            }
            Def::Result { op, .. } => {
                if source.op(op).placed {
                    return Err("importing reached a definition it did not copy".into());
                }
                made.extend(self.copy_op(source, op, map, Calls::Carried)?);
                Ok(map[value.index()].expect("the operation copied its result"))
            }
        }
    }
}
