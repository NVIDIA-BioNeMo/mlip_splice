// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Function, OpId, ValueId};

use super::{BTreeMap, BTreeSet, Definition, OpKind};

/// How many steps a body runs, region interiors included.
///
/// A region-owning step counts one for itself and one for each step inside it.
pub(in super::super) fn statement_count(f: &Function) -> usize {
    count_steps(f, &f.block_steps(f.entry()).collect::<Vec<_>>())
}

fn count_steps(f: &Function, ops: &[OpId]) -> usize {
    ops.iter()
        .map(|&op| {
            1 + f
                .op(op)
                .regions
                .iter()
                .flat_map(|&region| f.region(region).blocks.iter())
                .map(|&block| count_steps(f, &f.block_steps(block).collect::<Vec<_>>()))
                .sum::<usize>()
        })
        .sum()
}

/// A profile's flat bindings, resolved into the closure index space.
///
/// Every value is interned once: a binding takes the index of its step, and a
/// value the profile only reads — an input load — takes an index past the last
/// binding. Each definition records the entries it reads in that space, so
/// walking a dependency closure indexes arrays instead of following use-def
/// edges again.
pub(in super::super) struct Bindings {
    definitions: Vec<Definition>,
    index: BTreeMap<ValueId, usize>,
    /// The input loads in intern order, so the `k`-th holds entry
    /// `definitions.len() + k`.
    inputs: Vec<ValueId>,
}

impl Bindings {
    /// Resolves `ops`, which must be a flat list of bindings. Values are read
    /// before dependencies, so a binding is a binding wherever it is written.
    pub(crate) fn of(f: &Function, ops: &[OpId]) -> Result<Self, String> {
        let mut index = BTreeMap::new();
        let mut inits = Vec::with_capacity(ops.len());
        for (position, &op) in ops.iter().enumerate() {
            if !matches!(f.op(op).kind, OpKind::Bind) {
                return Err("loom evaluation profile expects flat bindings".to_string());
            }
            let held = f.op(op);
            index.insert(held.results[0], position);
            inits.push(held.operands[0]);
        }
        let mut bindings = Bindings {
            definitions: Vec::with_capacity(ops.len()),
            index,
            inputs: Vec::new(),
        };
        for init in inits {
            let dependencies = bindings.entries(f, init);
            bindings.definitions.push(Definition { dependencies, init });
        }
        Ok(bindings)
    }

    /// Interns the values `outputs` read that no binding defines, so the closure
    /// index space covers every input load the profile performs.
    pub(crate) fn intern_outputs(&mut self, f: &Function, outputs: &[Vec<ValueId>]) {
        let mut read = BTreeSet::new();
        for &value in outputs.iter().flatten() {
            f.add_reads(value, &mut read);
        }
        let mut read = read.into_iter().collect::<Vec<_>>();
        read.sort_by(|a, b| f.name(*a).cmp(&f.name(*b)).then(a.cmp(b)));
        for value in read {
            self.intern(value);
        }
    }

    /// The closure entries a value reads, in the order their names sort.
    ///
    /// Which entry an input load interns to is decided by the order a walk first
    /// reaches it, and the dependency sets taken from this space are compared
    /// across compilers and across representations. Ordering by name is what
    /// fixes that space, so the same program interns the same way however its
    /// values are numbered.
    fn entries(&mut self, f: &Function, value: ValueId) -> Vec<usize> {
        reads_in_name_order(f, value)
            .into_iter()
            .map(|read| self.intern(read))
            .collect()
    }

    fn intern(&mut self, value: ValueId) -> usize {
        match self.index.get(&value) {
            Some(&entry) => entry,
            None => {
                let entry = self.index.len();
                self.index.insert(value, entry);
                self.inputs.push(value);
                entry
            }
        }
    }

    /// How many bindings there are: every closure entry below this is one, and
    /// every entry at or above it is an input load.
    pub(crate) fn definitions(&self) -> usize {
        self.definitions.len()
    }

    /// How many input loads the profile performs.
    pub(crate) fn inputs(&self) -> usize {
        self.inputs.len()
    }

    /// What each binding reads, in binding order: closure entries, ascending.
    pub(crate) fn dependency_sets(&self) -> Vec<Vec<usize>> {
        self.definitions
            .iter()
            .map(|definition| {
                let mut entries = definition.dependencies.clone();
                entries.sort_unstable();
                entries.dedup();
                entries
            })
            .collect()
    }

    /// The closure entry a value denotes, binding or input load alike.
    pub(crate) fn entry(&self, value: ValueId) -> Option<usize> {
        self.index.get(&value).copied()
    }

    /// The binding a value denotes, if it is one rather than an input load.
    pub(crate) fn definition(&self, value: ValueId) -> Option<usize> {
        self.entry(value)
            .filter(|&entry| entry < self.definitions.len())
    }

    /// Every input load the profile performs, with the entry it interned to, in
    /// intern order.
    pub(crate) fn input_entries(&self) -> impl Iterator<Item = (ValueId, usize)> {
        self.inputs
            .iter()
            .enumerate()
            .map(|(offset, &value)| (value, self.definitions.len() + offset))
    }

    /// Follows a chain of bindings to the value that ultimately defines it; an
    /// input load is its own definition.
    pub(crate) fn expanded(&self, mut value: ValueId) -> ValueId {
        while let Some(entry) = self.definition(value) {
            let init = self.definitions[entry].init;
            if init == value {
                break;
            }
            value = init;
        }
        value
    }

    /// Adds every binding a value reaches, transitively, to `required`,
    /// recording in `inserted` the entries this call is the one to add.
    pub(crate) fn add_dependencies(
        &self,
        f: &Function,
        value: ValueId,
        required: &mut BTreeSet<usize>,
        inserted: &mut Vec<usize>,
    ) {
        self.add_closure(f, value, false, required, inserted);
    }

    /// [`Bindings::add_dependencies`], with the input loads it reaches joining
    /// the closure as well.
    pub(crate) fn add_dependencies_with_inputs(
        &self,
        f: &Function,
        value: ValueId,
        required: &mut BTreeSet<usize>,
        inserted: &mut Vec<usize>,
    ) {
        self.add_closure(f, value, true, required, inserted);
    }

    fn add_closure(
        &self,
        f: &Function,
        value: ValueId,
        inputs: bool,
        required: &mut BTreeSet<usize>,
        inserted: &mut Vec<usize>,
    ) {
        for read in reads_in_name_order(f, value) {
            if let Some(&entry) = self.index.get(&read) {
                self.add_entry(entry, inputs, required, inserted);
            }
        }
    }

    fn add_entry(
        &self,
        entry: usize,
        inputs: bool,
        required: &mut BTreeSet<usize>,
        inserted: &mut Vec<usize>,
    ) {
        let Some(definition) = self.definitions.get(entry) else {
            if inputs && required.insert(entry) {
                inserted.push(entry);
            }
            return;
        };
        if !required.insert(entry) {
            return;
        }
        inserted.push(entry);
        for &dependency in &definition.dependencies {
            self.add_entry(dependency, inputs, required, inserted);
        }
    }
}

/// The places a value reads, in the order their names sort.
fn reads_in_name_order(f: &Function, value: ValueId) -> Vec<ValueId> {
    let mut read = f.reads(value).into_iter().collect::<Vec<_>>();
    read.sort_by(|a, b| f.name(*a).cmp(&f.name(*b)).then(a.cmp(b)));
    read
}

/// The bindings of `ops` that `roots` reach, in the order `ops` holds them.
///
/// Choosing which steps a body runs is all a slice is: no value is copied and
/// nothing is renumbered, so the ids that come back index the same arena.
pub(in super::super) fn slice_bindings(
    f: &Function,
    ops: &[OpId],
    roots: &[ValueId],
) -> Result<Vec<OpId>, String> {
    let mut needed = BTreeSet::new();
    for &root in roots {
        f.add_reads(root, &mut needed);
    }
    let mut kept = Vec::new();
    for &op in ops.iter().rev() {
        if !matches!(f.op(op).kind, OpKind::Bind) {
            return Err("loom evaluation slicing expects flat bindings".into());
        }
        let defined = f.op(op).results[0];
        if needed.remove(&defined) {
            f.add_reads(f.op(op).operands[0], &mut needed);
            kept.push(op);
        }
    }
    kept.reverse();
    Ok(kept)
}
