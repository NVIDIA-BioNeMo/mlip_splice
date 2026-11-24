// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;

use gpu_ir::{Body, Builder, Param, ParamTy, Scalar, Ty, Value, ValueId};

use super::super::*;

/// A profile written by hand: free inputs are parameters, bindings are named,
/// and the names are how a test says which value it means.
///
/// Values belong to the arena that defined them, so a test that wants to talk
/// about one builds it here and asks for it by name.
pub(super) struct Sketch {
    b: Builder,
    named: BTreeMap<String, Value>,
}

impl Default for Sketch {
    fn default() -> Sketch {
        Sketch::new()
    }
}

impl Sketch {
    pub(super) fn new() -> Sketch {
        Sketch {
            b: Builder::new(),
            named: BTreeMap::new(),
        }
    }

    pub(super) fn builder(&mut self) -> &mut Builder {
        &mut self.b
    }

    /// A free `f32` the profile reads and nothing defines. One name is one
    /// input, so asking twice is asking for the same one.
    pub(super) fn input(&mut self, name: &str) -> Value {
        if let Some(value) = self.named.get(name) {
            return value.clone();
        }
        let value = self.b.parameter(Param {
            name: name.to_string(),
            ty: ParamTy::Scalar(Scalar::F32),
        });
        self.named.insert(name.to_string(), value.clone());
        value
    }

    /// The value a name stands for, input or binding alike.
    pub(super) fn get(&self, name: &str) -> Value {
        self.named
            .get(name)
            .unwrap_or_else(|| panic!("{name} was never bound"))
            .clone()
    }

    /// An `f32` binding under a name a test reads it back by.
    pub(super) fn bind(&mut self, name: &str, init: Value) -> Value {
        let bound = self
            .b
            .bind_value_as(name.to_string(), Ty::scalar(Scalar::F32), init);
        self.named.insert(name.to_string(), bound.clone());
        bound
    }

    pub(super) fn add(&mut self, a: Value, b: Value) -> Value {
        self.b.add(a, b)
    }

    pub(super) fn mul(&mut self, a: Value, b: Value) -> Value {
        self.b.mul(a, b)
    }

    pub(super) fn sub(&mut self, a: Value, b: Value) -> Value {
        self.b.sub(a, b)
    }

    pub(super) fn f32(&mut self, value: f64) -> Value {
        self.b.float(Scalar::F32, value)
    }

    pub(super) fn finish(self) -> Sketched {
        let named = self
            .named
            .iter()
            .map(|(name, value)| (name.clone(), value.id_in(self.b.function())))
            .collect();
        Sketched {
            body: self.b.into_function().into(),
            named,
        }
    }
}

/// A finished sketch: the body it built, and the value each name stands for.
pub(super) struct Sketched {
    pub body: Body,
    pub named: BTreeMap<String, ValueId>,
}

impl Sketched {
    pub(super) fn function(&self) -> &gpu_ir::Function {
        self.body.function()
    }

    pub(super) fn ops(&self) -> Vec<gpu_ir::OpId> {
        self.body.steps()
    }

    pub(super) fn id(&self, name: &str) -> ValueId {
        *self
            .named
            .get(name)
            .unwrap_or_else(|| panic!("{name} was never bound"))
    }

    pub(super) fn bindings(&self) -> Bindings {
        Bindings::of(self.function(), &self.ops()).unwrap()
    }
}

pub(super) fn selection(element: usize, terms: Vec<usize>) -> ElementPart {
    let term_guards = vec![None; terms.len()];
    ElementPart {
        element,
        terms,
        continuation: false,
        read_later: false,
        reduction: None,
        guard: None,
        term_guards,
    }
}

pub(super) fn incidence(sketched: Sketched, outputs: Vec<Vec<ValueId>>) -> IncidenceProfile {
    let ops = sketched.body.steps().to_vec();
    let mut bindings = Bindings::of(sketched.body.function(), &ops).unwrap();
    bindings.intern_outputs(sketched.body.function(), &outputs);
    let guards = outputs
        .iter()
        .map(|values| vec![None; values.len()])
        .collect();
    IncidenceProfile {
        outputs,
        body: sketched.body,
        ops,
        bindings,
        guards,
        statement_guards: BTreeMap::new(),
        // A sketch's free inputs stand for themselves rather than for an
        // element of an operand, so nothing here has traffic to weigh.
        input_operands: BTreeMap::new(),
    }
}
