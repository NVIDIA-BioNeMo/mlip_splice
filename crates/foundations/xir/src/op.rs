// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The open op registry. Every operation — built-in or future custom —
//! implements [`Op`]. There is no privileged closed enum, so a custom op is as
//! first-class as `dot`: it supplies the same shape inference, sub-graph
//! references, and printing hooks.

use domain_index::FiniteMapGraph;
use smallvec::SmallVec;

use crate::ir::FuncId;
use crate::shape::{Shape, ShapeError};

/// The signature of a function: the shapes of its parameters and results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FuncSig {
    pub params: Vec<Shape>,
    pub results: Vec<Shape>,
}

/// What an op's [`Op::infer`] sees: the operand shapes, plus the signatures of
/// already-defined functions (so higher-order ops like `call`/`if` can take
/// their result shapes from the sub-graphs they invoke). Functions are always
/// built before the op that references them, so every referenced [`FuncId`] is
/// resolvable here.
pub struct InferCtx<'a> {
    inputs: &'a [Shape],
    sigs: &'a [FuncSig],
    finite_map_graph: &'a FiniteMapGraph,
}

impl<'a> InferCtx<'a> {
    pub fn new(
        inputs: &'a [Shape],
        sigs: &'a [FuncSig],
        finite_map_graph: &'a FiniteMapGraph,
    ) -> InferCtx<'a> {
        InferCtx {
            inputs,
            sigs,
            finite_map_graph,
        }
    }

    pub fn inputs(&self) -> &[Shape] {
        self.inputs
    }

    pub fn func_params(&self, f: FuncId) -> &[Shape] {
        &self.sigs[f.index()].params
    }

    pub fn func_results(&self, f: FuncId) -> &[Shape] {
        &self.sigs[f.index()].results
    }

    pub fn func_sig(&self, f: FuncId) -> Option<&FuncSig> {
        self.sigs.get(f.index())
    }

    pub fn finite_map_graph(&self) -> &FiniteMapGraph {
        self.finite_map_graph
    }
}

pub trait Op: std::fmt::Debug + Send + Sync {
    /// Short text name, e.g. `"dot"` or `"transpose"`.
    fn mnemonic(&self) -> &'static str;

    /// Shape inference and operand type-check: from the operand shapes (and, for
    /// higher-order ops, the signatures of called functions) return the shape of
    /// each result. This is the single oracle the builder uses to type new
    /// values and the verifier re-runs to check them.
    fn infer(&self, ctx: &InferCtx<'_>) -> Result<SmallVec<[Shape; 1]>, ShapeError>;

    /// Sub-graphs this op invokes (a reducer, a loop body, ...). Empty for
    /// first-order ops.
    fn called_funcs(&self) -> &[FuncId] {
        &[]
    }

    /// Named attributes for the text printer, e.g. `[("perm", "[1, 0]")]`.
    fn attrs(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }

    /// Escape hatch for future rewrite passes to recover the concrete op type.
    fn as_any(&self) -> &dyn std::any::Any;

    /// View this op as elementwise (scalar math applied pointwise), when it is
    /// one. The default `None` covers every non-elementwise op; the elementwise
    /// ops override it to `Some(self)`.
    fn as_elementwise(&self) -> Option<&dyn Elementwise> {
        None
    }

    /// Deep clone behind the trait object; keeps `Module`/`Func` cloneable.
    fn clone_box(&self) -> Box<dyn Op>;
}

/// An elementwise op: it applies one [`ScalarOp`](crate::ops::ScalarOp)
/// pointwise. Lowering reads `scalar()` and need not know the concrete op type.
pub trait Elementwise: Op {
    fn scalar(&self) -> crate::ops::ScalarOp;
}

impl Clone for Box<dyn Op> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Declare first-order ops with a fixed mnemonic. Generates the struct and the
/// boilerplate half of `impl Op` (`mnemonic`, `as_any`, `clone_box`); the
/// per-op `infer` and `attrs` bodies are written inline (each shape rule is
/// genuinely different — the macro removes scaffolding, not the math). The
/// `infer` body binds `inputs: &[Shape]` (the operand shapes).
///
/// Higher-order ops (those with `called_funcs`) and the kind-parameterized
/// elementwise ops are written by hand in [`crate::ops`].
macro_rules! def_ops {
    ($(
        $(#[$meta:meta])*
        $name:ident { $( $fname:ident : $fty:ty ),* $(,)? }
        mnemonic = $mn:literal
        attrs = | $this_a:ident | $attrs:block
        infer = | $this_i:ident, $inputs:ident | $infer:block
    )*) => {$(
        $(#[$meta])*
        #[derive(Clone, Debug, PartialEq)]
        pub struct $name { $( pub $fname : $fty, )* }

        impl $crate::op::Op for $name {
            fn mnemonic(&self) -> &'static str { $mn }

            fn infer(&self, ctx: &$crate::op::InferCtx<'_>)
                -> ::core::result::Result<
                    ::smallvec::SmallVec<[$crate::shape::Shape; 1]>,
                    $crate::shape::ShapeError,
                >
            {
                // The op body comes from the macro call site, so it cannot name
                // the method's `self` (hygiene); we hand it an explicit alias.
                #[allow(unused_variables)]
                let $this_i = self;
                let $inputs: &[$crate::shape::Shape] = ctx.inputs();
                $infer
            }

            fn attrs(&self) -> ::std::vec::Vec<(&'static str, ::std::string::String)> {
                #[allow(unused_variables)]
                let $this_a = self;
                $attrs
            }

            fn as_any(&self) -> &dyn ::std::any::Any { self }

            fn clone_box(&self) -> ::std::boxed::Box<dyn $crate::op::Op> {
                ::std::boxed::Box::new(::core::clone::Clone::clone(self))
            }
        }
    )*};
}

pub(crate) use def_ops;
