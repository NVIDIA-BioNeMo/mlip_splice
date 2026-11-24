// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fluent construction of a [`Module`]. The builder holds a **stack** of open
//! functions; `param`/`op` act on the innermost open function. This supports
//! building a sub-graph (e.g. a `reduce` reducer, from an inline StableHLO
//! region) while an enclosing function is still open: `begin_func` the sub-graph,
//! fill it, `end_func` it to get a [`FuncId`], then reference that id from the
//! enclosing function.
//!
//! Each op method looks up its operand shapes, runs the op's [`Op::infer`] to
//! type the new results (against the signatures of already-completed functions),
//! and appends the node — so a graph is shape-checked as it is built. `finish`
//! runs the full [`verify`] before handing back the module.
//!
//! Op methods panic on a shape error (constructing ill-typed IR is a builder
//! bug). The [`verify`] pass is the graceful checker for IR that
//! arrives by other means.

use smallvec::SmallVec;

use crate::dim_numbers::DotDimensionNumbers;
use crate::ir::{Def, Func, FuncId, Module, Node, ValueDef, ValueId};
use crate::op::{FuncSig, InferCtx, Op};
use crate::ops::{
    Abs, Add, And, Atan2, BroadcastInDim, Cbrt, Ceil, Compare, ComparisonDirection, ComparisonType,
    Constant, Cosine, CountLeadingZeros, Divide, DotAlgorithm, DotGeneral, Exponential,
    ExponentialMinusOne, Floor, GatherAlongMap, IsFinite, Literal, Log, LogPlusOne, Logistic,
    Maximum, Minimum, Multiply, Negate, Not, Or, Popcnt, Power, Reduce, Remainder, RoundNearestAfz,
    RoundNearestEven, Rsqrt, ShiftLeft, ShiftRightArithmetic, ShiftRightLogical, Sign, Sine, Sqrt,
    Subtract, SumAlongMap, Tan, Tanh, Transpose, Xor,
};
use crate::shape::{Dims, Shape, ShapeError};
use crate::verify::{self, VerifyError};

/// An in-progress function on the builder's stack.
struct Partial {
    name: String,
    params: Vec<ValueId>,
    nodes: Vec<Node>,
    values: Vec<ValueDef>,
}

impl Partial {
    fn new(name: &str) -> Partial {
        Partial {
            name: name.to_string(),
            params: Vec::new(),
            nodes: Vec::new(),
            values: Vec::new(),
        }
    }

    fn fresh(&mut self, shape: Shape, def: Def) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(ValueDef { shape, def });
        id
    }

    fn into_func(self, results: Vec<ValueId>) -> Func {
        Func {
            name: self.name,
            params: self.params,
            nodes: self.nodes,
            results,
            values: self.values,
        }
    }
}

#[derive(Default)]
pub struct ModuleBuilder {
    finite_map_graph: domain_index::FiniteMapGraph,
    funcs: Vec<Func>,
    sigs: Vec<FuncSig>,
    stack: Vec<Partial>,
}

impl ModuleBuilder {
    pub fn new() -> ModuleBuilder {
        ModuleBuilder::default()
    }

    pub fn with_finite_map_graph(finite_map_graph: domain_index::FiniteMapGraph) -> ModuleBuilder {
        ModuleBuilder {
            finite_map_graph,
            ..ModuleBuilder::default()
        }
    }

    pub fn with_function_prefix(
        finite_map_graph: domain_index::FiniteMapGraph,
        funcs: Vec<Func>,
    ) -> ModuleBuilder {
        let sigs = funcs
            .iter()
            .map(|func| FuncSig {
                params: func
                    .params
                    .iter()
                    .map(|&param| func.value_shape(param).clone())
                    .collect(),
                results: func
                    .results
                    .iter()
                    .map(|&result| func.value_shape(result).clone())
                    .collect(),
            })
            .collect();
        ModuleBuilder {
            finite_map_graph,
            funcs,
            sigs,
            stack: Vec::new(),
        }
    }

    pub fn finite_map_graph(&self) -> &domain_index::FiniteMapGraph {
        &self.finite_map_graph
    }

    /// Open a new function; subsequent `param`/`op` calls target it.
    pub fn begin_func(&mut self, name: &str) {
        self.stack.push(Partial::new(name));
    }

    /// Close the innermost open function with the given result values, returning
    /// its [`FuncId`]. Its signature becomes visible to later ops.
    pub fn end_func(&mut self, results: Vec<ValueId>) -> FuncId {
        let pf = self
            .stack
            .pop()
            .expect("xir builder: end_func with no open function");
        let func = pf.into_func(results);
        let id = FuncId(self.funcs.len() as u32);
        self.sigs.push(FuncSig {
            params: func
                .params
                .iter()
                .map(|&p| func.value_shape(p).clone())
                .collect(),
            results: func
                .results
                .iter()
                .map(|&r| func.value_shape(r).clone())
                .collect(),
        });
        self.funcs.push(func);
        id
    }

    /// Build a function via a closure — convenience over `begin_func`/`end_func`.
    /// The closure receives the builder (acting on the just-opened function) and
    /// returns the result values.
    pub fn func<F>(&mut self, name: &str, build: F) -> FuncId
    where
        F: FnOnce(&mut ModuleBuilder) -> Vec<ValueId>,
    {
        self.begin_func(name);
        let results = build(self);
        self.end_func(results)
    }

    pub fn finish(self, entry: FuncId) -> Result<Module, VerifyError> {
        assert!(
            self.stack.is_empty(),
            "xir builder: finish with {} function(s) still open",
            self.stack.len()
        );
        let module = Module {
            finite_map_graph: self.finite_map_graph,
            funcs: self.funcs,
            entry,
        };
        verify::verify(&module)?;
        Ok(module)
    }

    fn current(&mut self) -> &mut Partial {
        self.stack
            .last_mut()
            .expect("xir builder: no open function")
    }

    /// Add a parameter to the innermost open function.
    pub fn param(&mut self, shape: Shape) -> ValueId {
        let cur = self.current();
        let idx = cur.params.len();
        let id = cur.fresh(shape, Def::Param(idx));
        cur.params.push(id);
        id
    }

    /// Append any op to the innermost open function, returning all its results,
    /// or a shape error. The shape inference runs against the operand shapes and
    /// the signatures of already-completed functions.
    pub fn try_op(
        &mut self,
        op: impl Op + 'static,
        inputs: &[ValueId],
    ) -> Result<SmallVec<[ValueId; 1]>, ShapeError> {
        self.try_op_boxed(std::sync::Arc::new(op), inputs)
    }

    /// Append an already-boxed op (for the string-keyed factories, where the op
    /// type varies at runtime).
    pub fn try_op_boxed(
        &mut self,
        op: std::sync::Arc<dyn Op>,
        inputs: &[ValueId],
    ) -> Result<SmallVec<[ValueId; 1]>, ShapeError> {
        let input_shapes: SmallVec<[Shape; 4]> = {
            let cur = self.stack.last().expect("xir builder: no open function");
            inputs
                .iter()
                .map(|&v| cur.values[v.index()].shape.clone())
                .collect()
        };
        let result_shapes = {
            let ctx = InferCtx::new(&input_shapes, &self.sigs, &self.finite_map_graph);
            op.infer(&ctx)?
        };
        let cur = self.current();
        let node = cur.nodes.len();
        let results: SmallVec<[ValueId; 1]> = result_shapes
            .into_iter()
            .enumerate()
            .map(|(slot, shape)| cur.fresh(shape, Def::NodeResult { node, slot }))
            .collect();
        cur.nodes.push(Node {
            op,
            inputs: inputs.iter().copied().collect(),
            results: results.clone(),
        });
        Ok(results)
    }

    /// Append any op, panicking on a shape error (constructing ill-typed IR is a
    /// builder bug — see module docs). Use [`try_op`](Self::try_op) to handle the
    /// error.
    pub fn op(&mut self, op: impl Op + 'static, inputs: &[ValueId]) -> SmallVec<[ValueId; 1]> {
        self.try_op(op, inputs)
            .unwrap_or_else(|e| panic!("xir builder: {e}"))
    }

    /// Append a single-result op (panicking variant of [`try_op1`](Self::try_op1)).
    pub fn op1(&mut self, op: impl Op + 'static, inputs: &[ValueId]) -> ValueId {
        let r = self.op(op, inputs);
        assert_eq!(r.len(), 1, "op1 used on a multi-result op");
        r[0]
    }

    /// Append a single-result op, returning a shape error instead of panicking.
    pub fn try_op1(
        &mut self,
        op: impl Op + 'static,
        inputs: &[ValueId],
    ) -> Result<ValueId, ShapeError> {
        let r = self.try_op(op, inputs)?;
        assert_eq!(r.len(), 1, "try_op1 used on a multi-result op");
        Ok(r[0])
    }

    /// Append a single-result already-boxed op, returning a shape error instead
    /// of panicking.
    pub fn try_op1_boxed(
        &mut self,
        op: std::sync::Arc<dyn Op>,
        inputs: &[ValueId],
    ) -> Result<ValueId, ShapeError> {
        let r = self.try_op_boxed(op, inputs)?;
        assert_eq!(r.len(), 1, "try_op1_boxed used on a multi-result op");
        Ok(r[0])
    }

    // ---- convenience wrappers over `op1` for common ops ----

    /// Append the unary elementwise op named by its StableHLO mnemonic. The one
    /// place mapping a unary mnemonic string to its op type.
    pub fn unary(&mut self, name: &str, x: ValueId) -> Result<ValueId, ShapeError> {
        let op = unary_from_name(name)
            .ok_or_else(|| ShapeError::new("unary", format!("unknown unary op {name:?}")))?;
        self.try_op1_boxed(op.into(), &[x])
    }

    /// Append the binary elementwise op named by its StableHLO mnemonic. The one
    /// place mapping a binary mnemonic string to its op type (compare directions
    /// route through [`compare`](Self::compare)).
    pub fn binary(&mut self, name: &str, a: ValueId, b: ValueId) -> Result<ValueId, ShapeError> {
        let op = binary_from_name(name)
            .ok_or_else(|| ShapeError::new("binary", format!("unknown binary op {name:?}")))?;
        self.try_op1_boxed(op.into(), &[a, b])
    }

    pub fn compare(
        &mut self,
        dir: ComparisonDirection,
        compare_type: Option<ComparisonType>,
        a: ValueId,
        b: ValueId,
    ) -> ValueId {
        self.op1(
            Compare {
                comparison_direction: dir,
                compare_type,
            },
            &[a, b],
        )
    }

    pub fn add(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op1(Add, &[a, b])
    }

    pub fn multiply(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op1(Multiply, &[a, b])
    }

    pub fn gather_along_map(
        &mut self,
        value: ValueId,
        map: domain_index::FiniteMapId,
        axis: u32,
    ) -> ValueId {
        self.op1(GatherAlongMap { map, axis }, &[value])
    }

    pub fn sum_along_map(
        &mut self,
        value: ValueId,
        map: domain_index::FiniteMapId,
        axis: u32,
    ) -> ValueId {
        self.op1(SumAlongMap { map, axis }, &[value])
    }

    pub fn maximum(&mut self, a: ValueId, b: ValueId) -> ValueId {
        self.op1(Maximum, &[a, b])
    }

    pub fn dot_general(
        &mut self,
        lhs: ValueId,
        rhs: ValueId,
        dnums: DotDimensionNumbers,
        algorithm: DotAlgorithm,
    ) -> ValueId {
        self.op1(DotGeneral { dnums, algorithm }, &[lhs, rhs])
    }

    pub fn transpose(&mut self, x: ValueId, permutation: &[u32]) -> ValueId {
        self.op1(
            Transpose {
                permutation: SmallVec::from_slice(permutation),
            },
            &[x],
        )
    }

    pub fn broadcast_in_dim(
        &mut self,
        x: ValueId,
        out_dims: &[i64],
        broadcast_dimensions: &[u32],
    ) -> ValueId {
        self.op1(
            BroadcastInDim {
                out_dims: Dims::from_slice(out_dims),
                broadcast_dimensions: SmallVec::from_slice(broadcast_dimensions),
            },
            &[x],
        )
    }

    pub fn constant(&mut self, literal: Literal) -> ValueId {
        self.op1(Constant { literal }, &[])
    }

    pub fn reduce(
        &mut self,
        operand: ValueId,
        init: ValueId,
        dimensions: &[u32],
        reducer: FuncId,
    ) -> ValueId {
        self.op1(
            Reduce {
                dimensions: SmallVec::from_slice(dimensions),
                reducer,
            },
            &[operand, init],
        )
    }
}

/// Construct a unary elementwise op from its StableHLO mnemonic. The single
/// name→type map for unary ops, shared by the builder and the Python bindings.
pub fn unary_from_name(name: &str) -> Option<Box<dyn Op>> {
    Some(match name {
        "negate" => Box::new(Negate),
        "abs" => Box::new(Abs),
        "exponential" => Box::new(Exponential),
        "exponential_minus_one" => Box::new(ExponentialMinusOne),
        "log" => Box::new(Log),
        "log_plus_one" => Box::new(LogPlusOne),
        "sqrt" => Box::new(Sqrt),
        "rsqrt" => Box::new(Rsqrt),
        "cbrt" => Box::new(Cbrt),
        "tanh" => Box::new(Tanh),
        "sine" => Box::new(Sine),
        "cosine" => Box::new(Cosine),
        "tan" => Box::new(Tan),
        "floor" => Box::new(Floor),
        "ceil" => Box::new(Ceil),
        "round_nearest_even" => Box::new(RoundNearestEven),
        "round_nearest_afz" => Box::new(RoundNearestAfz),
        "sign" => Box::new(Sign),
        "not" => Box::new(Not),
        "count_leading_zeros" => Box::new(CountLeadingZeros),
        "popcnt" => Box::new(Popcnt),
        "is_finite" => Box::new(IsFinite),
        "logistic" => Box::new(Logistic),
        _ => return None,
    })
}

/// Construct a binary elementwise op from its StableHLO mnemonic. The single
/// name→type map for binary ops, shared by the builder and the Python bindings.
/// `compare` is excluded — it carries a direction (use [`ModuleBuilder::compare`]).
pub fn binary_from_name(name: &str) -> Option<Box<dyn Op>> {
    Some(match name {
        "add" => Box::new(Add),
        "subtract" => Box::new(Subtract),
        "multiply" => Box::new(Multiply),
        "divide" => Box::new(Divide),
        "maximum" => Box::new(Maximum),
        "minimum" => Box::new(Minimum),
        "power" => Box::new(Power),
        "remainder" => Box::new(Remainder),
        "and" => Box::new(And),
        "or" => Box::new(Or),
        "xor" => Box::new(Xor),
        "shift_left" => Box::new(ShiftLeft),
        "shift_right_arithmetic" => Box::new(ShiftRightArithmetic),
        "shift_right_logical" => Box::new(ShiftRightLogical),
        "atan2" => Box::new(Atan2),
        _ => return None,
    })
}
