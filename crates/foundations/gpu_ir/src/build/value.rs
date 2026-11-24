// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The operations a builder performs to make one value from others.
//!
//! Every operation is a method, and every operand is a value the same builder
//! defined. There is no operator sugar and no way to build a value without a
//! builder to put it in.
//!
//! An operation recorded here is an *operand*: it is evaluated wherever it is
//! read, so reading it twice computes it twice. [`Builder::bind_value`] is what
//! turns one into a place the body computes once.

use std::sync::Arc;

use super::Builder;
use super::owner::Value;
use crate::arena::ValueId;
use crate::op::{ConstArray, OpKind};
use crate::{BinOp, CmpOp, Evict, PtrTy, Scalar, Sreg, Ty, UnOp, ValTy};

impl Builder {
    /// What a value holds.
    pub fn value_ty(&self, value: &Value) -> ValTy {
        self.function().ty(value.id_in(self.function()))
    }

    /// The handle for a value of this arena, for a caller holding ids an
    /// analysis over the same arena produced.
    ///
    /// The scope a handle carries is what stops a value being read outside the
    /// region that bound it. An id from an analysis carries no scope, so this
    /// stamps the one being built; the analyses that produce them read the whole
    /// body at once, where that question does not arise.
    pub fn value_of(&self, id: ValueId) -> Value {
        let _ = self.function().ty(id);
        self.held(id)
    }

    /// The integer this value is a constant of, if it is one. A constant index
    /// folds a whole branch of an indexed expression away, so a caller that can
    /// specialise on one asks here.
    pub fn as_int_const(&self, value: &Value) -> Option<i64> {
        let f = self.function();
        let op = f.operand_op(value.id_in(f))?;
        match f.op(op).kind {
            OpKind::IntConst { value, .. } => Some(value),
            _ => None,
        }
    }

    fn ty_of(&self, value: &Value) -> ValTy {
        self.value_ty(value)
    }

    fn data_ty_of(&self, value: &Value) -> Ty {
        self.ty_of(value).data()
    }

    fn unary(&mut self, kind: OpKind, a: Value, ty: ValTy) -> Value {
        let a = self.read(a);
        self.operand(kind, vec![a], ty)
    }

    // ---- what a body starts from -------------------------------------

    pub fn int(&mut self, scalar: Scalar, value: i64) -> Value {
        assert!(
            !scalar.is_float() && scalar != Scalar::Pred,
            "integer constant of {scalar:?}"
        );
        self.constant(
            OpKind::IntConst { scalar, value },
            ValTy::Data(Ty::scalar(scalar)),
        )
    }

    pub fn i32(&mut self, value: i64) -> Value {
        assert!(i32::try_from(value).is_ok(), "i32 constant {value}");
        self.int(Scalar::S32, value)
    }

    pub fn i64(&mut self, value: i64) -> Value {
        self.int(Scalar::S64, value)
    }

    pub fn float(&mut self, scalar: Scalar, value: f64) -> Value {
        assert!(scalar.is_float(), "float constant of {scalar:?}");
        self.constant(
            OpKind::FloatConst { scalar, value },
            ValTy::Data(Ty::scalar(scalar)),
        )
    }

    /// The zero of `scalar`, as the right constant kind.
    pub fn zero(&mut self, scalar: Scalar) -> Value {
        if scalar.is_float() {
            self.float(scalar, 0.0)
        } else {
            self.constant(
                OpKind::IntConst { scalar, value: 0 },
                ValTy::Data(Ty::scalar(scalar)),
            )
        }
    }

    pub fn sreg(&mut self, sreg: Sreg) -> Value {
        self.operand(
            OpKind::Sreg(sreg),
            Vec::new(),
            ValTy::Data(Ty::scalar(Scalar::U32)),
        )
    }

    pub fn tid_x(&mut self) -> Value {
        self.sreg(Sreg::TidX)
    }

    pub fn ctaid_x(&mut self) -> Value {
        self.sreg(Sreg::CtaidX)
    }

    pub fn ctaid_y(&mut self) -> Value {
        self.sreg(Sreg::CtaidY)
    }

    pub fn ntid_x(&mut self) -> Value {
        self.sreg(Sreg::NtidX)
    }

    // ---- memory ------------------------------------------------------

    /// `pointer + elements`, in element units.
    pub fn offset(&mut self, pointer: Value, elements: Value) -> Value {
        let ty = self.ty_of(&pointer);
        let (pointer, elements) = (self.read(pointer), self.read(elements));
        self.operand(OpKind::Offset, vec![pointer, elements], ty)
    }

    pub fn load(&mut self, ptr: Value, lanes: u8, nc: bool) -> Value {
        self.load_evict(ptr, lanes, nc, Evict::Normal)
    }

    pub fn load_evict(&mut self, ptr: Value, lanes: u8, nc: bool, evict: Evict) -> Value {
        let elem = self.ty_of(&ptr).ptr().elem;
        self.unary(
            OpKind::Load { lanes, nc, evict },
            ptr,
            ValTy::Data(Ty::vec(elem, lanes)),
        )
    }

    /// A constant the renderer emits under `name`. That name is a `.const`
    /// symbol, not a value this body binds, so it is the caller's to choose.
    pub fn const_load(
        &mut self,
        name: impl Into<String>,
        scalar: Scalar,
        data: Vec<u8>,
        index: Value,
    ) -> Value {
        self.const_load_shared(name, scalar, data.into(), index)
    }

    pub fn const_load_shared(
        &mut self,
        name: impl Into<String>,
        scalar: Scalar,
        data: Arc<[u8]>,
        index: Value,
    ) -> Value {
        let array = Arc::new(ConstArray {
            name: name.into(),
            scalar,
            data,
        });
        self.unary(
            OpKind::ConstLoad(array),
            index,
            ValTy::Data(Ty::scalar(scalar)),
        )
    }

    // ---- arithmetic --------------------------------------------------

    pub fn un(&mut self, op: UnOp, a: Value) -> Value {
        let ty = self.ty_of(&a);
        self.unary(OpKind::Un(op), a, ty)
    }

    pub fn neg(&mut self, a: Value) -> Value {
        self.un(UnOp::Neg, a)
    }

    pub fn not(&mut self, a: Value) -> Value {
        self.un(UnOp::Not, a)
    }

    pub fn sqrt(&mut self, a: Value) -> Value {
        self.un(UnOp::Sqrt, a)
    }

    pub fn bin(&mut self, op: BinOp, a: Value, b: Value) -> Value {
        let ty = self.ty_of(&a);
        let (a, b) = (self.read(a), self.read(b));
        self.operand(OpKind::Bin(op), vec![a, b], ty)
    }

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Add, a, b)
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Sub, a, b)
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Mul, a, b)
    }

    pub fn div(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Div, a, b)
    }

    pub fn rem(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Rem, a, b)
    }

    pub fn min(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Min, a, b)
    }

    pub fn max(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Max, a, b)
    }

    pub fn and(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::And, a, b)
    }

    pub fn or(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Or, a, b)
    }

    pub fn xor(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Xor, a, b)
    }

    pub fn shl(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Shl, a, b)
    }

    pub fn shr(&mut self, a: Value, b: Value) -> Value {
        self.bin(BinOp::Shr, a, b)
    }

    pub fn fma(&mut self, a: Value, b: Value, c: Value) -> Value {
        let ty = self.ty_of(&a);
        let (a, b, c) = (self.read(a), self.read(b), self.read(c));
        self.operand(OpKind::Fma, vec![a, b, c], ty)
    }

    pub fn cmp(&mut self, op: CmpOp, a: Value, b: Value) -> Value {
        let lanes = self.data_ty_of(&a).lanes;
        let (a, b) = (self.read(a), self.read(b));
        self.operand(
            OpKind::Cmp(op),
            vec![a, b],
            ValTy::Data(Ty::vec(Scalar::Pred, lanes)),
        )
    }

    pub fn eq(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Eq, a, b)
    }

    pub fn ne(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Ne, a, b)
    }

    pub fn lt(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Lt, a, b)
    }

    pub fn le(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Le, a, b)
    }

    pub fn gt(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Gt, a, b)
    }

    pub fn ge(&mut self, a: Value, b: Value) -> Value {
        self.cmp(CmpOp::Ge, a, b)
    }

    pub fn select(&mut self, pred: Value, a: Value, b: Value) -> Value {
        let ty = self.ty_of(&a);
        let (pred, a, b) = (self.read(pred), self.read(a), self.read(b));
        self.operand(OpKind::Select, vec![pred, a, b], ty)
    }

    // ---- shape -------------------------------------------------------

    pub fn cast(&mut self, scalar: Scalar, a: Value) -> Value {
        let lanes = self.data_ty_of(&a).lanes;
        self.unary(OpKind::Cast(scalar), a, ValTy::Data(Ty::vec(scalar, lanes)))
    }

    pub fn splat(&mut self, lanes: u8, a: Value) -> Value {
        let scalar = self.data_ty_of(&a).scalar;
        self.unary(OpKind::Splat(lanes), a, ValTy::Data(Ty::vec(scalar, lanes)))
    }

    pub fn extract(&mut self, lane: u8, a: Value) -> Value {
        let scalar = self.data_ty_of(&a).scalar;
        self.unary(OpKind::Extract(lane), a, ValTy::Data(Ty::scalar(scalar)))
    }

    /// One value holding its parts' lanes end to end.
    ///
    /// A part may itself be a vector, so this is concatenation rather than a
    /// gather of scalars: what a caller means by packing chunk accumulators is
    /// the same operation as packing lanes, and stating it once keeps the two
    /// from drifting apart.
    pub fn pack(&mut self, parts: Vec<Value>) -> Value {
        let held = self.data_ty_of(parts.first().expect("non-empty pack"));
        let mut count: u8 = 0;
        for part in &parts {
            let ty = self.data_ty_of(part);
            assert_eq!(ty.scalar, held.scalar, "pack of mixed scalars");
            count = count.checked_add(ty.lanes).expect("pack of too many lanes");
        }
        let parts = parts
            .into_iter()
            .map(|part| self.read(part))
            .collect::<Vec<_>>();
        self.operand(
            OpKind::Pack,
            parts,
            ValTy::Data(Ty::vec(held.scalar, count)),
        )
    }

    // ---- warp --------------------------------------------------------

    pub fn shfl_down(&mut self, a: Value, delta: Value) -> Value {
        let ty = self.ty_of(&a);
        let (a, delta) = (self.read(a), self.read(delta));
        self.operand(OpKind::ShflDown, vec![a, delta], ty)
    }

    /// The lanes of this warp participating here, one bit each.
    pub fn active_mask(&mut self) -> Value {
        self.operand(
            OpKind::ActiveMask,
            Vec::new(),
            ValTy::Data(Ty::scalar(Scalar::U32)),
        )
    }

    /// `a` as `source_lane` holds it, over exactly the lanes `participants`
    /// names. The lane is absolute, so up, down and broadcast are lane
    /// arithmetic over this one operation.
    pub fn shfl_from(&mut self, a: Value, source_lane: Value, participants: Value) -> Value {
        let ty = self.ty_of(&a);
        let (a, source_lane, participants) = (
            self.read(a),
            self.read(source_lane),
            self.read(participants),
        );
        self.operand(OpKind::ShflFrom, vec![a, source_lane, participants], ty)
    }

    pub fn warp_any(&mut self, a: Value) -> Value {
        self.unary(OpKind::WarpAny, a, ValTy::Data(Ty::scalar(Scalar::Pred)))
    }

    // ---- places ------------------------------------------------------

    /// Computes `init` here, once, and gives back the value that stands for it.
    pub fn bind_value(&mut self, ty: Ty, init: Value) -> Value {
        self.bind(ValTy::Data(ty), init, None)
    }

    /// As [`Builder::bind_value`], under a name the caller chose.
    ///
    /// A name carries meaning the arena does not — which stage a value belongs
    /// to, which output it addresses — so a caller that has such a name says it
    /// here rather than reading it back off a definition later.
    pub fn bind_value_as(&mut self, name: impl Into<String>, ty: Ty, init: Value) -> Value {
        let bound = self.bind(ValTy::Data(ty), init, None);
        self.name_value(&bound, name);
        bound
    }

    /// Binds a pointer, for a value whose type is a pointer rather than a datum.
    pub fn bind_pointer(&mut self, ty: PtrTy, init: Value) -> Value {
        self.bind(ValTy::Ptr(ty), init, None)
    }

    /// As [`Builder::bind_pointer`], under a name the caller chose.
    pub fn bind_pointer_as(&mut self, name: impl Into<String>, ty: PtrTy, init: Value) -> Value {
        let bound = self.bind(ValTy::Ptr(ty), init, None);
        self.name_value(&bound, name);
        bound
    }

    /// Computes `init` into the storage `place` holds, and gives back the value
    /// that stands for it from here on.
    ///
    /// This is one value per definition, as everything else is — `place` is not
    /// written through, it is *replaced*, and every later read names the value
    /// this returns. What is shared is the storage, which is what an accumulator
    /// updated in a straight line means and what keeps it in one register.
    pub fn update(&mut self, place: Value, init: Value) -> Value {
        let ty = self.ty_of(&place);
        assert_eq!(
            ty,
            self.ty_of(&init),
            "an update changes what the place holds"
        );
        self.bind(ty, init, Some(place))
    }

    fn bind(&mut self, ty: ValTy, init: Value, place: Option<Value>) -> Value {
        let init = self.read(init);
        let into = match place {
            Some(place) => vec![Some(self.read(place))],
            None => Vec::new(),
        };
        let op = self.place(OpKind::Bind, vec![init], &[ty], into, Vec::new());
        let id = self.function().op(op).results[0];
        self.held(id)
    }
}
