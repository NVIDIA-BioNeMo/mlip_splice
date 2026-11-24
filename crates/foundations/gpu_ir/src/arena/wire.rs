// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The arena written down, and those bytes read back as the same arena under a
//! fresh tag.
//!
//! Only what the arena states is written: ids are dense indices, so they go out
//! as they stand and come back range-checked. What a program *means* is not
//! decided here — a decoded function is structurally addressable and nothing
//! more, which is why a caller reading bytes it did not write runs
//! [`crate::verify()`] over the result before believing it.
//!
//! `Arc`-shared callees and constant arrays are pooled by identity in first-use
//! order, so a body that calls one function ten times writes it once and reads
//! back ten uses of one `Arc`.

use std::fmt;
use std::sync::Arc;

use super::{
    ArenaTag, BlockDef, BlockId, Declaration, Def, Function, OpDef, OpId, RegionDef, RegionId,
    Terminator, ValueDef, ValueId,
};
use crate::effect::Sync;
use crate::op::{ConstArray, OpKind, Trip, WriteMode};
use crate::{
    Access, BinOp, CmpOp, DeviceFunction, Evict, Implementation, Param, ParamTy, PtrTy, Scalar,
    SharedArr, Signature, Space, Sreg, Ty, UnOp, ValTy,
};

/// Why bytes are not an arena. Every one of these is a fact about the bytes, so
/// none of them can be answered by looking at the program they claim to be.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WireError {
    Truncated {
        needed: usize,
        left: usize,
    },
    Trailing {
        left: usize,
    },
    UnknownTag {
        what: &'static str,
        tag: u64,
    },
    OutOfRange {
        what: &'static str,
        index: u64,
        count: usize,
    },
    NotUtf8,
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WireError::Truncated { needed, left } => {
                write!(f, "{needed} bytes were asked for and {left} remain")
            }
            WireError::Trailing { left } => {
                write!(f, "the arena ends with {left} bytes after it")
            }
            WireError::UnknownTag { what, tag } => {
                write!(f, "{tag} is no {what} this language has")
            }
            WireError::OutOfRange { what, index, count } => {
                write!(f, "{what} {index} is named where there are {count}")
            }
            WireError::NotUtf8 => f.write_str("a name is not text"),
        }
    }
}

impl std::error::Error for WireError {}

type Result<T> = std::result::Result<T, WireError>;

struct Sink(Vec<u8>);

impl Sink {
    fn u8(&mut self, value: u8) {
        self.0.push(value);
    }

    fn u32(&mut self, value: u32) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.0.extend_from_slice(&value.to_le_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.u64(value.cast_unsigned());
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn count(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn text(&mut self, value: &str) {
        self.count(value.len());
        self.0.extend_from_slice(value.as_bytes());
    }

    fn blob(&mut self, value: &[u8]) {
        self.count(value.len());
        self.0.extend_from_slice(value);
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn left(&self) -> usize {
        self.bytes.len() - self.at
    }

    fn take(&mut self, nbytes: usize) -> Result<&'a [u8]> {
        if nbytes > self.left() {
            return Err(WireError::Truncated {
                needed: nbytes,
                left: self.left(),
            });
        }
        let taken = &self.bytes[self.at..self.at + nbytes];
        self.at += nbytes;
        Ok(taken)
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().expect("four")))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().expect("eight")))
    }

    fn i64(&mut self) -> Result<i64> {
        Ok(self.u64()?.cast_signed())
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_bits(self.u64()?))
    }

    /// A length, refused before it is allocated for. Nothing written here has
    /// an empty element, so a count past the bytes that remain is a count these
    /// bytes could not hold — which is what running out early looks like.
    fn count(&mut self) -> Result<usize> {
        let count = self.u64()?;
        let left = self.left();
        if count > left as u64 {
            return Err(WireError::Truncated {
                needed: count as usize,
                left,
            });
        }
        Ok(count as usize)
    }

    fn index(&mut self, what: &'static str, count: usize) -> Result<usize> {
        let index = self.u64()?;
        if index >= count as u64 {
            return Err(WireError::OutOfRange { what, index, count });
        }
        Ok(index as usize)
    }

    fn indices(&mut self, what: &'static str, count: usize) -> Result<Vec<usize>> {
        let length = self.count()?;
        (0..length).map(|_| self.index(what, count)).collect()
    }

    fn text(&mut self) -> Result<String> {
        let length = self.count()?;
        String::from_utf8(self.take(length)?.to_vec()).map_err(|_| WireError::NotUtf8)
    }

    fn blob(&mut self) -> Result<&'a [u8]> {
        let length = self.count()?;
        self.take(length)
    }

    fn end(&self) -> Result<()> {
        match self.left() {
            0 => Ok(()),
            left => Err(WireError::Trailing { left }),
        }
    }
}

/// A tag per variant of a payload-free enum, in the order listed. The match
/// carries no value: it is there so adding a variant without listing it here
/// fails to compile rather than to encode.
macro_rules! plain_enum {
    ($write:ident, $read:ident, $ty:ident, $what:literal, [$($variant:ident),+ $(,)?]) => {
        fn $write(sink: &mut Sink, value: $ty) {
            const ALL: &[$ty] = &[$($ty::$variant),+];
            match value {
                $($ty::$variant => ()),+
            }
            let tag = ALL
                .iter()
                .position(|held| *held == value)
                .expect("every variant is listed");
            sink.u8(tag as u8);
        }

        fn $read(reader: &mut Reader<'_>) -> Result<$ty> {
            const ALL: &[$ty] = &[$($ty::$variant),+];
            let tag = reader.u8()?;
            ALL.get(usize::from(tag))
                .copied()
                .ok_or(WireError::UnknownTag { what: $what, tag: u64::from(tag) })
        }
    };
}

plain_enum!(
    write_scalar,
    read_scalar,
    Scalar,
    "scalar",
    [
        Pred, S8, U8, S16, U16, S32, U32, S64, U64, F16, BF16, F32, F64
    ]
);
plain_enum!(
    write_space,
    read_space,
    Space,
    "memory space",
    [Global, Shared]
);
plain_enum!(
    write_evict,
    read_evict,
    Evict,
    "eviction priority",
    [Normal, First, Last]
);
plain_enum!(
    write_sreg,
    read_sreg,
    Sreg,
    "special register",
    [TidX, CtaidX, CtaidY, NtidX]
);
plain_enum!(
    write_un,
    read_un,
    UnOp,
    "unary operator",
    [
        Neg,
        Not,
        Sqrt,
        Exp2,
        Log2,
        Sin,
        Cos,
        Tanh,
        Floor,
        Ceil,
        RoundEven,
        LeadingZeros,
        OneBits
    ]
);
plain_enum!(
    write_bin,
    read_bin,
    BinOp,
    "binary operator",
    [Add, Sub, Mul, Div, Rem, Min, Max, And, Or, Xor, Shl, Shr]
);
plain_enum!(
    write_cmp,
    read_cmp,
    CmpOp,
    "comparison",
    [Eq, Ne, Lt, Le, Gt, Ge]
);
plain_enum!(write_mode, read_mode, WriteMode, "write mode", [Set, Add]);
plain_enum!(
    write_trip,
    read_trip,
    Trip,
    "trip count",
    [PerThread, BlockUniform]
);
plain_enum!(
    write_sync,
    read_sync,
    Sync,
    "synchronisation",
    [Warp, Block]
);

/// The byte a scalar is written as, for a format that carries one of its own.
/// There is one such table and this is it.
pub fn scalar_tag(scalar: Scalar) -> u8 {
    let mut sink = Sink(Vec::new());
    write_scalar(&mut sink, scalar);
    sink.0[0]
}

pub fn scalar_from_tag(tag: u8) -> Option<Scalar> {
    read_scalar(&mut Reader {
        bytes: &[tag],
        at: 0,
    })
    .ok()
}

fn write_ty(sink: &mut Sink, ty: Ty) {
    write_scalar(sink, ty.scalar);
    sink.u8(ty.lanes);
}

fn read_ty(reader: &mut Reader<'_>) -> Result<Ty> {
    let scalar = read_scalar(reader)?;
    let lanes = reader.u8()?;
    if lanes == 0 {
        return Err(WireError::UnknownTag {
            what: "lane count",
            tag: 0,
        });
    }
    Ok(Ty { scalar, lanes })
}

fn write_valty(sink: &mut Sink, ty: ValTy) {
    match ty {
        ValTy::Data(ty) => {
            sink.u8(0);
            write_ty(sink, ty);
        }
        ValTy::Ptr(ty) => {
            sink.u8(1);
            write_scalar(sink, ty.elem);
            write_space(sink, ty.space);
        }
    }
}

fn read_valty(reader: &mut Reader<'_>) -> Result<ValTy> {
    match reader.u8()? {
        0 => Ok(ValTy::Data(read_ty(reader)?)),
        1 => Ok(ValTy::Ptr(PtrTy {
            elem: read_scalar(reader)?,
            space: read_space(reader)?,
        })),
        tag => Err(WireError::UnknownTag {
            what: "value type",
            tag: u64::from(tag),
        }),
    }
}

fn write_param(sink: &mut Sink, param: &Param) {
    sink.text(&param.name);
    match param.ty {
        ParamTy::Ptr(elem) => {
            sink.u8(0);
            write_scalar(sink, elem);
        }
        ParamTy::Scalar(scalar) => {
            sink.u8(1);
            write_scalar(sink, scalar);
        }
    }
}

fn read_param(reader: &mut Reader<'_>) -> Result<Param> {
    let name = reader.text()?;
    let ty = match reader.u8()? {
        0 => ParamTy::Ptr(read_scalar(reader)?),
        1 => ParamTy::Scalar(read_scalar(reader)?),
        tag => {
            return Err(WireError::UnknownTag {
                what: "parameter type",
                tag: u64::from(tag),
            });
        }
    };
    Ok(Param { name, ty })
}

fn write_declaration(sink: &mut Sink, declared: &Declaration) {
    match declared {
        Declaration::Param(param) => {
            sink.u8(0);
            write_param(sink, param);
        }
        Declaration::Shared(array) => {
            sink.u8(1);
            sink.text(&array.name);
            write_scalar(sink, array.elem);
            sink.i64(array.len);
        }
    }
}

fn read_declaration(reader: &mut Reader<'_>) -> Result<Declaration> {
    match reader.u8()? {
        0 => Ok(Declaration::Param(read_param(reader)?)),
        1 => Ok(Declaration::Shared(SharedArr {
            name: reader.text()?,
            elem: read_scalar(reader)?,
            len: reader.i64()?,
        })),
        tag => Err(WireError::UnknownTag {
            what: "declaration",
            tag: u64::from(tag),
        }),
    }
}

fn write_signature(sink: &mut Sink, signature: &Signature) {
    sink.count(signature.parameters.len());
    for param in &signature.parameters {
        write_param(sink, param);
    }
    sink.count(signature.results.len());
    for &result in &signature.results {
        write_valty(sink, result);
    }
    sink.count(signature.accesses.len());
    for access in &signature.accesses {
        sink.u8(u8::from(access.read));
        sink.u8(u8::from(access.written));
        sink.u8(access.lanes);
    }
    match signature.sync {
        None => sink.u8(0),
        Some(sync) => {
            sink.u8(1);
            write_sync(sink, sync);
        }
    }
}

fn read_signature(reader: &mut Reader<'_>) -> Result<Signature> {
    let parameters = read_pool(reader, read_param)?;
    let results = read_pool(reader, read_valty)?;
    let accesses = read_pool(reader, |reader| {
        Ok(Access {
            read: reader.u8()? != 0,
            written: reader.u8()? != 0,
            lanes: reader.u8()?,
        })
    })?;
    let sync = match reader.u8()? {
        0 => None,
        1 => Some(read_sync(reader)?),
        tag => {
            return Err(WireError::UnknownTag {
                what: "presence",
                tag: u64::from(tag),
            });
        }
    };
    Ok(Signature {
        parameters,
        results,
        accesses,
        sync,
    })
}

fn write_const_array(sink: &mut Sink, array: &ConstArray) {
    sink.text(&array.name);
    write_scalar(sink, array.scalar);
    sink.blob(&array.data);
}

fn read_const_array(reader: &mut Reader<'_>) -> Result<ConstArray> {
    Ok(ConstArray {
        name: reader.text()?,
        scalar: read_scalar(reader)?,
        data: reader.blob()?.into(),
    })
}

/// Which shared payloads a body uses, in first-use order over its operations.
fn pools(f: &Function) -> (Vec<Arc<ConstArray>>, Vec<Arc<DeviceFunction>>) {
    let mut arrays: Vec<Arc<ConstArray>> = Vec::new();
    let mut callees: Vec<Arc<DeviceFunction>> = Vec::new();
    for op in &f.ops {
        match &op.kind {
            OpKind::ConstLoad(array) if !arrays.iter().any(|held| Arc::ptr_eq(held, array)) => {
                arrays.push(array.clone());
            }
            OpKind::Call(callee) if !callees.iter().any(|held| Arc::ptr_eq(held, callee)) => {
                callees.push(callee.clone());
            }
            _ => {}
        }
    }
    (arrays, callees)
}

fn pooled<T>(pool: &[Arc<T>], held: &Arc<T>) -> u64 {
    pool.iter()
        .position(|entry| Arc::ptr_eq(entry, held))
        .expect("the pool was built from these operations") as u64
}

fn write_kind(
    sink: &mut Sink,
    kind: &OpKind,
    arrays: &[Arc<ConstArray>],
    callees: &[Arc<DeviceFunction>],
) {
    match kind {
        OpKind::IntConst { scalar, value } => {
            sink.u8(0);
            write_scalar(sink, *scalar);
            sink.i64(*value);
        }
        OpKind::FloatConst { scalar, value } => {
            sink.u8(1);
            write_scalar(sink, *scalar);
            sink.f64(*value);
        }
        OpKind::Sreg(sreg) => {
            sink.u8(2);
            write_sreg(sink, *sreg);
        }
        OpKind::Load { lanes, nc, evict } => {
            sink.u8(3);
            sink.u8(*lanes);
            sink.u8(u8::from(*nc));
            write_evict(sink, *evict);
        }
        OpKind::Offset => sink.u8(4),
        OpKind::Un(op) => {
            sink.u8(5);
            write_un(sink, *op);
        }
        OpKind::Bin(op) => {
            sink.u8(6);
            write_bin(sink, *op);
        }
        OpKind::Fma => sink.u8(7),
        OpKind::Cmp(op) => {
            sink.u8(8);
            write_cmp(sink, *op);
        }
        OpKind::Select => sink.u8(9),
        OpKind::Cast(scalar) => {
            sink.u8(10);
            write_scalar(sink, *scalar);
        }
        OpKind::Splat(lanes) => {
            sink.u8(11);
            sink.u8(*lanes);
        }
        OpKind::Extract(lane) => {
            sink.u8(12);
            sink.u8(*lane);
        }
        OpKind::Pack => sink.u8(13),
        OpKind::ShflDown => sink.u8(14),
        OpKind::WarpAny => sink.u8(15),
        OpKind::ConstLoad(array) => {
            sink.u8(16);
            sink.u64(pooled(arrays, array));
        }
        OpKind::Bind => sink.u8(17),
        OpKind::Write { mode, evict } => {
            sink.u8(18);
            write_mode(sink, *mode);
            write_evict(sink, *evict);
        }
        OpKind::If => sink.u8(19),
        OpKind::Loop { index, trip } => {
            sink.u8(20);
            write_scalar(sink, *index);
            write_trip(sink, *trip);
        }
        OpKind::Scope => sink.u8(21),
        OpKind::Call(callee) => {
            sink.u8(22);
            sink.u64(pooled(callees, callee));
        }
        OpKind::Break => sink.u8(23),
        OpKind::Ret => sink.u8(24),
        OpKind::Bar => sink.u8(25),
        OpKind::ActiveMask => sink.u8(26),
        OpKind::ShflFrom => sink.u8(27),
        OpKind::FetchAdd => sink.u8(28),
    }
}

fn read_kind(
    reader: &mut Reader<'_>,
    arrays: &[Arc<ConstArray>],
    callees: &[Arc<DeviceFunction>],
) -> Result<OpKind> {
    Ok(match reader.u8()? {
        0 => OpKind::IntConst {
            scalar: read_scalar(reader)?,
            value: reader.i64()?,
        },
        1 => OpKind::FloatConst {
            scalar: read_scalar(reader)?,
            value: reader.f64()?,
        },
        2 => OpKind::Sreg(read_sreg(reader)?),
        3 => OpKind::Load {
            lanes: reader.u8()?,
            nc: reader.u8()? != 0,
            evict: read_evict(reader)?,
        },
        4 => OpKind::Offset,
        5 => OpKind::Un(read_un(reader)?),
        6 => OpKind::Bin(read_bin(reader)?),
        7 => OpKind::Fma,
        8 => OpKind::Cmp(read_cmp(reader)?),
        9 => OpKind::Select,
        10 => OpKind::Cast(read_scalar(reader)?),
        11 => OpKind::Splat(reader.u8()?),
        12 => OpKind::Extract(reader.u8()?),
        13 => OpKind::Pack,
        14 => OpKind::ShflDown,
        15 => OpKind::WarpAny,
        16 => OpKind::ConstLoad(arrays[reader.index("constant array", arrays.len())?].clone()),
        17 => OpKind::Bind,
        18 => OpKind::Write {
            mode: read_mode(reader)?,
            evict: read_evict(reader)?,
        },
        19 => OpKind::If,
        20 => OpKind::Loop {
            index: read_scalar(reader)?,
            trip: read_trip(reader)?,
        },
        21 => OpKind::Scope,
        22 => OpKind::Call(callees[reader.index("callee", callees.len())?].clone()),
        23 => OpKind::Break,
        24 => OpKind::Ret,
        25 => OpKind::Bar,
        26 => OpKind::ActiveMask,
        27 => OpKind::ShflFrom,
        28 => OpKind::FetchAdd,
        tag => {
            return Err(WireError::UnknownTag {
                what: "operation",
                tag: u64::from(tag),
            });
        }
    })
}

/// The counts every id in the rest of the record is checked against.
struct Sizes {
    values: usize,
    ops: usize,
    blocks: usize,
    regions: usize,
}

pub fn encode_function(f: &Function) -> Vec<u8> {
    let mut sink = Sink(Vec::new());
    write_function(&mut sink, f);
    sink.0
}

pub fn decode_function(bytes: &[u8]) -> Result<Function> {
    let mut reader = Reader { bytes, at: 0 };
    let f = read_function(&mut reader)?;
    reader.end()?;
    Ok(f)
}

pub fn encode_device_function(f: &DeviceFunction) -> Vec<u8> {
    let mut sink = Sink(Vec::new());
    write_device_function(&mut sink, f);
    sink.0
}

pub fn decode_device_function(bytes: &[u8]) -> Result<DeviceFunction> {
    let mut reader = Reader { bytes, at: 0 };
    let f = read_device_function(&mut reader)?;
    reader.end()?;
    Ok(f)
}

fn write_device_function(sink: &mut Sink, f: &DeviceFunction) {
    sink.text(&f.name);
    match &f.implementation {
        Implementation::Body(body) => {
            sink.u8(0);
            write_function(sink, body);
        }
        Implementation::Declared(signature) => {
            sink.u8(1);
            write_signature(sink, signature);
        }
    }
}

fn read_device_function(reader: &mut Reader<'_>) -> Result<DeviceFunction> {
    let name = reader.text()?;
    let implementation = match reader.u8()? {
        0 => Implementation::Body(read_function(reader)?),
        1 => Implementation::Declared(read_signature(reader)?),
        tag => {
            return Err(WireError::UnknownTag {
                what: "implementation",
                tag: u64::from(tag),
            });
        }
    };
    Ok(DeviceFunction {
        name,
        implementation,
    })
}

/// One arena's bytes.
///
/// What is written is the *program*, so an arena an editor deleted from is
/// written from [`Function::compacted`]: a deleted operation is not part of the
/// program, and carrying its slot would make the bytes of two equal programs
/// differ by what one of them used to hold. Ids are dense in bytes for the same
/// reason they are dense in text — everything reading a whole program addresses
/// by index — and an arena nothing was deleted from comes through position for
/// position, which is why writing an unedited program down is unchanged.
fn write_function(sink: &mut Sink, f: &Function) {
    if f.has_deletions() {
        write_dense(sink, &f.compacted());
        return;
    }
    write_dense(sink, f);
}

fn write_dense(sink: &mut Sink, f: &Function) {
    sink.count(f.values.len());
    sink.count(f.ops.len());
    sink.count(f.blocks.len());
    sink.count(f.regions.len());

    let (arrays, callees) = pools(f);
    sink.count(arrays.len());
    for array in &arrays {
        write_const_array(sink, array);
    }
    sink.count(callees.len());
    for callee in &callees {
        write_device_function(sink, callee);
    }

    for value in &f.values {
        write_valty(sink, value.ty);
        match value.def {
            Def::Param { block, index } => {
                sink.u8(0);
                sink.u64(block.index() as u64);
                sink.u32(index);
            }
            Def::Result { op, index } => {
                sink.u8(1);
                sink.u64(op.index() as u64);
                sink.u32(index);
            }
        }
    }
    for op in &f.ops {
        write_kind(sink, &op.kind, &arrays, &callees);
        sink.u8(u8::from(op.placed));
        write_ids(sink, &op.operands, ValueId::index);
        write_ids(sink, &op.results, ValueId::index);
        sink.count(op.into.len());
        for destination in &op.into {
            match destination {
                None => sink.u8(0),
                Some(place) => {
                    sink.u8(1);
                    sink.u64(place.index() as u64);
                }
            }
        }
        write_ids(sink, &op.regions, RegionId::index);
    }
    for block in &f.blocks {
        write_ids(sink, &block.params, ValueId::index);
        write_ids(sink, &block.ops, OpId::index);
        write_ids(sink, &block.terminator.yields, ValueId::index);
        match block.region {
            None => sink.u8(0),
            Some(region) => {
                sink.u8(1);
                sink.u64(region.index() as u64);
            }
        }
    }
    for region in &f.regions {
        match region.owner {
            None => sink.u8(0),
            Some(op) => {
                sink.u8(1);
                sink.u64(op.index() as u64);
            }
        }
        write_ids(sink, &region.blocks, BlockId::index);
    }
    sink.u64(f.entry.index() as u64);
    sink.count(f.declarations.len());
    for declared in &f.declarations {
        write_declaration(sink, declared);
    }
    sink.count(f.names.len());
    for (&value, name) in &f.names {
        sink.u64(u64::from(value));
        sink.text(name);
    }
}

fn write_ids<T: Copy>(sink: &mut Sink, ids: &[T], index: impl Fn(T) -> usize) {
    sink.count(ids.len());
    for &id in ids {
        sink.u64(index(id) as u64);
    }
}

fn read_function(reader: &mut Reader<'_>) -> Result<Function> {
    let sizes = Sizes {
        values: reader.count()?,
        ops: reader.count()?,
        blocks: reader.count()?,
        regions: reader.count()?,
    };

    let arrays = read_pool(reader, |reader| Ok(Arc::new(read_const_array(reader)?)))?;
    let callees = read_pool(reader, |reader| Ok(Arc::new(read_device_function(reader)?)))?;

    let tag = ArenaTag::mint();
    let value = |index: usize| ValueId::new(tag, index);
    let op = |index: usize| OpId::new(tag, index);
    let block = |index: usize| BlockId::new(tag, index);
    let region = |index: usize| RegionId::new(tag, index);

    let mut values = Vec::with_capacity(sizes.values);
    for _ in 0..sizes.values {
        let ty = read_valty(reader)?;
        let def = match reader.u8()? {
            0 => Def::Param {
                block: block(reader.index("block", sizes.blocks)?),
                index: reader.u32()?,
            },
            1 => Def::Result {
                op: op(reader.index("operation", sizes.ops)?),
                index: reader.u32()?,
            },
            tag => {
                return Err(WireError::UnknownTag {
                    what: "definition",
                    tag: u64::from(tag),
                });
            }
        };
        values.push(ValueDef { ty, def });
    }

    let mut ops = Vec::with_capacity(sizes.ops);
    for _ in 0..sizes.ops {
        let kind = read_kind(reader, &arrays, &callees)?;
        let placed = reader.u8()? != 0;
        let operands = read_ids(reader, "value", sizes.values, value)?;
        let results = read_ids(reader, "value", sizes.values, value)?;
        let length = reader.count()?;
        let mut into = Vec::with_capacity(length);
        for _ in 0..length {
            into.push(match reader.u8()? {
                0 => None,
                1 => Some(value(reader.index("value", sizes.values)?)),
                tag => {
                    return Err(WireError::UnknownTag {
                        what: "destination",
                        tag: u64::from(tag),
                    });
                }
            });
        }
        let regions = read_ids(reader, "region", sizes.regions, region)?;
        ops.push(OpDef {
            kind,
            placed,
            operands,
            results,
            into,
            regions,
        });
    }

    let mut blocks = Vec::with_capacity(sizes.blocks);
    for _ in 0..sizes.blocks {
        blocks.push(BlockDef {
            params: read_ids(reader, "value", sizes.values, value)?,
            ops: read_ids(reader, "operation", sizes.ops, op)?,
            terminator: Terminator {
                yields: read_ids(reader, "value", sizes.values, value)?,
            },
            region: read_option(reader, "region", sizes.regions, region)?,
        });
    }

    let mut regions = Vec::with_capacity(sizes.regions);
    for _ in 0..sizes.regions {
        regions.push(RegionDef {
            owner: read_option(reader, "operation", sizes.ops, op)?,
            blocks: read_ids(reader, "block", sizes.blocks, block)?,
        });
    }

    let entry = block(reader.index("block", sizes.blocks)?);
    let declarations = (0..reader.count()?)
        .map(|_| read_declaration(reader))
        .collect::<Result<Vec<_>>>()?;
    let mut names = std::collections::BTreeMap::new();
    for _ in 0..reader.count()? {
        let value = reader.index("value", sizes.values)?;
        names.insert(value as u32, reader.text()?);
    }

    Ok(Function {
        tag,
        values,
        ops,
        blocks,
        regions,
        entry,
        declarations,
        names,
        deleted: super::Deleted::default(),
    })
}

fn read_pool<T>(
    reader: &mut Reader<'_>,
    read: impl Fn(&mut Reader<'_>) -> Result<T>,
) -> Result<Vec<T>> {
    let length = reader.count()?;
    (0..length).map(|_| read(reader)).collect()
}

fn read_ids<T>(
    reader: &mut Reader<'_>,
    what: &'static str,
    count: usize,
    id: impl Fn(usize) -> T,
) -> Result<Vec<T>> {
    Ok(reader
        .indices(what, count)?
        .into_iter()
        .map(id)
        .collect::<Vec<_>>())
}

fn read_option<T>(
    reader: &mut Reader<'_>,
    what: &'static str,
    count: usize,
    id: impl Fn(usize) -> T,
) -> Result<Option<T>> {
    match reader.u8()? {
        0 => Ok(None),
        1 => Ok(Some(id(reader.index(what, count)?))),
        tag => Err(WireError::UnknownTag {
            what: "presence",
            tag: u64::from(tag),
        }),
    }
}

#[cfg(test)]
#[path = "wire_tests.rs"]
mod tests;
