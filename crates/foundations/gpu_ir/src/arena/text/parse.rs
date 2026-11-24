// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text read back as a function.
//!
//! Reading happens in two moves, because a call names a function the reader has
//! not reached yet. [`parse_function`] decides only what the text *says* — every
//! id explicit, every index range-checked — and [`RawFunction::build`] turns
//! that into an arena once the caller can answer which function each symbol is.
//! A declaration names nothing outside itself, so it arrives complete and
//! [`RawDeviceFunction::build`] hands it back with no symbol to answer.
//!
//! What is decided here is decodability and nothing more. A body whose types
//! disagree, whose regions are owned by the wrong operation or whose values are
//! read where they are not defined comes through intact, and so does a
//! signature stating one access for two parameters, because those are things
//! [`crate::verify`] says and things an editor is allowed to pass through on
//! the way between two valid programs.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{
    Ahead, Reader, Result, binary_named, comparison_named, evict_named, scalar_named, space_named,
    sreg_named, sync_named, trip_named, unary_named, write_mode_named,
};
use crate::arena::{
    ArenaTag, BlockDef, BlockId, Declaration, Def, Function, OpDef, OpId, RegionDef, RegionId,
    Terminator, ValueDef, ValueId,
};
use crate::func::{Access, DeviceFunction, Signature};
use crate::op::{ConstArray, OpKind};
use crate::{Param, ParamTy, PtrTy, SharedArr, Ty, ValTy};

/// Where one thing the text states was written, so a refusal about it points at
/// it rather than at the body it is in.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct At {
    pub line: usize,
    pub column: usize,
}

/// One id the text names, and the token that named it.
///
/// Whether an id is in range is only decidable once the whole body has been
/// read, and by then the reader has moved on — so where it was read is carried
/// with it, and a refusal points at the token rather than at the line or the
/// body around it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ref {
    pub index: usize,
    pub at: At,
}

/// Where a value comes from, by index rather than by id.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RawDef {
    Param { block: Ref, index: u32 },
    Result { op: Ref, index: u32 },
}

#[derive(Clone, Debug)]
pub struct RawValue {
    pub ty: ValTy,
    pub def: RawDef,
    pub at: At,
}

/// An operation's kind, with the two that name something outside the arena left
/// as what the text says: a position in this body's constant pool, and a module
/// function symbol.
#[derive(Clone, Debug)]
pub enum RawKind {
    Direct(OpKind),
    ConstLoad(Ref),
    Call(Ref),
}

#[derive(Clone, Debug)]
pub struct RawOp {
    pub kind: RawKind,
    pub placed: bool,
    pub operands: Vec<Ref>,
    pub results: Vec<Ref>,
    pub into: Vec<Option<Ref>>,
    pub regions: Vec<Ref>,
    pub at: At,
}

#[derive(Clone, Debug, Default)]
pub struct RawBlock {
    pub region: Option<Ref>,
    pub params: Vec<Ref>,
    pub ops: Vec<Ref>,
    pub yields: Vec<Ref>,
    pub at: At,
}

#[derive(Clone, Debug)]
pub struct RawRegion {
    pub owner: Option<Ref>,
    pub blocks: Vec<Ref>,
    pub at: At,
}

/// One arena as the text states it, before anything outside it is resolved.
#[derive(Clone, Debug, Default)]
pub struct RawFunction {
    /// Where this body began, so a refusal that only becomes decidable once the
    /// whole body is read still points into the text.
    pub line: usize,
    pub column: usize,
    pub constants: Vec<ConstArray>,
    pub declarations: Vec<Declaration>,
    pub values: Vec<RawValue>,
    pub ops: Vec<RawOp>,
    pub blocks: Vec<RawBlock>,
    pub regions: Vec<RawRegion>,
    pub entry: Option<Ref>,
    pub names: Vec<(Ref, String)>,
}

impl Ref {
    /// This reference, once it is one: in range against what the body holds, or
    /// refused where it was written.
    fn against(self, what: &str, count: usize) -> Result<usize> {
        if self.index >= count {
            return Err(super::TextError {
                line: self.at.line,
                column: self.at.column,
                detail: format!("{what} {} is named where there are {count}", self.index),
            });
        }
        Ok(self.index)
    }
}

impl RawFunction {
    /// Which module function each call runs, by operation index.
    pub fn calls(&self) -> Vec<(usize, usize)> {
        self.ops
            .iter()
            .enumerate()
            .filter_map(|(index, op)| match op.kind {
                RawKind::Call(symbol) => Some((index, symbol.index)),
                _ => None,
            })
            .collect()
    }

    /// Why a body that reads cleanly is still not an arena. Only decidable once
    /// the whole body has been read, so it points at where that body began.
    fn refused<T>(&self, detail: impl Into<String>) -> Result<T> {
        Err(super::TextError {
            line: self.line,
            column: self.column,
            detail: detail.into(),
        })
    }

    /// The same, pointing at the one line that states the thing at fault.
    fn refused_at<T>(&self, at: At, detail: impl Into<String>) -> Result<T> {
        Err(super::TextError {
            line: at.line,
            column: at.column,
            detail: detail.into(),
        })
    }

    /// Who owns what, decided from the text alone.
    ///
    /// These are the facts an id *is*: one operation is a step of one block,
    /// one value is defined once at the ordinal it names, one block belongs to
    /// one region and one region to one operation, and both ends say the same
    /// thing. A text breaking one of them does not describe an arena at all —
    /// there is no program to repair — so it is refused here rather than
    /// carried to [`crate::verify`].
    ///
    /// What is *not* decided here is meaning. A body whose types disagree, whose
    /// values are read where they are not defined or whose effects are
    /// undeclared comes through intact: that is a program, and a wrong one is
    /// what an editor is holding on the way to a right one.
    fn ownership(&self) -> Result<()> {
        let mut home: BTreeMap<usize, usize> = BTreeMap::new();
        for (index, block) in self.blocks.iter().enumerate() {
            for op in &block.ops {
                if let Some(held) = home.insert(op.index, index) {
                    return self.refused_at(
                        op.at,
                        format!(
                            "operation *{} is a step of block ^{held} and of block ^{index}, \
                             so what it defines would be defined twice",
                            op.index
                        ),
                    );
                }
            }
        }
        for (index, op) in self.ops.iter().enumerate() {
            if !home.contains_key(&index) {
                return self.refused_at(op.at, format!("operation *{index} belongs to no block"));
            }
        }
        for (index, value) in self.values.iter().enumerate() {
            let held = match value.def {
                RawDef::Param { block, index } => self.blocks[block.index]
                    .params
                    .get(index as usize)
                    .map(|held| held.index),
                RawDef::Result { op, index } => self.ops[op.index]
                    .results
                    .get(index as usize)
                    .map(|held| held.index),
            };
            if held != Some(index) {
                let (what, at, ordinal) = match value.def {
                    RawDef::Param { block, index } => ("block ^", block, index),
                    RawDef::Result { op, index } => ("operation *", op, index),
                };
                return self.refused_at(
                    at.at,
                    format!("{what}{} does not define %{index} at {ordinal}", at.index),
                );
            }
        }
        let mut inside: BTreeMap<usize, usize> = BTreeMap::new();
        for (index, region) in self.regions.iter().enumerate() {
            for block in &region.blocks {
                if let Some(held) = inside.insert(block.index, index) {
                    return self.refused_at(
                        block.at,
                        format!(
                            "block ^{} is held by region !{held} and by region !{index}",
                            block.index
                        ),
                    );
                }
            }
        }
        for (index, block) in self.blocks.iter().enumerate() {
            if block.region.map(|held| held.index) != inside.get(&index).copied() {
                return self.refused_at(
                    block.at,
                    match block.region {
                        None => format!("block ^{index} belongs to no region and one holds it"),
                        Some(region) => format!(
                            "block ^{index} belongs to region !{} and that region does \
                             not hold it",
                            region.index
                        ),
                    },
                );
            }
        }
        let mut owned: BTreeMap<usize, usize> = BTreeMap::new();
        for (index, op) in self.ops.iter().enumerate() {
            for region in &op.regions {
                if let Some(held) = owned.insert(region.index, index) {
                    return self.refused_at(
                        region.at,
                        format!(
                            "region !{} is listed by operation *{held} and by operation \
                             *{index}",
                            region.index
                        ),
                    );
                }
            }
        }
        for (index, region) in self.regions.iter().enumerate() {
            if region.owner.map(|held| held.index) != owned.get(&index).copied() {
                return self.refused_at(
                    region.at,
                    match region.owner {
                        None => {
                            format!("region !{index} is owned by nothing and an operation lists it")
                        }
                        Some(op) => format!(
                            "region !{index} is owned by operation *{} and that operation does \
                             not list it",
                            op.index
                        ),
                    },
                );
            }
        }
        Ok(())
    }

    /// The arena this text states, with every symbol answered by `callee`.
    pub fn build(&self, callee: &dyn Fn(usize) -> Option<Arc<DeviceFunction>>) -> Result<Function> {
        let pool: Vec<Arc<ConstArray>> = self
            .constants
            .iter()
            .map(|array| Arc::new(array.clone()))
            .collect();
        let tag = ArenaTag::mint();
        let value = |index: usize| ValueId::new(tag, index);
        let op = |index: usize| OpId::new(tag, index);
        let block = |index: usize| BlockId::new(tag, index);
        let region = |index: usize| RegionId::new(tag, index);
        let values_held = self.values.len();
        let ops_held = self.ops.len();
        let blocks_held = self.blocks.len();
        let regions_held = self.regions.len();
        let Some(entry) = self.entry else {
            return self.refused("a body states which block it enters");
        };
        let entry = entry.against("block", blocks_held)?;

        let mut values = Vec::with_capacity(values_held);
        for held in &self.values {
            values.push(ValueDef {
                ty: held.ty,
                def: match held.def {
                    RawDef::Param { block: at, index } => Def::Param {
                        block: block(at.against("block", blocks_held)?),
                        index,
                    },
                    RawDef::Result { op: at, index } => Def::Result {
                        op: op(at.against("operation", ops_held)?),
                        index,
                    },
                },
            });
        }

        let mut ops = Vec::with_capacity(ops_held);
        for held_op in &self.ops {
            let kind = match &held_op.kind {
                RawKind::Direct(kind) => kind.clone(),
                RawKind::ConstLoad(held) => {
                    OpKind::ConstLoad(pool[held.against("constant", pool.len())?].clone())
                }
                RawKind::Call(symbol) => match callee(symbol.index) {
                    Some(callee) => OpKind::Call(callee),
                    None => {
                        return self.refused_at(
                            symbol.at,
                            format!("@{} is no function of this module", symbol.index),
                        );
                    }
                },
            };
            let mut operands = Vec::with_capacity(held_op.operands.len());
            for held in &held_op.operands {
                operands.push(value(held.against("value", values_held)?));
            }
            let mut results = Vec::with_capacity(held_op.results.len());
            for held in &held_op.results {
                results.push(value(held.against("value", values_held)?));
            }
            let mut into = Vec::with_capacity(held_op.into.len());
            for destination in &held_op.into {
                into.push(match destination {
                    None => None,
                    Some(held) => Some(value(held.against("value", values_held)?)),
                });
            }
            let mut regions = Vec::with_capacity(held_op.regions.len());
            for held in &held_op.regions {
                regions.push(region(held.against("region", regions_held)?));
            }
            ops.push(OpDef {
                kind,
                placed: held_op.placed,
                operands,
                results,
                into,
                regions,
            });
        }

        let mut blocks = Vec::with_capacity(blocks_held);
        for held_block in &self.blocks {
            let mut params = Vec::with_capacity(held_block.params.len());
            for held in &held_block.params {
                params.push(value(held.against("value", values_held)?));
            }
            let mut steps = Vec::with_capacity(held_block.ops.len());
            for held in &held_block.ops {
                steps.push(op(held.against("operation", ops_held)?));
            }
            let mut yields = Vec::with_capacity(held_block.yields.len());
            for held in &held_block.yields {
                yields.push(value(held.against("value", values_held)?));
            }
            blocks.push(BlockDef {
                params,
                ops: steps,
                terminator: Terminator { yields },
                region: match held_block.region {
                    None => None,
                    Some(held) => Some(region(held.against("region", regions_held)?)),
                },
            });
        }

        let mut regions = Vec::with_capacity(regions_held);
        for held_region in &self.regions {
            let mut held_blocks = Vec::with_capacity(held_region.blocks.len());
            for held in &held_region.blocks {
                held_blocks.push(block(held.against("block", blocks_held)?));
            }
            regions.push(RegionDef {
                owner: match held_region.owner {
                    None => None,
                    Some(held) => Some(op(held.against("operation", ops_held)?)),
                },
                blocks: held_blocks,
            });
        }

        // One value is named once. A second record for it would make which
        // name it carries depend on which was read last, so it is refused
        // rather than resolved in favour of either.
        let mut names = std::collections::BTreeMap::new();
        for (held, name) in &self.names {
            let index = held.against("value", values_held)?;
            if names.insert(index as u32, name.clone()).is_some() {
                return self.refused_at(held.at, format!("%{index} is named twice"));
            }
        }

        // Every id is in range now, so who owns what is decidable — and it is
        // decided before an arena exists rather than left to be met later.
        self.ownership()?;

        Ok(Function {
            tag,
            values,
            ops,
            blocks,
            regions,
            entry: block(entry),
            declarations: self.declarations.clone(),
            names,
            deleted: crate::arena::Deleted::default(),
        })
    }
}

/// Where one device function's implementation is, as the text states it.
///
/// A body is what the text says and no more, because a call in it names a
/// symbol only its module can answer. A declaration names nothing outside
/// itself, so it is already the contract it will be.
#[derive(Clone, Debug)]
pub enum RawImplementation {
    Body(RawFunction),
    Declared(Signature),
}

/// One device function as the text states it: its name, and where its
/// implementation is.
#[derive(Clone, Debug)]
pub struct RawDeviceFunction {
    pub name: String,
    pub implementation: RawImplementation,
}

impl RawDeviceFunction {
    /// The function this text states, with every symbol its body calls answered
    /// by `callee`. A declaration calls nothing and asks nothing of it.
    pub fn build(
        &self,
        callee: &dyn Fn(usize) -> Option<Arc<DeviceFunction>>,
    ) -> Result<DeviceFunction> {
        Ok(match &self.implementation {
            RawImplementation::Body(raw) => {
                DeviceFunction::new(self.name.clone(), raw.build(callee)?)
            }
            RawImplementation::Declared(signature) => {
                DeviceFunction::declared(self.name.clone(), signature.clone())
            }
        })
    }
}

/// One device function's worth of text, from where the reader stands until the
/// first word that is not one of this grammar's.
pub fn parse_device_function(reader: &mut Reader<'_>) -> Result<RawDeviceFunction> {
    reader.keyword("function")?;
    let name = reader.quoted()?;
    let implementation = if reader.taking("declared") {
        RawImplementation::Declared(parse_signature(reader)?)
    } else {
        reader.keyword("body")?;
        RawImplementation::Body(parse_function(reader)?)
    };
    Ok(RawDeviceFunction {
        name,
        implementation,
    })
}

/// The contract a function with no body here states, read as the three lists
/// and the one fact it is. Nothing is paired on the way in: the *n*-th access
/// is what the *n*-th parameter is touched by, and a text stating a different
/// number of each says so.
pub fn parse_signature(reader: &mut Reader<'_>) -> Result<Signature> {
    let mut held = Signature {
        parameters: Vec::new(),
        results: Vec::new(),
        accesses: Vec::new(),
        sync: None,
    };
    let mut synchronised = false;
    loop {
        let Ahead::Word(word) = reader.ahead() else {
            return Ok(held);
        };
        match word {
            "param" => {
                reader.keyword("param")?;
                held.parameters.push(parameter(reader)?);
            }
            "access" => {
                reader.keyword("access")?;
                held.accesses.push(access(reader)?);
            }
            "result" => {
                reader.keyword("result")?;
                held.results.push(valty(reader)?);
            }
            "sync" => {
                reader.keyword("sync")?;
                if synchronised {
                    return reader
                        .fail("a function needs one synchronisation and this states a second");
                }
                synchronised = true;
                held.sync = synchronisation(reader)?;
            }
            _ => return Ok(held),
        }
    }
}

/// What a function does to the memory behind one parameter: which way it is
/// touched, and the widest access through it.
fn access(reader: &mut Reader<'_>) -> Result<Access> {
    let word = reader.word()?;
    let (read, written) = match word {
        "none" => (false, false),
        "read" => (true, false),
        "write" => (false, true),
        "read-write" => (true, true),
        _ => return reader.fail(format!("{word:?} is no way of touching memory")),
    };
    reader.keyword("lanes")?;
    Ok(Access {
        read,
        written,
        lanes: reader.number("lane count")?,
    })
}

/// The threads a function needs to arrive together, and nothing where it needs
/// none.
fn synchronisation(reader: &mut Reader<'_>) -> Result<Option<crate::effect::Sync>> {
    let word = reader.word()?;
    if word == "none" {
        return Ok(None);
    }
    match sync_named(word) {
        Some(sync) => Ok(Some(sync)),
        None => reader.fail(format!("{word:?} is no synchronisation")),
    }
}

/// One arena's worth of text, from where the reader stands until the first word
/// that is not one of this grammar's.
pub fn parse_function(reader: &mut Reader<'_>) -> Result<RawFunction> {
    let (line, column) = reader.here();
    let mut raw = RawFunction {
        line,
        column,
        ..RawFunction::default()
    };
    loop {
        let Ahead::Word(word) = reader.ahead() else {
            return Ok(raw);
        };
        match word {
            "constant" => {
                reader.keyword("constant")?;
                let index = reader.tagged('$', "constant")?;
                if index != raw.constants.len() {
                    return reader.fail(format!(
                        "constants are stated in order and ${} belongs here",
                        raw.constants.len()
                    ));
                }
                let name = reader.quoted()?;
                let scalar = scalar(reader)?;
                let data = reader.blob()?;
                raw.constants.push(ConstArray {
                    name,
                    scalar,
                    data: data.into(),
                });
            }
            "param" => {
                reader.keyword("param")?;
                raw.declarations
                    .push(Declaration::Param(parameter(reader)?));
            }
            "shared" => {
                reader.keyword("shared")?;
                let name = reader.quoted()?;
                let elem = scalar(reader)?;
                let len = reader.number("length")?;
                raw.declarations
                    .push(Declaration::Shared(SharedArr { name, elem, len }));
            }
            "value" => {
                let at = where_it_says(reader);
                reader.keyword("value")?;
                let index = reader.tagged('%', "value")?;
                if index != raw.values.len() {
                    return reader.fail(format!(
                        "values are stated in order and %{} belongs here",
                        raw.values.len()
                    ));
                }
                let ty = valty(reader)?;
                let def = if reader.taking("param") {
                    RawDef::Param {
                        block: reference(reader, '^', "block")?,
                        index: reader.number("parameter position")?,
                    }
                } else {
                    reader.keyword("result")?;
                    RawDef::Result {
                        op: reference(reader, '*', "operation")?,
                        index: reader.number("result position")?,
                    }
                };
                raw.values.push(RawValue { ty, def, at });
            }
            "op" => {
                let at = where_it_says(reader);
                reader.keyword("op")?;
                let index = reader.tagged('*', "operation")?;
                if index != raw.ops.len() {
                    return reader.fail(format!(
                        "operations are stated in order and *{} belongs here",
                        raw.ops.len()
                    ));
                }
                let placed = if reader.taking("step") {
                    true
                } else {
                    reader.keyword("operand")?;
                    false
                };
                let kind = kind(reader)?;
                reader.keyword("operands")?;
                let operands = reader.list(|reader| reference(reader, '%', "value"))?;
                reader.keyword("results")?;
                let results = reader.list(|reader| reference(reader, '%', "value"))?;
                reader.keyword("into")?;
                let into = reader.list(|reader| {
                    if reader.taking("none") {
                        return Ok(None);
                    }
                    Ok(Some(reference(reader, '%', "value")?))
                })?;
                reader.keyword("regions")?;
                let regions = reader.list(|reader| reference(reader, '!', "region"))?;
                raw.ops.push(RawOp {
                    kind,
                    placed,
                    operands,
                    results,
                    into,
                    regions,
                    at,
                });
            }
            "block" => {
                let at = where_it_says(reader);
                reader.keyword("block")?;
                let index = reader.tagged('^', "block")?;
                if index != raw.blocks.len() {
                    return reader.fail(format!(
                        "blocks are stated in order and ^{} belongs here",
                        raw.blocks.len()
                    ));
                }
                reader.keyword("region")?;
                let region = if reader.taking("none") {
                    None
                } else {
                    Some(reference(reader, '!', "region")?)
                };
                reader.keyword("params")?;
                let params = reader.list(|reader| reference(reader, '%', "value"))?;
                reader.keyword("ops")?;
                let ops = reader.list(|reader| reference(reader, '*', "operation"))?;
                reader.keyword("yields")?;
                let yields = reader.list(|reader| reference(reader, '%', "value"))?;
                raw.blocks.push(RawBlock {
                    region,
                    params,
                    ops,
                    yields,
                    at,
                });
            }
            "region" => {
                let at = where_it_says(reader);
                reader.keyword("region")?;
                let index = reader.tagged('!', "region")?;
                if index != raw.regions.len() {
                    return reader.fail(format!(
                        "regions are stated in order and !{} belongs here",
                        raw.regions.len()
                    ));
                }
                reader.keyword("owner")?;
                let owner = if reader.taking("none") {
                    None
                } else {
                    Some(reference(reader, '*', "operation")?)
                };
                reader.keyword("blocks")?;
                let blocks = reader.list(|reader| reference(reader, '^', "block"))?;
                raw.regions.push(RawRegion { owner, blocks, at });
            }
            "entry" => {
                reader.keyword("entry")?;
                if raw.entry.is_some() {
                    return reader.fail("a body enters one block and this states a second");
                }
                raw.entry = Some(reference(reader, '^', "block")?);
            }
            "name" => {
                reader.keyword("name")?;
                let value = reference(reader, '%', "value")?;
                raw.names.push((value, reader.quoted()?));
            }
            _ => return Ok(raw),
        }
    }
}

/// Where the line about to be read begins.
fn where_it_says(reader: &mut Reader<'_>) -> At {
    let (line, column) = reader.here();
    At { line, column }
}

/// One id, and where it was written.
fn reference(reader: &mut Reader<'_>, sigil: char, what: &str) -> Result<Ref> {
    let index = reader.tagged(sigil, what)?;
    let (line, column) = reader.taken();
    Ok(Ref {
        index,
        at: At { line, column },
    })
}

/// What one parameter is, after the word saying it is one: a name and a type.
/// One spelling serves an arena's declaration and a signature's, because they
/// are the same thing.
fn parameter(reader: &mut Reader<'_>) -> Result<Param> {
    let name = reader.quoted()?;
    let ty = if reader.taking("ptr") {
        ParamTy::Ptr(scalar(reader)?)
    } else {
        reader.keyword("scalar")?;
        ParamTy::Scalar(scalar(reader)?)
    };
    Ok(Param { name, ty })
}

fn scalar(reader: &mut Reader<'_>) -> Result<crate::Scalar> {
    let word = reader.word()?;
    match scalar_named(word) {
        Some(scalar) => Ok(scalar),
        None => reader.fail(format!("{word:?} is no scalar this language has")),
    }
}

fn valty(reader: &mut Reader<'_>) -> Result<ValTy> {
    if reader.taking("ptr") {
        let elem = scalar(reader)?;
        let word = reader.word()?;
        let Some(space) = space_named(word) else {
            return reader.fail(format!("{word:?} is no memory space"));
        };
        return Ok(ValTy::Ptr(PtrTy { elem, space }));
    }
    reader.keyword("data")?;
    let held = scalar(reader)?;
    let word = reader.word()?;
    let Some(lanes) = word.strip_prefix('x') else {
        return reader.fail(format!(
            "a lane count is written x<number> and this says {word:?}"
        ));
    };
    let Ok(lanes) = lanes.parse::<u8>() else {
        return reader.fail(format!("{lanes:?} is no lane count"));
    };
    Ok(ValTy::Data(Ty {
        scalar: held,
        lanes,
    }))
}

fn kind(reader: &mut Reader<'_>) -> Result<RawKind> {
    let word = reader.word()?;
    let direct = |kind| Ok(RawKind::Direct(kind));
    match word {
        "intconst" => direct(OpKind::IntConst {
            scalar: scalar(reader)?,
            value: reader.number("whole number")?,
        }),
        "floatconst" => {
            let scalar = scalar(reader)?;
            direct(OpKind::FloatConst {
                scalar,
                value: f64::from_bits(reader.bits(16)?),
            })
        }
        "sreg" => {
            let held = reader.word()?;
            match sreg_named(held) {
                Some(sreg) => direct(OpKind::Sreg(sreg)),
                None => reader.fail(format!("{held:?} is no special register")),
            }
        }
        "load" => {
            reader.keyword("lanes")?;
            let lanes = reader.number("lane count")?;
            reader.keyword("nc")?;
            let nc = boolean(reader)?;
            reader.keyword("evict")?;
            direct(OpKind::Load {
                lanes,
                nc,
                evict: evict(reader)?,
            })
        }
        "offset" => direct(OpKind::Offset),
        "un" => {
            let held = reader.word()?;
            match unary_named(held) {
                Some(op) => direct(OpKind::Un(op)),
                None => reader.fail(format!("{held:?} is no unary operator")),
            }
        }
        "bin" => {
            let held = reader.word()?;
            match binary_named(held) {
                Some(op) => direct(OpKind::Bin(op)),
                None => reader.fail(format!("{held:?} is no binary operator")),
            }
        }
        "fma" => direct(OpKind::Fma),
        "cmp" => {
            let held = reader.word()?;
            match comparison_named(held) {
                Some(op) => direct(OpKind::Cmp(op)),
                None => reader.fail(format!("{held:?} is no comparison")),
            }
        }
        "select" => direct(OpKind::Select),
        "cast" => direct(OpKind::Cast(scalar(reader)?)),
        "splat" => direct(OpKind::Splat(reader.number("lane count")?)),
        "extract" => direct(OpKind::Extract(reader.number("lane")?)),
        "pack" => direct(OpKind::Pack),
        "shfl-down" => direct(OpKind::ShflDown),
        "active-mask" => direct(OpKind::ActiveMask),
        "shfl-from" => direct(OpKind::ShflFrom),
        "warp-any" => direct(OpKind::WarpAny),
        "const-load" => Ok(RawKind::ConstLoad(reference(reader, '$', "constant")?)),
        "bind" => direct(OpKind::Bind),
        "write" => {
            let held = reader.word()?;
            let Some(mode) = write_mode_named(held) else {
                return reader.fail(format!("{held:?} is no write mode"));
            };
            reader.keyword("evict")?;
            direct(OpKind::Write {
                mode,
                evict: evict(reader)?,
            })
        }
        "fetch-add" => direct(OpKind::FetchAdd),
        "if" => direct(OpKind::If),
        "loop" => {
            let index = scalar(reader)?;
            let held = reader.word()?;
            match trip_named(held) {
                Some(trip) => direct(OpKind::Loop { index, trip }),
                None => reader.fail(format!("{held:?} is no trip count")),
            }
        }
        "scope" => direct(OpKind::Scope),
        "call" => Ok(RawKind::Call(reference(reader, '@', "function")?)),
        "break" => direct(OpKind::Break),
        "ret" => direct(OpKind::Ret),
        "bar" => direct(OpKind::Bar),
        _ => reader.fail(format!("{word:?} is no operation this language has")),
    }
}

fn boolean(reader: &mut Reader<'_>) -> Result<bool> {
    let word = reader.word()?;
    match word {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => reader.fail(format!("{word:?} is neither true nor false")),
    }
}

fn evict(reader: &mut Reader<'_>) -> Result<crate::Evict> {
    let word = reader.word()?;
    match evict_named(word) {
        Some(evict) => Ok(evict),
        None => reader.fail(format!("{word:?} is no eviction priority")),
    }
}
