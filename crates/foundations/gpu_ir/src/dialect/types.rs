// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use weave::{AttributeData, TypeData};

use crate::ParamTy;
use crate::arena::Declaration;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DataType(pub crate::Ty);

impl TypeData for DataType {
    const NAME: &'static str = "!gpu.data";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = payload(syntax, Self::NAME)?;
        let (scalar, lanes) = payload
            .split_once('x')
            .ok_or_else(|| "expected !gpu.data<scalarxlanes>".to_owned())?;
        let scalar = crate::arena::text::scalar_named(scalar)
            .ok_or_else(|| format!("unknown GPU scalar `{scalar}`"))?;
        let lanes = lanes.parse::<u8>().map_err(|error| error.to_string())?;
        if lanes == 0 {
            return Err("a GPU data type needs at least one lane".into());
        }
        Ok(Self(crate::Ty::vec(scalar, lanes)))
    }

    fn print(&self) -> String {
        format!(
            "{}<{}x{}>",
            Self::NAME,
            crate::arena::text::scalar_name(self.0.scalar),
            self.0.lanes
        )
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointerType(pub crate::PtrTy);

impl TypeData for PointerType {
    const NAME: &'static str = "!gpu.ptr";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = payload(syntax, Self::NAME)?;
        let (space, elem) = payload
            .split_once(',')
            .ok_or_else(|| "expected !gpu.ptr<space,scalar>".to_owned())?;
        let space = crate::arena::text::space_named(space)
            .ok_or_else(|| format!("unknown GPU address space `{space}`"))?;
        let elem = crate::arena::text::scalar_named(elem)
            .ok_or_else(|| format!("unknown GPU scalar `{elem}`"))?;
        Ok(Self(crate::PtrTy { elem, space }))
    }

    fn print(&self) -> String {
        format!(
            "{}<{},{}>",
            Self::NAME,
            crate::arena::text::space_name(self.0.space),
            crate::arena::text::scalar_name(self.0.elem)
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuAttribute {
    Bool(bool),
    UInt(usize),
    SInt(i64),
    F64Bits(u64),
    Scalar(crate::Scalar),
    Symbol(String),
    Text(String),
    Bytes(Vec<u8>),
    Indices(Vec<Option<usize>>),
    Declaration(Declaration),
    Constant(usize),
}

impl GpuAttribute {
    pub(super) fn from_payload(payload: impl Into<String>) -> Self {
        let syntax = format!("#gpu.attr<{}>", payload.into());
        Self::parse(&syntax).expect("converter emits valid GPU attributes")
    }

    pub fn payload(&self) -> String {
        match self {
            Self::Bool(value) => format!("bool({value})"),
            Self::UInt(value) => format!("uint({value})"),
            Self::SInt(value) => format!("sint({value})"),
            Self::F64Bits(value) => format!("f64_bits({value:016x})"),
            Self::Scalar(value) => {
                format!("scalar({})", crate::arena::text::scalar_name(*value))
            }
            Self::Symbol(value) => format!("enum({value})"),
            Self::Text(value) => format!("text({})", crate::dialect::hex(value.as_bytes())),
            Self::Bytes(value) => format!("bytes({})", crate::dialect::hex(value)),
            Self::Indices(values) => format!(
                "indices({})",
                values
                    .iter()
                    .map(|value| value.map_or_else(|| "n".to_owned(), |value| value.to_string()))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            Self::Declaration(Declaration::Param(param)) => match param.ty {
                ParamTy::Ptr(elem) => format!(
                    "param({},ptr,{})",
                    crate::dialect::hex(param.name.as_bytes()),
                    crate::arena::text::scalar_name(elem)
                ),
                ParamTy::Scalar(scalar) => format!(
                    "param({},scalar,{})",
                    crate::dialect::hex(param.name.as_bytes()),
                    crate::arena::text::scalar_name(scalar)
                ),
            },
            Self::Declaration(Declaration::Shared(array)) => format!(
                "shared({},{},{})",
                crate::dialect::hex(array.name.as_bytes()),
                crate::arena::text::scalar_name(array.elem),
                array.len
            ),
            Self::Constant(value) => format!("constant({value})"),
        }
    }
}

impl AttributeData for GpuAttribute {
    const NAME: &'static str = "#gpu.attr";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = payload(syntax, Self::NAME)?;
        let (kind, body) = payload
            .split_once('(')
            .and_then(|(kind, body)| body.strip_suffix(')').map(|body| (kind, body)))
            .ok_or_else(|| "expected #gpu.attr<kind(payload)>".to_owned())?;
        match kind {
            "bool" => match body {
                "true" => Ok(Self::Bool(true)),
                "false" => Ok(Self::Bool(false)),
                _ => Err("expected bool(true) or bool(false)".into()),
            },
            "uint" => body
                .parse()
                .map(Self::UInt)
                .map_err(|error: std::num::ParseIntError| error.to_string()),
            "sint" => body
                .parse()
                .map(Self::SInt)
                .map_err(|error: std::num::ParseIntError| error.to_string()),
            "f64_bits" if body.len() == 16 => u64::from_str_radix(body, 16)
                .map(Self::F64Bits)
                .map_err(|error| error.to_string()),
            "scalar" => crate::arena::text::scalar_named(body)
                .map(Self::Scalar)
                .ok_or_else(|| format!("unknown GPU scalar `{body}`")),
            "enum" if !body.is_empty() => Ok(Self::Symbol(body.to_owned())),
            "text" => Ok(Self::Text(
                String::from_utf8(unhex(body)?).map_err(|_| "GPU text is not UTF-8")?,
            )),
            "bytes" => Ok(Self::Bytes(unhex(body)?)),
            "indices" => {
                let values = if body.is_empty() {
                    Vec::new()
                } else {
                    body.split(',')
                        .map(|value| {
                            if value == "n" {
                                Ok(None)
                            } else {
                                value
                                    .parse()
                                    .map(Some)
                                    .map_err(|error: std::num::ParseIntError| error.to_string())
                            }
                        })
                        .collect::<Result<_, _>>()?
                };
                Ok(Self::Indices(values))
            }
            "param" => {
                let parts = body.split(',').collect::<Vec<_>>();
                if parts.len() != 3 {
                    return Err("expected param(hex,kind,scalar)".into());
                }
                let name = String::from_utf8(unhex(parts[0])?)
                    .map_err(|_| "GPU parameter name is not UTF-8")?;
                let scalar = crate::arena::text::scalar_named(parts[2])
                    .ok_or_else(|| format!("unknown GPU scalar `{}`", parts[2]))?;
                let ty = match parts[1] {
                    "ptr" => ParamTy::Ptr(scalar),
                    "scalar" => ParamTy::Scalar(scalar),
                    _ => return Err("expected ptr or scalar parameter kind".into()),
                };
                Ok(Self::Declaration(Declaration::Param(crate::Param {
                    name,
                    ty,
                })))
            }
            "constant" => body
                .parse()
                .map(Self::Constant)
                .map_err(|error: std::num::ParseIntError| error.to_string()),
            "shared" => {
                let parts = body.split(',').collect::<Vec<_>>();
                if parts.len() != 3 {
                    return Err("expected shared(hex,scalar,len)".into());
                }
                let name = String::from_utf8(unhex(parts[0])?)
                    .map_err(|_| "GPU shared name is not UTF-8")?;
                let elem = crate::arena::text::scalar_named(parts[1])
                    .ok_or_else(|| format!("unknown GPU scalar `{}`", parts[1]))?;
                let len = parts[2]
                    .parse()
                    .map_err(|error: std::num::ParseIntError| error.to_string())?;
                Ok(Self::Declaration(Declaration::Shared(crate::SharedArr {
                    name,
                    elem,
                    len,
                })))
            }
            _ => Err(format!("unknown GPU attribute family `{kind}`")),
        }
    }

    fn print(&self) -> String {
        format!("{}<{}>", Self::NAME, self.payload())
    }
}

fn unhex(value: &str) -> Result<Vec<u8>, String> {
    if !value.len().is_multiple_of(2) {
        return Err("hex payload has odd length".into());
    }
    value
        .as_bytes()
        .chunks(2)
        .map(|pair| {
            std::str::from_utf8(pair)
                .map_err(|error| error.to_string())
                .and_then(|pair| u8::from_str_radix(pair, 16).map_err(|error| error.to_string()))
        })
        .collect()
}

fn payload<'a>(syntax: &'a str, name: &str) -> Result<&'a str, String> {
    syntax
        .strip_prefix(name)
        .and_then(|rest| rest.strip_prefix('<'))
        .and_then(|rest| rest.strip_suffix('>'))
        .ok_or_else(|| format!("expected {name}<...>"))
}
