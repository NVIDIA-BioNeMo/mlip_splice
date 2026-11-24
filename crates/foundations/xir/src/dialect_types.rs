// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed XIR types and attributes.

use domain_index::{FiniteMapGraph, FiniteSetId};
use weave::{AttributeData, TypeData};

use crate::dtype::DType;
use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TensorType(pub Shape);

impl TypeData for TensorType {
    const NAME: &'static str = "!xir.tensor";

    fn parse(syntax: &str) -> Result<Self, String> {
        let body = syntax
            .strip_prefix("!xir.tensor<")
            .and_then(|value| value.strip_suffix('>'))
            .ok_or_else(|| "expected !xir.tensor<dtype;domains;dims>".to_owned())?;
        let mut fields = body.split(';');
        let dtype = parse_dtype(fields.next().unwrap_or_default())?;
        let domain_axes = parse_list(fields.next().ok_or("missing tensor domain axes")?)?
            .into_iter()
            .map(|value| {
                u32::try_from(value)
                    .map(FiniteSetId)
                    .map_err(|_| "invalid domain axis")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let dims = parse_list(fields.next().ok_or("missing tensor dimensions")?)?;
        if fields.next().is_some() {
            return Err("too many tensor type fields".into());
        }
        let parsed = Self(Shape::new(dtype, dims).with_domain_axes(domain_axes));
        if parsed.print() != syntax {
            return Err("tensor type must use canonical spelling".into());
        }
        Ok(parsed)
    }

    fn print(&self) -> String {
        format!(
            "!xir.tensor<{};{};{}>",
            self.0.dtype.name(),
            join(self.0.domain_axes.iter().map(|axis| axis.index())),
            join(self.0.dims.iter().copied())
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StringAttribute(pub String);

impl AttributeData for StringAttribute {
    const NAME: &'static str = "#xir.string";

    fn parse(syntax: &str) -> Result<Self, String> {
        let value = syntax
            .strip_prefix("#xir.string<")
            .and_then(|value| value.strip_suffix('>'))
            .ok_or_else(|| "expected #xir.string<hex>".to_owned())?;
        let parsed =
            Self(String::from_utf8(decode_hex(value)?).map_err(|error| error.to_string())?);
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        format!("#xir.string<{}>", encode_hex(self.0.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerAttribute(pub i64);

impl AttributeData for IntegerAttribute {
    const NAME: &'static str = "#xir.integer";

    fn parse(syntax: &str) -> Result<Self, String> {
        let parsed = parse_payload(syntax, Self::NAME)?
            .parse()
            .map(Self)
            .map_err(|error: std::num::ParseIntError| error.to_string())?;
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        format!("#xir.integer<{}>", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoolAttribute(pub bool);

impl AttributeData for BoolAttribute {
    const NAME: &'static str = "#xir.bool";

    fn parse(syntax: &str) -> Result<Self, String> {
        let parsed = match parse_payload(syntax, Self::NAME)? {
            "true" => Self(true),
            "false" => Self(false),
            _ => return Err("expected true or false".into()),
        };
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        format!("#xir.bool<{}>", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BytesAttribute(pub Vec<u8>);

impl AttributeData for BytesAttribute {
    const NAME: &'static str = "#xir.bytes";

    fn parse(syntax: &str) -> Result<Self, String> {
        let parsed = Self(decode_hex(parse_payload(syntax, Self::NAME)?)?);
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        format!("#xir.bytes<{}>", encode_hex(&self.0))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegersAttribute(pub Vec<i64>);

impl AttributeData for IntegersAttribute {
    const NAME: &'static str = "#xir.integers";

    fn parse(syntax: &str) -> Result<Self, String> {
        let parsed = Self(parse_list(parse_payload(syntax, Self::NAME)?)?);
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        format!("#xir.integers<{}>", join(self.0.iter().copied()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct EnumAttribute(pub String);

impl AttributeData for EnumAttribute {
    const NAME: &'static str = "#xir.enum";

    fn parse(syntax: &str) -> Result<Self, String> {
        let value = StringAttribute::parse(&syntax.replacen(Self::NAME, StringAttribute::NAME, 1))?;
        require_canonical(Self(value.0), syntax)
    }

    fn print(&self) -> String {
        format!("#xir.enum<{}>", encode_hex(self.0.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BoolsAttribute(pub Vec<bool>);

impl AttributeData for BoolsAttribute {
    const NAME: &'static str = "#xir.bools";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = parse_payload(syntax, Self::NAME)?;
        let values = if payload.is_empty() {
            Vec::new()
        } else {
            payload
                .split(',')
                .map(|value| value.parse().map_err(|_| "invalid boolean".to_owned()))
                .collect::<Result<_, _>>()?
        };
        require_canonical(Self(values), syntax)
    }

    fn print(&self) -> String {
        format!("#xir.bools<{}>", join(self.0.iter().copied()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IntegerPairsAttribute(pub Vec<[i64; 2]>);

impl AttributeData for IntegerPairsAttribute {
    const NAME: &'static str = "#xir.integer_pairs";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = parse_payload(syntax, Self::NAME)?;
        let values = if payload.is_empty() {
            Vec::new()
        } else {
            payload
                .split(',')
                .map(|pair| {
                    let (left, right) = pair.split_once(':').ok_or("expected integer pair")?;
                    Ok([
                        left.parse().map_err(|_| "invalid integer")?,
                        right.parse().map_err(|_| "invalid integer")?,
                    ])
                })
                .collect::<Result<_, String>>()?
        };
        require_canonical(Self(values), syntax)
    }

    fn print(&self) -> String {
        format!(
            "#xir.integer_pairs<{}>",
            self.0
                .iter()
                .map(|[left, right]| format!("{left}:{right}"))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolNameAttribute(pub String);

impl AttributeData for SymbolNameAttribute {
    const NAME: &'static str = "#xir.symbol";

    fn parse(syntax: &str) -> Result<Self, String> {
        let value = StringAttribute::parse(&syntax.replacen(Self::NAME, StringAttribute::NAME, 1))?;
        require_canonical(Self(value.0), syntax)
    }

    fn print(&self) -> String {
        format!("#xir.symbol<{}>", encode_hex(self.0.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SymbolRefAttribute(pub String);

impl AttributeData for SymbolRefAttribute {
    const NAME: &'static str = "#xir.symbol_ref";

    fn parse(syntax: &str) -> Result<Self, String> {
        let value = StringAttribute::parse(&syntax.replacen(Self::NAME, StringAttribute::NAME, 1))?;
        require_canonical(Self(value.0), syntax)
    }

    fn print(&self) -> String {
        format!("#xir.symbol_ref<{}>", encode_hex(self.0.as_bytes()))
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FunctionRefsAttribute(pub Vec<String>);

impl AttributeData for FunctionRefsAttribute {
    const NAME: &'static str = "#xir.function_refs";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = parse_payload(syntax, Self::NAME)?;
        if payload.is_empty() {
            return require_canonical(Self(Vec::new()), syntax);
        }
        payload
            .split(',')
            .map(|value| String::from_utf8(decode_hex(value)?).map_err(|error| error.to_string()))
            .collect::<Result<Vec<_>, _>>()
            .map(Self)
            .and_then(|parsed| require_canonical(parsed, syntax))
    }

    fn print(&self) -> String {
        format!(
            "#xir.function_refs<{}>",
            self.0
                .iter()
                .map(|value| encode_hex(value.as_bytes()))
                .collect::<Vec<_>>()
                .join(",")
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteMapGraphAttribute(pub FiniteMapGraph);

impl AttributeData for FiniteMapGraphAttribute {
    const NAME: &'static str = "#xir.finite_map_graph";

    fn parse(syntax: &str) -> Result<Self, String> {
        let payload = parse_payload(syntax, Self::NAME)?;
        let (sets, maps) = payload.split_once(';').ok_or("expected sets;maps")?;
        let mut graph = FiniteMapGraph::default();
        for extent in parse_list(sets)? {
            graph.add_set(extent)?;
        }
        if !maps.is_empty() {
            for map in maps.split(',') {
                let (source, target) = map.split_once(':').ok_or("expected source:target")?;
                graph.add_map(
                    FiniteSetId(source.parse().map_err(|_| "invalid map source")?),
                    FiniteSetId(target.parse().map_err(|_| "invalid map target")?),
                )?;
            }
        }
        graph.verify()?;
        let parsed = Self(graph);
        require_canonical(parsed, syntax)
    }

    fn print(&self) -> String {
        let sets = join(self.0.sets.iter().map(|set| set.extent));
        let maps = self
            .0
            .maps
            .iter()
            .map(|map| format!("{}:{}", map.source.index(), map.target.index()))
            .collect::<Vec<_>>()
            .join(",");
        format!("#xir.finite_map_graph<{sets};{maps}>")
    }
}

fn require_canonical<T: AttributeData>(value: T, syntax: &str) -> Result<T, String> {
    if value.print() == syntax {
        Ok(value)
    } else {
        Err(format!("{} must use canonical spelling", T::NAME))
    }
}

fn parse_dtype(value: &str) -> Result<DType, String> {
    match value {
        "f16" => Ok(DType::F16),
        "bf16" => Ok(DType::Bf16),
        "f32" => Ok(DType::F32),
        "f64" => Ok(DType::F64),
        "i1" => Ok(DType::Bool),
        "i8" => Ok(DType::I8),
        "i16" => Ok(DType::I16),
        "i32" => Ok(DType::I32),
        "i64" => Ok(DType::I64),
        "u8" => Ok(DType::U8),
        "u16" => Ok(DType::U16),
        "u32" => Ok(DType::U32),
        "u64" => Ok(DType::U64),
        "c64" => Ok(DType::Complex64),
        "c128" => Ok(DType::Complex128),
        _ => Err(format!("unknown XIR dtype `{value}`")),
    }
}

fn parse_payload<'a>(syntax: &'a str, name: &str) -> Result<&'a str, String> {
    syntax
        .strip_prefix(name)
        .and_then(|value| value.strip_prefix('<'))
        .and_then(|value| value.strip_suffix('>'))
        .ok_or_else(|| format!("expected {name}<...>"))
}

fn parse_list(value: &str) -> Result<Vec<i64>, String> {
    if value.is_empty() {
        return Ok(Vec::new());
    }
    value
        .split(',')
        .map(|item| {
            item.parse()
                .map_err(|error: std::num::ParseIntError| error.to_string())
        })
        .collect()
}

fn join<T: ToString>(values: impl IntoIterator<Item = T>) -> String {
    values
        .into_iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decode_hex(value: &str) -> Result<Vec<u8>, String> {
    if !value.is_ascii() {
        return Err("hex payload must be ASCII".into());
    }
    if !value.len().is_multiple_of(2) {
        return Err("hex payload has odd length".into());
    }
    (0..value.len())
        .step_by(2)
        .map(|index| {
            u8::from_str_radix(&value[index..index + 2], 16).map_err(|error| error.to_string())
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ExactValuesAttribute(pub Vec<algebraic_real::AlgebraicReal>);

impl AttributeData for ExactValuesAttribute {
    const NAME: &'static str = "#xir.exact";

    fn parse(syntax: &str) -> Result<Self, String> {
        use algebraic_real::{AlgebraicReal, AlgebraicRealData};
        use num_bigint::{BigInt, BigUint};
        let payload = parse_payload(syntax, Self::NAME)?;
        let values = payload
            .strip_prefix("1|")
            .ok_or("unsupported exact-value schema")?;
        if values.is_empty() {
            return require_canonical(Self(Vec::new()), syntax);
        }
        let values = values
            .split(';')
            .map(|encoded| {
                let mut fields = encoded.split(',');
                let tag = fields.next().unwrap_or_default();
                let data = match tag {
                    "m" => {
                        let terms = fields
                            .map(|term| {
                                let (coefficient, radicand) =
                                    term.split_once(':').ok_or("invalid multiquadratic term")?;
                                let (numerator, denominator) = coefficient
                                    .split_once('/')
                                    .ok_or("invalid rational coefficient")?;
                                Ok((
                                    numerator
                                        .parse::<BigInt>()
                                        .map_err(|error| error.to_string())?,
                                    denominator
                                        .parse::<BigInt>()
                                        .map_err(|error| error.to_string())?,
                                    radicand
                                        .parse::<BigUint>()
                                        .map_err(|error| error.to_string())?,
                                ))
                            })
                            .collect::<Result<Vec<_>, String>>()?;
                        AlgebraicRealData::Multiquadratic(terms)
                    }
                    "p" => {
                        let root_index = fields
                            .next()
                            .ok_or("missing algebraic root index")?
                            .parse::<usize>()
                            .map_err(|error| error.to_string())?;
                        let coefficients = fields
                            .map(|coefficient| {
                                coefficient
                                    .parse::<BigInt>()
                                    .map_err(|error| error.to_string())
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        AlgebraicRealData::Polynomial {
                            coefficients,
                            root_index,
                        }
                    }
                    _ => return Err("unknown algebraic-real representation tag".to_owned()),
                };
                AlgebraicReal::from_data(data).map_err(|error| error.to_string())
            })
            .collect::<Result<Vec<_>, String>>()?;
        require_canonical(Self(values), syntax)
    }

    fn print(&self) -> String {
        use algebraic_real::AlgebraicRealData;
        let values = self
            .0
            .iter()
            .map(|value| match value.to_data() {
                AlgebraicRealData::Multiquadratic(terms) => {
                    let suffix = terms
                        .into_iter()
                        .map(|(numerator, denominator, radicand)| {
                            format!("{numerator}/{denominator}:{radicand}")
                        })
                        .collect::<Vec<_>>()
                        .join(",");
                    if suffix.is_empty() {
                        "m".to_owned()
                    } else {
                        format!("m,{suffix}")
                    }
                }
                AlgebraicRealData::Polynomial {
                    coefficients,
                    root_index,
                } => {
                    let coefficients = coefficients
                        .into_iter()
                        .map(|value| value.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    format!("p,{root_index},{coefficients}")
                }
            })
            .collect::<Vec<_>>()
            .join(";");
        format!("#xir.exact<1|{values}>")
    }
}
