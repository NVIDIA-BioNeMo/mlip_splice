// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A readable, HLO-ish text rendering of a [`Module`]. Values are `%n`,
//! functions are `@name`/`@id`, shapes are `dtype[d0,d1,...]`. This is the
//! debugging and test oracle for the IR.

use std::fmt;

use crate::ir::{Func, Module, ValueId};
use crate::shape::Shape;

fn shape_str(s: &Shape) -> String {
    let dims = s
        .dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let domains = s
        .domain_axes
        .iter()
        .map(|domain| format!("@s{}", domain.index()))
        .collect::<Vec<_>>()
        .join(",");
    if domains.is_empty() {
        format!("{}[{}]", s.dtype.name(), dims)
    } else {
        format!("{}<{domains}>[{}]", s.dtype.name(), dims)
    }
}

fn val_list(vs: &[ValueId]) -> String {
    vs.iter()
        .map(|v| format!("%{}", v.index()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn results_type(f: &Func) -> String {
    let tys = f
        .results
        .iter()
        .map(|&r| shape_str(f.value_shape(r)))
        .collect::<Vec<_>>()
        .join(", ");
    if f.results.len() == 1 {
        tys
    } else {
        format!("({tys})")
    }
}

fn write_func(f: &Func, out: &mut String) {
    let params = f
        .params
        .iter()
        .map(|&p| format!("%{}: {}", p.index(), shape_str(f.value_shape(p))))
        .collect::<Vec<_>>()
        .join(", ");
    out.push_str(&format!(
        "func @{}({params}) -> {} {{\n",
        f.name,
        results_type(f)
    ));

    for node in &f.nodes {
        let lhs = val_list(&node.results);
        let operands = if node.inputs.is_empty() {
            String::new()
        } else {
            format!(" {}", val_list(&node.inputs))
        };
        let attrs = node.op.attrs();
        let attr_str = if attrs.is_empty() {
            String::new()
        } else {
            let body = attrs
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(" {{{body}}}")
        };
        let res_ty = node
            .results
            .iter()
            .map(|&r| shape_str(f.value_shape(r)))
            .collect::<Vec<_>>()
            .join(", ");
        out.push_str(&format!(
            "  {lhs} = {}{operands}{attr_str} : {res_ty}\n",
            node.op.mnemonic()
        ));
    }

    out.push_str(&format!("  return {}\n}}\n", val_list(&f.results)));
}

pub fn module_to_string(m: &Module) -> String {
    let mut out = String::new();
    for (index, set) in m.finite_map_graph.sets.iter().enumerate() {
        out.push_str(&format!("index_set @s{index} {{extent={}}}\n", set.extent));
    }
    for (index, map) in m.finite_map_graph.maps.iter().enumerate() {
        out.push_str(&format!(
            "index_map @m{index} : @s{} -> @s{}\n",
            map.source.index(),
            map.target.index()
        ));
    }
    if !m.finite_map_graph.sets.is_empty() {
        out.push('\n');
    }
    for (i, f) in m.funcs.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        write_func(f, &mut out);
    }
    out
}

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&module_to_string(self))
    }
}

impl fmt::Display for Func {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut out = String::new();
        write_func(self, &mut out);
        f.write_str(&out)
    }
}
