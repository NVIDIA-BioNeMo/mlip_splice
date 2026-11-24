// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::diagnostic::Diagnostic;
use crate::ir::{BlockId, Ir, OperationId, RegionId, ValueId};
use crate::name::OperationName;
use crate::registry::Registry;
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Debug)]
struct AstOperation {
    results: Vec<String>,
    name: String,
    operands: Vec<String>,
    successors: Vec<(String, Vec<String>)>,
    attributes: Vec<(String, String)>,
    operand_types: Vec<String>,
    result_types: Vec<String>,
    regions: Vec<AstRegion>,
}
#[derive(Debug)]
struct AstRegion {
    blocks: Vec<AstBlock>,
}
#[derive(Debug)]
struct AstBlock {
    name: String,
    arguments: Vec<(String, String)>,
    operations: Vec<AstOperation>,
}

pub fn parse(source: &str, registry: &Registry) -> Result<Ir, Vec<Diagnostic>> {
    let mut parser = Parser { source, cursor: 0 };
    let ast = match parser.document() {
        Ok(ast) => ast,
        Err(diagnostic) => return Err(vec![diagnostic]),
    };
    let mut lowerer = Lowerer {
        registry,
        ir: Ir::new(),
        value_scopes: vec![HashMap::new()],
        diagnostics: Vec::new(),
    };
    for operation in &ast {
        if let Some(id) = lowerer.operation(operation, None, None)
            && let Err(error) = lowerer.ir.add_root(id)
        {
            lowerer
                .diagnostics
                .push(Diagnostic::error(error.to_string()));
        }
    }
    if lowerer.diagnostics.is_empty() {
        Ok(lowerer.ir)
    } else {
        Err(lowerer.diagnostics)
    }
}

pub fn print(ir: &Ir) -> Result<String, Diagnostic> {
    let mut printer = Printer {
        ir,
        value_names: HashMap::new(),
        next_value: 0,
        output: String::new(),
    };
    for root in ir.roots() {
        printer.assign_operation(*root)?;
    }
    printer.output.push_str("module {\n");
    for root in ir.roots() {
        printer.operation(*root, 1, None)?;
    }
    printer.output.push_str("}\n");
    Ok(printer.output)
}

struct Lowerer<'a> {
    registry: &'a Registry,
    ir: Ir,
    value_scopes: Vec<HashMap<String, ValueId>>,
    diagnostics: Vec<Diagnostic>,
}

impl Lowerer<'_> {
    fn operation(
        &mut self,
        ast: &AstOperation,
        parent: Option<BlockId>,
        blocks: Option<&HashMap<String, BlockId>>,
    ) -> Option<OperationId> {
        if ast.results.len() != ast.result_types.len() {
            self.diagnostics.push(Diagnostic::error(format!(
                "operation `{}` has {} result names but {} result types",
                ast.name,
                ast.results.len(),
                ast.result_types.len()
            )));
            return None;
        }
        if ast.operands.len() != ast.operand_types.len() {
            self.diagnostics.push(Diagnostic::error(format!(
                "operation `{}` has {} operands but {} operand types",
                ast.name,
                ast.operands.len(),
                ast.operand_types.len()
            )));
            return None;
        }
        let name = match OperationName::new(ast.name.clone()) {
            Ok(name) => name,
            Err(error) => {
                self.diagnostics.push(Diagnostic::error(error.to_string()));
                return None;
            }
        };
        let operands: Vec<_> = ast
            .operands
            .iter()
            .filter_map(|name| self.lookup_value(name))
            .collect();
        if operands.len() != ast.operands.len() {
            return None;
        }
        let mut operand_types = Vec::new();
        for syntax in &ast.operand_types {
            match self.registry.parse_type(syntax) {
                Ok(ty) => operand_types.push(ty),
                Err(error) => self.diagnostics.push(Diagnostic::error(format!(
                    "cannot parse type `{syntax}`: {error}"
                ))),
            }
        }
        if operand_types.len() != ast.operand_types.len() {
            return None;
        }
        let mut operand_types_match = true;
        for ((operand, declared), name) in operands.iter().zip(&operand_types).zip(&ast.operands) {
            let actual = &self
                .ir
                .value(*operand)
                .expect("resolved SSA value exists")
                .ty;
            if actual != declared {
                operand_types_match = false;
                self.diagnostics.push(Diagnostic::error(format!(
                    "operand `%{name}` has type `{actual}`, but the operation declares `{declared}`"
                )));
            }
        }
        if !operand_types_match {
            return None;
        }
        let mut result_types = Vec::new();
        for syntax in &ast.result_types {
            match self.registry.parse_type(syntax) {
                Ok(ty) => result_types.push(ty),
                Err(error) => self.diagnostics.push(Diagnostic::error(format!(
                    "cannot parse type `{syntax}`: {error}"
                ))),
            }
        }
        if result_types.len() != ast.result_types.len() {
            return None;
        }
        let id = match self.ir.create_operation(name, operands, result_types) {
            Ok(id) => id,
            Err(error) => {
                self.diagnostics.push(Diagnostic::error(error.to_string()));
                return None;
            }
        };
        if let Some(block) = parent
            && let Err(error) = self.ir.append_operation(block, id)
        {
            self.diagnostics.push(Diagnostic::error(error.to_string()));
        }
        let results = self
            .ir
            .operation(id)
            .expect("new operation")
            .results
            .clone();
        for (name, syntax) in &ast.attributes {
            match self.registry.parse_attribute(syntax) {
                Ok(attribute) => {
                    if let Ok(Some(_)) = self.ir.set_attribute(id, name, attribute) {
                        self.diagnostics
                            .push(Diagnostic::error(format!("duplicate attribute `{name}`")));
                    }
                }
                Err(error) => self.diagnostics.push(Diagnostic::error(format!(
                    "cannot parse attribute `{syntax}`: {error}"
                ))),
            }
        }
        for region in &ast.regions {
            self.value_scopes.push(HashMap::new());
            self.region(region, id);
            self.value_scopes.pop();
        }
        if let Some(blocks) = blocks {
            for (block_name, argument_names) in &ast.successors {
                let Some(block) = blocks.get(block_name).copied() else {
                    self.diagnostics.push(Diagnostic::error(format!(
                        "unknown successor `^{block_name}`"
                    )));
                    continue;
                };
                let arguments: Vec<_> = argument_names
                    .iter()
                    .filter_map(|name| self.lookup_value(name))
                    .collect();
                if arguments.len() == argument_names.len()
                    && let Err(error) = self.ir.add_successor(id, block, arguments)
                {
                    self.diagnostics.push(Diagnostic::error(error.to_string()));
                }
            }
        } else if !ast.successors.is_empty() {
            self.diagnostics
                .push(Diagnostic::error("root operation cannot have successors"));
        }
        for (name, value) in ast.results.iter().zip(results) {
            self.bind_value(name, value);
        }
        Some(id)
    }

    fn region(&mut self, ast: &AstRegion, parent: OperationId) {
        let Ok(region) = self.ir.append_region(parent) else {
            return;
        };
        let mut blocks = HashMap::new();
        for block in &ast.blocks {
            let mut types = Vec::new();
            for (_, syntax) in &block.arguments {
                match self.registry.parse_type(syntax) {
                    Ok(ty) => types.push(ty),
                    Err(error) => self.diagnostics.push(Diagnostic::error(format!(
                        "cannot parse type `{syntax}`: {error}"
                    ))),
                }
            }
            if types.len() != block.arguments.len() {
                continue;
            }
            let id = self
                .ir
                .append_block(region, types)
                .expect("new region exists");
            if blocks.insert(block.name.clone(), id).is_some() {
                self.diagnostics.push(Diagnostic::error(format!(
                    "duplicate block `^{}`",
                    block.name
                )));
            }
            let values = self
                .ir
                .block(id)
                .expect("new block exists")
                .arguments
                .clone();
            for ((name, _), value) in block.arguments.iter().zip(values) {
                self.bind_value(name, value);
            }
        }
        for block in &ast.blocks {
            if let Some(id) = blocks.get(&block.name).copied() {
                for operation in &block.operations {
                    self.operation(operation, Some(id), Some(&blocks));
                }
            }
        }
    }

    fn bind_value(&mut self, name: &str, value: ValueId) {
        let scope = self
            .value_scopes
            .last_mut()
            .expect("parser always has a value scope");
        if scope.contains_key(name) {
            self.diagnostics
                .push(Diagnostic::error(format!("duplicate SSA value `%{name}`")));
        } else {
            scope.insert(name.to_owned(), value);
        }
    }

    fn lookup_value(&mut self, name: &str) -> Option<ValueId> {
        self.value_scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
            .or_else(|| {
                self.diagnostics
                    .push(Diagnostic::error(format!("unknown SSA value `%{name}`")));
                None
            })
    }
}

struct Printer<'a> {
    ir: &'a Ir,
    value_names: HashMap<ValueId, usize>,
    next_value: usize,
    output: String,
}

impl Printer<'_> {
    fn assign_value(&mut self, value: ValueId) {
        self.value_names.entry(value).or_insert_with(|| {
            let name = self.next_value;
            self.next_value += 1;
            name
        });
    }
    fn assign_operation(&mut self, id: OperationId) -> Result<(), Diagnostic> {
        let operation = self
            .ir
            .operation(id)
            .ok_or_else(|| Diagnostic::error(format!("missing operation {id:?}")))?;
        for result in &operation.results {
            self.assign_value(*result);
        }
        for region in &operation.regions {
            let region = self
                .ir
                .region(*region)
                .ok_or_else(|| Diagnostic::error(format!("missing region {region:?}")))?;
            for block in &region.blocks {
                let block = self
                    .ir
                    .block(*block)
                    .ok_or_else(|| Diagnostic::error(format!("missing block {block:?}")))?;
                for argument in &block.arguments {
                    self.assign_value(*argument);
                }
                for operation in &block.operations {
                    self.assign_operation(*operation)?;
                }
            }
        }
        Ok(())
    }
    fn value(&self, value: ValueId) -> Result<usize, Diagnostic> {
        self.value_names.get(&value).copied().ok_or_else(|| {
            Diagnostic::error(format!("value {value:?} is outside printed ownership tree"))
        })
    }
    fn operation(
        &mut self,
        id: OperationId,
        indent: usize,
        block_names: Option<&HashMap<BlockId, usize>>,
    ) -> Result<(), Diagnostic> {
        let operation = self
            .ir
            .operation(id)
            .ok_or_else(|| Diagnostic::error(format!("missing operation {id:?}")))?;
        write!(self.output, "{}(", "  ".repeat(indent)).unwrap();
        for (index, result) in operation.results.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            write!(self.output, "%{}", self.value(*result)?).unwrap();
        }
        write!(self.output, ") = \"{}\"(", operation.name).unwrap();
        for (index, operand) in operation.operands.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            write!(self.output, "%{}", self.value(*operand)?).unwrap();
        }
        self.output.push_str(") [");
        for (index, successor) in operation.successors.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            let number = block_names
                .and_then(|names| names.get(&successor.block))
                .ok_or_else(|| Diagnostic::error("successor is outside operation region"))?;
            write!(self.output, "^bb{number}(").unwrap();
            for (arg_index, argument) in successor.arguments.iter().enumerate() {
                if arg_index != 0 {
                    self.output.push_str(", ");
                }
                write!(self.output, "%{}", self.value(*argument)?).unwrap();
            }
            self.output.push(')');
        }
        self.output.push_str("] {");
        for (index, (name, attribute)) in operation.attributes.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            write!(self.output, "{name} = \"{}\"", escape(attribute.syntax())).unwrap();
        }
        self.output.push_str("} : (");
        for (index, operand) in operation.operands.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            let ty = &self
                .ir
                .value(*operand)
                .ok_or_else(|| Diagnostic::error(format!("missing value {operand:?}")))?
                .ty;
            write!(self.output, "\"{}\"", escape(ty.syntax())).unwrap();
        }
        self.output.push_str(") -> (");
        for (index, result) in operation.results.iter().enumerate() {
            if index != 0 {
                self.output.push_str(", ");
            }
            let ty = &self
                .ir
                .value(*result)
                .ok_or_else(|| Diagnostic::error(format!("missing value {result:?}")))?
                .ty;
            write!(self.output, "\"{}\"", escape(ty.syntax())).unwrap();
        }
        self.output.push(')');
        if operation.regions.is_empty() {
            self.output.push('\n');
            return Ok(());
        }
        self.output.push_str(" regions (\n");
        for (index, region) in operation.regions.iter().enumerate() {
            if index != 0 {
                self.output.push_str(",\n");
            }
            self.region(*region, indent + 1)?;
        }
        write!(self.output, "\n{})\n", "  ".repeat(indent)).unwrap();
        Ok(())
    }
    fn region(&mut self, id: RegionId, indent: usize) -> Result<(), Diagnostic> {
        let region = self
            .ir
            .region(id)
            .ok_or_else(|| Diagnostic::error(format!("missing region {id:?}")))?;
        let names: HashMap<_, _> = region
            .blocks
            .iter()
            .enumerate()
            .map(|(number, block)| (*block, number))
            .collect();
        writeln!(self.output, "{}{{", "  ".repeat(indent)).unwrap();
        for block_id in &region.blocks {
            let block = self
                .ir
                .block(*block_id)
                .ok_or_else(|| Diagnostic::error(format!("missing block {block_id:?}")))?;
            write!(
                self.output,
                "{}^bb{}(",
                "  ".repeat(indent + 1),
                names[block_id]
            )
            .unwrap();
            for (index, argument) in block.arguments.iter().enumerate() {
                if index != 0 {
                    self.output.push_str(", ");
                }
                let value = self
                    .ir
                    .value(*argument)
                    .ok_or_else(|| Diagnostic::error(format!("missing value {argument:?}")))?;
                write!(
                    self.output,
                    "%{}: \"{}\"",
                    self.value(*argument)?,
                    escape(value.ty.syntax())
                )
                .unwrap();
            }
            self.output.push_str("):\n");
            for operation in &block.operations {
                self.operation(*operation, indent + 2, Some(&names))?;
            }
        }
        write!(self.output, "{}}}", "  ".repeat(indent)).unwrap();
        Ok(())
    }
}

fn escape(value: &str) -> String {
    value
        .chars()
        .flat_map(|c| match c {
            '\\' => "\\\\".chars().collect::<Vec<_>>(),
            '"' => "\\\"".chars().collect(),
            '\n' => "\\n".chars().collect(),
            '\r' => "\\r".chars().collect(),
            '\t' => "\\t".chars().collect(),
            c => vec![c],
        })
        .collect()
}

struct Parser<'a> {
    source: &'a str,
    cursor: usize,
}

impl Parser<'_> {
    fn document(&mut self) -> Result<Vec<AstOperation>, Diagnostic> {
        self.keyword("module")?;
        self.punct('{')?;
        let mut operations = Vec::new();
        while !self.peek('}') {
            operations.push(self.operation()?);
        }
        self.punct('}')?;
        self.space();
        if self.cursor != self.source.len() {
            return Err(self.error("trailing input"));
        }
        Ok(operations)
    }
    fn operation(&mut self) -> Result<AstOperation, Diagnostic> {
        let results = self.names('(', ')', '%')?;
        self.punct('=')?;
        let name = self.string()?;
        let operands = self.names('(', ')', '%')?;
        self.punct('[')?;
        let mut successors = Vec::new();
        while !self.peek(']') {
            self.punct('^')?;
            let block = self.ident()?;
            let arguments = self.names('(', ')', '%')?;
            successors.push((block, arguments));
            if !self.take(',') {
                break;
            }
        }
        self.punct(']')?;
        self.punct('{')?;
        let mut attributes = Vec::new();
        while !self.peek('}') {
            let name = self.ident()?;
            self.punct('=')?;
            let value = self.string()?;
            attributes.push((name, value));
            if !self.take(',') {
                break;
            }
        }
        self.punct('}')?;
        self.punct(':')?;
        let operand_types = self.strings()?;
        self.arrow()?;
        let result_types = self.strings()?;
        let mut regions = Vec::new();
        if self.try_keyword("regions") {
            self.punct('(')?;
            while !self.peek(')') {
                regions.push(self.region()?);
                if !self.take(',') {
                    break;
                }
            }
            self.punct(')')?;
        }
        Ok(AstOperation {
            results,
            name,
            operands,
            successors,
            attributes,
            operand_types,
            result_types,
            regions,
        })
    }
    fn region(&mut self) -> Result<AstRegion, Diagnostic> {
        self.punct('{')?;
        let mut blocks = Vec::new();
        while !self.peek('}') {
            self.punct('^')?;
            let name = self.ident()?;
            self.punct('(')?;
            let mut arguments = Vec::new();
            while !self.peek(')') {
                self.punct('%')?;
                let argument = self.ident()?;
                self.punct(':')?;
                let ty = self.string()?;
                arguments.push((argument, ty));
                if !self.take(',') {
                    break;
                }
            }
            self.punct(')')?;
            self.punct(':')?;
            let mut operations = Vec::new();
            while !self.peek('^') && !self.peek('}') {
                operations.push(self.operation()?);
            }
            blocks.push(AstBlock {
                name,
                arguments,
                operations,
            });
        }
        self.punct('}')?;
        Ok(AstRegion { blocks })
    }
    fn strings(&mut self) -> Result<Vec<String>, Diagnostic> {
        self.punct('(')?;
        let mut values = Vec::new();
        while !self.peek(')') {
            values.push(self.string()?);
            if !self.take(',') {
                break;
            }
        }
        self.punct(')')?;
        Ok(values)
    }
    fn names(&mut self, open: char, close: char, sigil: char) -> Result<Vec<String>, Diagnostic> {
        self.punct(open)?;
        let mut names = Vec::new();
        while !self.peek(close) {
            self.punct(sigil)?;
            names.push(self.ident()?);
            if !self.take(',') {
                break;
            }
        }
        self.punct(close)?;
        Ok(names)
    }
    fn string(&mut self) -> Result<String, Diagnostic> {
        self.space();
        let start = self.cursor;
        self.punct_raw('"')?;
        let mut value = String::new();
        loop {
            let Some(c) = self.next() else {
                return Err(Diagnostic::error("unterminated string").at(start, self.cursor - start));
            };
            match c {
                '"' => break,
                '\\' => {
                    let Some(escaped) = self.next() else {
                        return Err(self.error("unterminated escape"));
                    };
                    value.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '"' => '"',
                        _ => return Err(self.error("unsupported escape")),
                    });
                }
                c if c.is_control() => return Err(self.error("control character in string")),
                c => value.push(c),
            }
        }
        self.space();
        Ok(value)
    }
    fn ident(&mut self) -> Result<String, Diagnostic> {
        self.space();
        let start = self.cursor;
        while self.source[self.cursor..]
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-'))
        {
            self.next();
        }
        if self.cursor == start {
            Err(self.error("expected identifier"))
        } else {
            let value = self.source[start..self.cursor].to_owned();
            self.space();
            Ok(value)
        }
    }
    fn keyword(&mut self, value: &str) -> Result<(), Diagnostic> {
        if self.try_keyword(value) {
            Ok(())
        } else {
            Err(self.error(format!("expected `{value}`")))
        }
    }
    fn try_keyword(&mut self, value: &str) -> bool {
        self.space();
        if self.source[self.cursor..].starts_with(value) {
            let end = self.cursor + value.len();
            if self.source[end..]
                .chars()
                .next()
                .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_')
            {
                self.cursor = end;
                self.space();
                return true;
            }
        }
        false
    }
    fn arrow(&mut self) -> Result<(), Diagnostic> {
        self.space();
        if self.source[self.cursor..].starts_with("->") {
            self.cursor += 2;
            self.space();
            Ok(())
        } else {
            Err(self.error("expected `->`"))
        }
    }
    fn punct(&mut self, expected: char) -> Result<(), Diagnostic> {
        self.space();
        self.punct_raw(expected)?;
        self.space();
        Ok(())
    }
    fn punct_raw(&mut self, expected: char) -> Result<(), Diagnostic> {
        if self.next() == Some(expected) {
            Ok(())
        } else {
            Err(self.error(format!("expected `{expected}`")))
        }
    }
    fn take(&mut self, expected: char) -> bool {
        self.space();
        if self.source[self.cursor..].starts_with(expected) {
            self.next();
            self.space();
            true
        } else {
            false
        }
    }
    fn peek(&mut self, expected: char) -> bool {
        self.space();
        self.source[self.cursor..].starts_with(expected)
    }
    fn next(&mut self) -> Option<char> {
        let c = self.source[self.cursor..].chars().next()?;
        self.cursor += c.len_utf8();
        Some(c)
    }
    fn space(&mut self) {
        loop {
            while self.source[self.cursor..]
                .chars()
                .next()
                .is_some_and(char::is_whitespace)
            {
                self.next();
            }
            if self.source[self.cursor..].starts_with("//") {
                while self.next().is_some_and(|c| c != '\n') {}
            } else {
                break;
            }
        }
    }
    fn error(&self, message: impl Into<String>) -> Diagnostic {
        Diagnostic::error(message).at(self.cursor, 1)
    }
}
