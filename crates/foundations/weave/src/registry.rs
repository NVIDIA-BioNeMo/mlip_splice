// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::diagnostic::Diagnostic;
use crate::ir::{Ir, OperationId};
use crate::name::{DialectName, OperationName};
use crate::types::{Attribute, AttributeData, Type, TypeData};
use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

type OperationVerifier = dyn Fn(&Ir, OperationId, &mut Vec<Diagnostic>) + Send + Sync;
type TypeParser = dyn Fn(&str) -> Result<Type, String> + Send + Sync;
type AttributeParser = dyn Fn(&str) -> Result<Attribute, String> + Send + Sync;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperationTraits {
    pub terminator: bool,
}

pub struct OperationDefinition {
    pub traits: OperationTraits,
    verifier: Arc<OperationVerifier>,
    interfaces: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl OperationDefinition {
    pub fn new() -> Self {
        Self {
            traits: OperationTraits::default(),
            verifier: Arc::new(|_, _, _| {}),
            interfaces: HashMap::new(),
        }
    }
    pub fn terminator(mut self) -> Self {
        self.traits.terminator = true;
        self
    }
    pub fn verifier(
        mut self,
        verifier: impl Fn(&Ir, OperationId, &mut Vec<Diagnostic>) + Send + Sync + 'static,
    ) -> Self {
        self.verifier = Arc::new(verifier);
        self
    }
    pub fn interface<T: Any + Send + Sync>(mut self, interface: T) -> Self {
        self.interfaces
            .insert(TypeId::of::<T>(), Arc::new(interface));
        self
    }
    pub fn get_interface<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.interfaces.get(&TypeId::of::<T>())?.downcast_ref()
    }
    pub(crate) fn verify(
        &self,
        ir: &Ir,
        operation: OperationId,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        (self.verifier)(ir, operation, diagnostics);
    }
}
impl Default for OperationDefinition {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub enum RegistryError {
    DuplicateDialect(String),
    DuplicateOperation(String),
    DuplicateType(String),
    DuplicateAttribute(String),
    MissingDialect(String),
    InvalidExtensionName(String),
}
impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for RegistryError {}

#[derive(Default)]
pub struct Registry {
    dialects: BTreeMap<DialectName, ()>,
    operations: BTreeMap<OperationName, OperationDefinition>,
    types: BTreeMap<String, Arc<TypeParser>>,
    attributes: BTreeMap<String, Arc<AttributeParser>>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_dialect(&mut self, name: DialectName) -> Result<(), RegistryError> {
        if self.dialects.contains_key(&name) {
            return Err(RegistryError::DuplicateDialect(name.to_string()));
        }
        self.dialects.insert(name, ());
        Ok(())
    }

    pub fn register_operation(
        &mut self,
        name: OperationName,
        definition: OperationDefinition,
    ) -> Result<(), RegistryError> {
        if !self.dialects.keys().any(|d| d.as_str() == name.dialect()) {
            return Err(RegistryError::MissingDialect(name.dialect().into()));
        }
        if self.operations.contains_key(&name) {
            return Err(RegistryError::DuplicateOperation(name.to_string()));
        }
        self.operations.insert(name, definition);
        Ok(())
    }

    pub fn register_type<T: TypeData>(&mut self) -> Result<(), RegistryError> {
        validate_extension_name(T::NAME, '!')?;
        let name = T::NAME[1..].to_owned();
        if self.types.contains_key(&name) {
            return Err(RegistryError::DuplicateType(name));
        }
        self.types
            .insert(name, Arc::new(|syntax| T::parse(syntax).map(Type::typed)));
        Ok(())
    }

    pub fn register_attribute<T: AttributeData>(&mut self) -> Result<(), RegistryError> {
        validate_extension_name(T::NAME, '#')?;
        let name = T::NAME[1..].to_owned();
        if self.attributes.contains_key(&name) {
            return Err(RegistryError::DuplicateAttribute(name));
        }
        self.attributes.insert(
            name,
            Arc::new(|syntax| T::parse(syntax).map(Attribute::typed)),
        );
        Ok(())
    }

    pub fn operation(&self, name: &OperationName) -> Option<&OperationDefinition> {
        self.operations.get(name)
    }
    pub fn contains_dialect(&self, name: &str) -> bool {
        self.dialects.keys().any(|d| d.as_str() == name)
    }
    pub fn contains_type(&self, syntax: &str) -> bool {
        extension_name(syntax, '!').is_none_or(|name| self.types.contains_key(name))
    }
    pub fn contains_attribute(&self, syntax: &str) -> bool {
        extension_name(syntax, '#').is_none_or(|name| self.attributes.contains_key(name))
    }

    pub fn parse_type(&self, syntax: &str) -> Result<Type, String> {
        match extension_name(syntax, '!').and_then(|name| self.types.get(name)) {
            Some(parser) => parser(syntax),
            None => Type::opaque(syntax),
        }
    }

    pub fn parse_attribute(&self, syntax: &str) -> Result<Attribute, String> {
        match extension_name(syntax, '#').and_then(|name| self.attributes.get(name)) {
            Some(parser) => parser(syntax),
            None => Attribute::opaque(syntax),
        }
    }
}

fn validate_extension_name(name: &str, sigil: char) -> Result<(), RegistryError> {
    if extension_name(name, sigil) == Some(&name[1..]) && !name[1..].contains(['<', '(', '[', '{'])
    {
        Ok(())
    } else {
        Err(RegistryError::InvalidExtensionName(name.into()))
    }
}

fn extension_name(syntax: &str, sigil: char) -> Option<&str> {
    let rest = syntax.strip_prefix(sigil)?;
    let end = rest.find(['<', '(', '[', '{']).unwrap_or(rest.len());
    let name = &rest[..end];
    let (dialect, mnemonic) = name.split_once('.')?;
    (!dialect.is_empty() && !mnemonic.is_empty()).then_some(name)
}
