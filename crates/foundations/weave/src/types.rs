// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::any::Any;
use std::fmt;
use std::sync::Arc;

pub trait TypeData: Any + Send + Sync {
    const NAME: &'static str;
    fn parse(syntax: &str) -> Result<Self, String>
    where
        Self: Sized;
    fn print(&self) -> String;
}

pub trait AttributeData: Any + Send + Sync {
    const NAME: &'static str;
    fn parse(syntax: &str) -> Result<Self, String>
    where
        Self: Sized;
    fn print(&self) -> String;
}

#[derive(Clone)]
pub struct Type {
    syntax: Arc<str>,
    data: Option<Arc<dyn Any + Send + Sync>>,
}

#[derive(Clone)]
pub struct Attribute {
    syntax: Arc<str>,
    data: Option<Arc<dyn Any + Send + Sync>>,
}

fn valid_syntax(syntax: &str) -> bool {
    !syntax.trim().is_empty() && syntax.trim() == syntax && !syntax.contains(['\n', '\r'])
}

impl Type {
    pub fn opaque(syntax: impl Into<String>) -> Result<Self, String> {
        let syntax = syntax.into();
        if !valid_syntax(&syntax) {
            return Err("type syntax must be non-empty, trimmed, and single-line".into());
        }
        Ok(Self {
            syntax: syntax.into(),
            data: None,
        })
    }
    pub fn typed<T: TypeData>(data: T) -> Self {
        Self {
            syntax: data.print().into(),
            data: Some(Arc::new(data)),
        }
    }
    pub fn syntax(&self) -> &str {
        &self.syntax
    }
    pub fn extension_name(&self) -> Option<&str> {
        extension_name(&self.syntax, '!')
    }
    pub fn downcast_ref<T: TypeData>(&self) -> Option<&T> {
        self.data.as_deref()?.downcast_ref()
    }
}

impl Attribute {
    pub fn opaque(syntax: impl Into<String>) -> Result<Self, String> {
        let syntax = syntax.into();
        if !valid_syntax(&syntax) {
            return Err("attribute syntax must be non-empty, trimmed, and single-line".into());
        }
        Ok(Self {
            syntax: syntax.into(),
            data: None,
        })
    }
    pub fn typed<T: AttributeData>(data: T) -> Self {
        Self {
            syntax: data.print().into(),
            data: Some(Arc::new(data)),
        }
    }
    pub fn syntax(&self) -> &str {
        &self.syntax
    }
    pub fn extension_name(&self) -> Option<&str> {
        extension_name(&self.syntax, '#')
    }
    pub fn downcast_ref<T: AttributeData>(&self) -> Option<&T> {
        self.data.as_deref()?.downcast_ref()
    }
}

fn extension_name(syntax: &str, sigil: char) -> Option<&str> {
    let rest = syntax.strip_prefix(sigil)?;
    let end = rest.find(['<', '(', '[', '{']).unwrap_or(rest.len());
    let name = &rest[..end];
    let (dialect, mnemonic) = name.split_once('.')?;
    (!dialect.is_empty() && !mnemonic.is_empty()).then_some(name)
}

macro_rules! value_impls {
    ($ty:ident) => {
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.syntax == other.syntax
            }
        }
        impl Eq for $ty {}
        impl std::hash::Hash for $ty {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                self.syntax.hash(state);
            }
        }
        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_tuple(stringify!($ty)).field(&self.syntax).finish()
            }
        }
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                self.syntax.fmt(f)
            }
        }
    };
}
value_impls!(Type);
value_impls!(Attribute);
