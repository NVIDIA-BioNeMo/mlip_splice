// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, hash_map::DefaultHasher},
    hash::{Hash, Hasher},
};

use smallvec::SmallVec;

use crate::analysis::{FuncFacts, OpKey, is_pure, op_key};
use crate::ir::{Module, ValueId};
use crate::optimize::rebuild::Rebuilder;
use crate::shape::Shape;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct CseKey {
    op: OpKey,
    inputs: Vec<ValueId>,
    results: Vec<Shape>,
}

type CseResults = SmallVec<[ValueId; 1]>;
// `OpKey` may memoize algebraic facts. Keep the full key as a bucket value;
// the scalar fingerprint indexes buckets and full equality resolves collisions.
type CseFingerprint = u64;
type CseBucket = SmallVec<[(CseKey, CseResults); 1]>;

fn cse_fingerprint(key: &CseKey) -> CseFingerprint {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

pub(in crate::optimize) fn cse(module: &mut Module) -> Result<bool, String> {
    let mut changed = false;
    let funcs = module
        .funcs
        .iter()
        .map(|old| {
            let mut new = Rebuilder::new(old);
            let mut available = HashMap::<CseFingerprint, CseBucket>::new();
            for node in &old.nodes {
                let inputs = new.mapped_inputs(node)?;
                let key = is_pure(node.op.as_ref()).then(|| CseKey {
                    op: op_key(node.op.as_ref()),
                    inputs: inputs.to_vec(),
                    results: node
                        .results
                        .iter()
                        .map(|&result| old.value_shape(result).clone())
                        .collect(),
                });
                if let Some(key) = key {
                    let fingerprint = cse_fingerprint(&key);
                    if let Some(existing) = available.get(&fingerprint).and_then(|bucket| {
                        bucket
                            .iter()
                            .find_map(|(candidate, results)| (candidate == &key).then_some(results))
                    }) {
                        let existing = existing.clone();
                        new.alias(&node.results, &existing)?;
                        changed = true;
                        continue;
                    }
                    let results = new.emit(old, &node.results, node.op.clone(), inputs)?;
                    available
                        .entry(fingerprint)
                        .or_default()
                        .push((key, results));
                } else {
                    new.emit(old, &node.results, node.op.clone(), inputs)?;
                }
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}

/// Dead-code elimination consumes the shared analysis: it keeps exactly the
/// nodes the fact table marks live (an impure op, or one whose results reach a
/// function result) and drops the rest.
pub(in crate::optimize) fn dce(module: &mut Module) -> Result<bool, String> {
    let mut changed = false;
    let funcs = module
        .funcs
        .iter()
        .map(|old| {
            let facts = FuncFacts::analyze(old);
            let mut new = Rebuilder::new(old);
            for (index, node) in old.nodes.iter().enumerate() {
                if !facts.node(index).live {
                    changed = true;
                    continue;
                }
                let inputs = new.mapped_inputs(node)?;
                new.emit(old, &node.results, node.op.clone(), inputs)?;
            }
            new.finish(old)
        })
        .collect::<Result<Vec<_>, String>>()?;
    module.funcs = funcs;
    Ok(changed)
}
