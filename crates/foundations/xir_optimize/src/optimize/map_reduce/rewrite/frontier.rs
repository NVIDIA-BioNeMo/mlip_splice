// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub(in crate::optimize) struct FiniteMapFrontier {
    descriptions: ModuleFrontier,
    original: Option<ModuleExtraction>,
}

impl Iterator for FiniteMapFrontier {
    type Item = FiniteMapAlternative;
    fn next(&mut self) -> Option<Self::Item> {
        self.descriptions
            .next()
            .or_else(|| self.original.take())
            .map(|extraction| extraction.alternative)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.descriptions.len() + usize::from(self.original.is_some());
        (n, Some(n))
    }
}
impl ExactSizeIterator for FiniteMapFrontier {}

pub(in crate::optimize) fn describe_finite_map_frontier(
    module: &Module,
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
    checkpoint: &mut impl FnMut() -> Result<(), String>,
) -> Result<FiniteMapFrontier, String> {
    let mut descriptions = describe_module_frontier(module, max_terms, budget, checkpoint)?;
    let original = if descriptions
        .selections
        .as_slice()
        .iter()
        .all(|selection| selection.changed)
        && descriptions.original.is_none()
    {
        if descriptions.selections.len() == MAX_FRONTIER {
            let mut selections = descriptions.selections.collect::<Vec<_>>();
            selections.remove(MAX_FRONTIER / 2);
            descriptions.selections = selections.into_iter();
        }
        // An unchanged fallback is already a complete input module, not a new lowering.
        Some(original_module_extraction(module)?)
    } else {
        None
    };
    Ok(FiniteMapFrontier {
        descriptions,
        original,
    })
}

pub(super) struct ModuleFrontier {
    source: Module,
    paths: Vec<FuncPath>,
    selections: std::vec::IntoIter<PartialModuleExtraction>,
    original: Option<ModuleExtraction>,
}

impl Iterator for ModuleFrontier {
    type Item = ModuleExtraction;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(original) = self.original.take() {
            return Some(original);
        }
        let extraction = self.selections.next()?;
        let mut funcs = Vec::with_capacity(self.source.funcs.len());
        let mut path = extraction.path;
        while let Some(index) = path {
            let node = &self.paths[index];
            funcs.push((*node.func).clone());
            path = node.previous;
        }
        funcs.reverse();
        Some(ModuleExtraction {
            alternative: FiniteMapAlternative {
                module: Module {
                    finite_map_graph: self.source.finite_map_graph.clone(),
                    funcs,
                    entry: self.source.entry,
                },
                work: extraction.work,
            },
            physical_work: extraction.physical_work,
            changed: extraction.changed,
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.selections.len() + usize::from(self.original.is_some());
        (n, Some(n))
    }
}
impl ExactSizeIterator for ModuleFrontier {}

pub(super) fn describe_module_frontier(
    module: &Module,
    max_terms: usize,
    budget: &mut AlgebraicWorkBudget,
    checkpoint: &mut impl FnMut() -> Result<(), String>,
) -> Result<ModuleFrontier, String> {
    checkpoint()?;
    if max_terms == 0 || budget.remaining == 0 || budget.expression_limit == 0 {
        return Ok(ModuleFrontier {
            source: module.clone(),
            paths: vec![],
            selections: vec![].into_iter(),
            original: Some(original_module_extraction(module)?),
        });
    }
    let mut modules = vec![PartialModuleExtraction {
        path: None,
        work: FiniteMapWork::default(),
        physical_work: 0,
        changed: false,
    }];
    let mut paths = Vec::new();
    for (function, func) in module.funcs.iter().enumerate() {
        checkpoint()?;
        let funcs =
            extract_func_frontier(func, &module.finite_map_graph, function, max_terms, budget)?
                .into_iter()
                .map(|(func, work, changed)| {
                    let physical = physical_work(&func, &module.finite_map_graph)?;
                    Ok((Arc::new(func), work, physical, changed))
                })
                .collect::<Result<Vec<_>, String>>()?;
        let mut next = Vec::new();
        for accumulated in &modules {
            for (func, work, physical, changed) in &funcs {
                let path = paths.len();
                paths.push(FuncPath {
                    previous: accumulated.path,
                    func: func.clone(),
                });
                let extraction = PartialModuleExtraction {
                    path: Some(path),
                    work: accumulated.work.combine(*work),
                    physical_work: accumulated.physical_work + physical,
                    changed: accumulated.changed || *changed,
                };
                insert_partial_module_extraction(&mut next, extraction);
            }
        }
        modules = next;
    }
    checkpoint()?;
    Ok(ModuleFrontier {
        source: module.clone(),
        paths,
        selections: modules.into_iter(),
        original: None,
    })
}
