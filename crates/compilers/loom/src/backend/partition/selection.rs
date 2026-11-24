// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use gpu_ir::{Builder, Function, OpId, Ty, ValueId};

use super::{
    BTreeSet, Bindings, ElementOrder, KernelPart, SelectedOutputs, SelectedTerms, Term,
    additive_term_spans, cover_terms, order,
};

/// Per selected output element, the additive terms of its left-fold sum, in fold
/// order (a Reductions-policy init is the first term).
/// The terms each selected output element is made of, in the arena `f` is.
///
/// A part names its elements and its terms, never the values of the evaluation
/// it was planned against, so each selection is re-derived from `f`.
pub(in super::super) fn selected_term_values(
    f: &Function,
    ops: &[OpId],
    outputs: &[Vec<ValueId>],
    part: &KernelPart,
) -> Result<(SelectedTerms, Vec<Term>), String> {
    let bindings = Bindings::of(f, ops)?;
    let mut selected: SelectedTerms = outputs
        .iter()
        .map(|values| vec![None; values.len()])
        .collect::<Vec<_>>();
    let mut roots = Vec::new();
    for (output, selections) in part.elements.iter().enumerate() {
        for selection in selections {
            let (terms, spans) =
                additive_term_spans(f, outputs[output][selection.element], &bindings);
            // Term enumeration distributes products over sums; re-folding the
            // enumerated terms would undo the factored form the algebra chose
            // wherever no split actually happens. A selection holding every one
            // of several terms keeps the element's original value, and a partial
            // selection re-folds through the largest complete subtrees its term
            // set covers, so only the boundary of an actual cut pays
            // distribution. Single terms stay on the enumerated form, which
            // carries multiplication-by-zero rewrites. Term
            // guards never block re-folding: they are provenance for skip
            // regions, and both the original value and each subtree span carry
            // their guards in place.
            let refold = terms.len() > 1;
            let chosen = if refold && selection.terms.len() == terms.len() {
                vec![Term::Held(outputs[output][selection.element])]
            } else if refold {
                cover_terms(&selection.terms, &terms, &spans).ok_or_else(|| {
                    format!(
                        "loom output {output} element {} selects a term beyond {}",
                        selection.element,
                        terms.len()
                    )
                })?
            } else {
                selection
                    .terms
                    .iter()
                    .map(|&index| {
                        terms.get(index).cloned().ok_or_else(|| {
                            format!(
                                "loom output {output} element {} has no additive term {index}",
                                selection.element
                            )
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?
            };
            roots.extend(chosen.iter().cloned());
            selected[output][selection.element] = Some(chosen);
        }
    }
    Ok((selected, roots))
}

/// Regroups the flat bindings by consuming output element: each element's
/// not-yet-placed dependency closure is placed as one block (in trace order, so
/// every value still follows its operands), then the element's terms fold into
/// `loom_ret_{o}_{e}_{k}` accumulator bindings in unchanged left-fold order —
/// results stay bit-identical under every math policy. An element's private
/// values die at its block boundary instead of staying live across every
/// element's interleaved trace, which is what keeps ptxas register pressure
/// bounded; only values shared between elements persist, and `order` sequences
/// the blocks (see [`ElementOrder`]).
pub(in super::super) fn group_bindings_by_element(
    b: &mut Builder,
    bindings: Vec<OpId>,
    terms: &SelectedTerms,
    order: ElementOrder,
    element_ty: impl Fn(usize) -> Ty,
) -> Result<(Vec<OpId>, SelectedOutputs), String> {
    let (index_of, operands) = order::binding_operands(b.function(), &bindings)?;

    let mut emitted = vec![false; bindings.len()];
    let mut block = Vec::with_capacity(bindings.len());
    let mut emit_closure = |roots: &BTreeSet<usize>, block: &mut Vec<usize>| {
        // Bindings only reference earlier bindings, so ascending index order
        // inside the closure is a valid topological order.
        let mut closure = BTreeSet::new();
        let mut stack = roots
            .iter()
            .copied()
            .filter(|&root| !emitted[root])
            .collect::<Vec<_>>();
        while let Some(index) = stack.pop() {
            if !closure.insert(index) {
                continue;
            }
            stack.extend(
                operands[index]
                    .iter()
                    .copied()
                    .filter(|&dep| !emitted[dep] && !closure.contains(&dep)),
            );
        }
        for index in closure {
            emitted[index] = true;
            block.push(index);
        }
    };

    let mut selected: SelectedOutputs = terms
        .iter()
        .map(|values| vec![None; values.len()])
        .collect();
    let mut slots = bindings.into_iter().map(Some).collect::<Vec<_>>();
    let mut scheduled = Vec::with_capacity(slots.len());
    let elements = order::element_roots(b.function(), terms, &index_of);
    let sequence = order::block_sequence(order, &elements, &operands, slots.len());
    for index in sequence {
        let (output, element, roots) = &elements[index];
        let (output, element) = (*output, *element);
        let value = terms[output][element].as_ref().expect("element has terms");
        block.clear();
        emit_closure(roots, &mut block);
        for index in &block {
            scheduled.push(slots[*index].take().expect("binding placed once"));
        }
        let mut folded: Option<gpu_ir::Value> = None;
        for (k, term) in value.iter().enumerate() {
            let built = term.build(b);
            folded = Some(match folded {
                None => built,
                Some(held) => {
                    let sum = b.add(held, built);
                    let bound = b.bind_value_as(
                        format!("loom_ret_{output}_{element}_{k}"),
                        element_ty(output),
                        sum,
                    );
                    scheduled.push(b.defining_op(&bound).expect("a binding is an operation"));
                    bound
                }
            });
        }
        selected[output][element] = folded;
    }
    // Guard predicates and other non-root consumers may still need bindings no
    // element closure reached.
    scheduled.extend(slots.into_iter().flatten());
    Ok((scheduled, selected))
}

pub(in super::super) fn selected_bindings(selected: &SelectedOutputs) -> Vec<gpu_ir::Value> {
    selected.iter().flatten().filter_map(Clone::clone).collect()
}
