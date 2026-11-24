// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What a set of operations reaches, and whether it reaches back.

#[cfg(test)]
mod tests;

/// The operations reachable from `from`, the starting ones excluded unless
/// something reaches them.
///
/// `successors` states the relation walked: it calls its argument with each
/// operation the given one leads to. Which relation that is decides which
/// question this answers — what a set reads through its operands, what the
/// bodies of its regions evaluate as well, what reads *it* — and the walk
/// itself is the same either way.
///
/// Membership is a bit per operation rather than a set, since an operation is
/// a dense index: the whole walk is one allocation, and each operation is
/// visited once however many others lead to it.
pub fn reachable(
    operation_count: usize,
    from: impl IntoIterator<Item = usize>,
    successors: impl FnMut(usize, &mut dyn FnMut(usize)),
) -> Reached {
    let mut reached = Reached::default();
    reached.walk(operation_count, from, successors);
    reached
}

/// Which operations a walk reached.
///
/// Membership is a bit per operation rather than a set, since an operation is
/// a dense index: the whole walk is one allocation, and each operation is
/// visited once however many others lead to it.
///
/// A caller walking repeatedly reuses one of these through [`Reached::walk`].
/// Allocating per walk costs the size of the arena each time, which is what
/// dominates when the walks are many and each is small.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Reached {
    seen: Vec<bool>,
    /// The operations marked in `seen`, so that clearing costs what was
    /// reached rather than the size of the arena.
    reached: Vec<usize>,
    /// Kept between walks so that its capacity is not paid for again.
    walk: Vec<usize>,
}

impl Reached {
    /// Walks again from `from`, forgetting what a previous walk reached.
    ///
    /// `successors` states the relation walked: it calls its argument with
    /// each operation the given one leads to. Which relation that is decides
    /// which question this answers — what a set reads through its operands,
    /// what the bodies of its regions evaluate as well, what reads *it* — and
    /// the walk itself is the same either way.
    pub fn walk(
        &mut self,
        operation_count: usize,
        from: impl IntoIterator<Item = usize>,
        mut successors: impl FnMut(usize, &mut dyn FnMut(usize)),
    ) {
        for &operation in &self.reached {
            self.seen[operation] = false;
        }
        self.reached.clear();
        if self.seen.len() < operation_count {
            self.seen.resize(operation_count, false);
        }
        self.walk.clear();
        self.walk.extend(from);
        while let Some(operation) = self.walk.pop() {
            let seen = &mut self.seen;
            let found = &mut self.reached;
            let walk = &mut self.walk;
            successors(operation, &mut |next| {
                if !seen[next] {
                    seen[next] = true;
                    found.push(next);
                    walk.push(next);
                }
            });
        }
    }

    pub fn contains(&self, operation: usize) -> bool {
        self.seen.get(operation).copied().unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.reached.len()
    }

    pub fn is_empty(&self) -> bool {
        self.reached.is_empty()
    }

    /// The operations reached, in arena order.
    pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        let mut order = self.reached.clone();
        order.sort_unstable();
        order.into_iter()
    }
}

/// The first operation found to lead back to itself, and `None` when the
/// relation is acyclic.
///
/// A value is readable after the operation defining it, so a program's use-def
/// relation has no cycle; one found means the arena stopped describing a
/// program. What an operation leads to is asked once, when it is pushed:
/// asking per edge re-walks its successors for each one it has, which is a
/// square in the degree, and a region-owning operation's degree is the size of
/// its body.
pub fn first_cycle(
    operation_count: usize,
    roots: impl IntoIterator<Item = usize>,
    mut successors: impl FnMut(usize, &mut dyn FnMut(usize)),
) -> Option<usize> {
    // Unvisited, on the stack, or finished. An operation on the stack that is
    // reached again is reached from within its own descent, which is what a
    // cycle is.
    const UNVISITED: u8 = 0;
    const DESCENDING: u8 = 1;
    const FINISHED: u8 = 2;
    let mut state = vec![UNVISITED; operation_count];
    let mut gathered = Vec::new();
    for root in roots {
        if state[root] != UNVISITED {
            continue;
        }
        state[root] = DESCENDING;
        gathered.clear();
        successors(root, &mut |next| gathered.push(next));
        let mut stack = vec![(root, 0usize, std::mem::take(&mut gathered))];
        while let Some((operation, next, leads_to)) = stack.last_mut() {
            if *next == leads_to.len() {
                state[*operation] = FINISHED;
                stack.pop();
                continue;
            }
            let successor = leads_to[*next];
            *next += 1;
            match state[successor] {
                UNVISITED => {
                    state[successor] = DESCENDING;
                    let mut leads_to = Vec::new();
                    successors(successor, &mut |next| leads_to.push(next));
                    stack.push((successor, 0, leads_to));
                }
                DESCENDING => return Some(successor),
                _ => {}
            }
        }
    }
    None
}
