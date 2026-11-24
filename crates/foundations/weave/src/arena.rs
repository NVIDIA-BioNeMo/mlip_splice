// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_ARENA: AtomicU32 = AtomicU32::new(1);

pub struct Id<T> {
    arena: u32,
    slot: u32,
    marker: PhantomData<fn() -> T>,
}

impl<T> Copy for Id<T> {}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        (self.arena, self.slot) == (other.arena, other.slot)
    }
}
impl<T> Eq for Id<T> {}
impl<T> std::hash::Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        (self.arena, self.slot).hash(state);
    }
}
impl<T> fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}@{}", self.slot, self.arena)
    }
}
impl<T> Id<T> {
    pub fn slot(self) -> u32 {
        self.slot
    }

    pub(crate) fn arena(self) -> u32 {
        self.arena
    }
}

struct Slot<T> {
    value: Option<T>,
}

pub struct Arena<T> {
    identity: u32,
    slots: Vec<Slot<T>>,
    len: usize,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Arena<T> {
    pub fn new() -> Self {
        let identity = NEXT_ARENA
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .expect("arena identities exhausted");
        Self {
            identity,
            slots: Vec::new(),
            len: 0,
        }
    }

    pub fn insert(&mut self, value: T) -> Id<T> {
        let slot = u32::try_from(self.slots.len()).expect("arena capacity exceeds u32");
        self.slots.push(Slot { value: Some(value) });
        self.len += 1;
        Id {
            arena: self.identity,
            slot,
            marker: PhantomData,
        }
    }

    pub fn contains(&self, id: Id<T>) -> bool {
        self.get(id).is_some()
    }

    pub(crate) fn id_at(&self, slot: usize) -> Option<Id<T>> {
        let entry = self.slots.get(slot)?;
        entry.value.as_ref()?;
        Some(Id {
            arena: self.identity,
            slot: slot as u32,
            marker: PhantomData,
        })
    }

    pub fn get(&self, id: Id<T>) -> Option<&T> {
        if id.arena != self.identity {
            return None;
        }
        let slot = self.slots.get(id.slot as usize)?;
        slot.value.as_ref()
    }

    pub fn get_mut(&mut self, id: Id<T>) -> Option<&mut T> {
        if id.arena != self.identity {
            return None;
        }
        let slot = self.slots.get_mut(id.slot as usize)?;
        slot.value.as_mut()
    }

    pub fn remove(&mut self, id: Id<T>) -> Option<T> {
        if id.arena != self.identity {
            return None;
        }
        let entry = self.slots.get_mut(id.slot as usize)?;
        let value = entry.value.take()?;
        self.len -= 1;
        Some(value)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Id<T>, &T)> {
        let identity = self.identity;
        self.slots
            .iter()
            .enumerate()
            .filter_map(move |(slot, entry)| {
                entry.value.as_ref().map(|value| {
                    (
                        Id {
                            arena: identity,
                            slot: slot as u32,
                            marker: PhantomData,
                        },
                        value,
                    )
                })
            })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn into_values(self) -> impl Iterator<Item = (Id<T>, T)> {
        let identity = self.identity;
        self.slots
            .into_iter()
            .enumerate()
            .filter_map(move |(slot, entry)| {
                entry.value.map(|value| {
                    (
                        Id {
                            arena: identity,
                            slot: slot as u32,
                            marker: PhantomData,
                        },
                        value,
                    )
                })
            })
    }

    pub(crate) fn identity(&self) -> u32 {
        self.identity
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn ids_are_two_words_of_u32() {
        assert_eq!(std::mem::size_of::<super::Id<u8>>(), 8);
    }

    #[test]
    fn rejects_stale_and_foreign_ids() {
        let mut first = Arena::new();
        let mut second = Arena::new();
        let old = first.insert(1);
        assert_eq!(first.remove(old), Some(1));
        let replacement = first.insert(2);
        let foreign = second.insert(3);
        assert!(replacement.slot() > old.slot());
        assert!(first.get(old).is_none());
        assert!(first.get(foreign).is_none());
    }
}
