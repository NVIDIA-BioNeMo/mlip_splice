// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opaque maps between finite sets, independent of their storage or execution.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteSetId(pub u32);

impl FiniteSetId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteMapId(pub u32);

impl FiniteMapId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FiniteSet {
    pub extent: i64,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FiniteMap {
    pub source: FiniteSetId,
    pub target: FiniteSetId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FiniteMapGraph {
    pub sets: Vec<FiniteSet>,
    pub maps: Vec<FiniteMap>,
}

impl FiniteMapGraph {
    pub fn verify(&self) -> Result<(), String> {
        for (index, set) in self.sets.iter().enumerate() {
            if set.extent < 1 {
                return Err(format!(
                    "finite set {index} extent must be positive, got {}",
                    set.extent
                ));
            }
        }
        for (index, map) in self.maps.iter().enumerate() {
            self.set(map.source)
                .map_err(|error| format!("finite map {index} source: {error}"))?;
            self.set(map.target)
                .map_err(|error| format!("finite map {index} target: {error}"))?;
        }
        Ok(())
    }

    pub fn add_set(&mut self, extent: i64) -> Result<FiniteSetId, String> {
        if extent < 1 {
            return Err(format!("finite set extent must be positive, got {extent}"));
        }
        let id = FiniteSetId(
            u32::try_from(self.sets.len()).map_err(|_| "too many finite sets".to_string())?,
        );
        self.sets.push(FiniteSet { extent });
        Ok(id)
    }

    pub fn add_map(
        &mut self,
        source: FiniteSetId,
        target: FiniteSetId,
    ) -> Result<FiniteMapId, String> {
        self.set(source)?;
        self.set(target)?;
        let id = FiniteMapId(
            u32::try_from(self.maps.len()).map_err(|_| "too many finite maps".to_string())?,
        );
        self.maps.push(FiniteMap { source, target });
        Ok(id)
    }

    pub fn set(&self, id: FiniteSetId) -> Result<&FiniteSet, String> {
        self.sets
            .get(id.index())
            .ok_or_else(|| format!("finite set {} is out of range", id.index()))
    }

    pub fn map(&self, id: FiniteMapId) -> Result<&FiniteMap, String> {
        self.maps
            .get(id.index())
            .ok_or_else(|| format!("finite map {} is out of range", id.index()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_require_declared_endpoints() {
        let mut graph = FiniteMapGraph::default();
        let source = graph.add_set(8).unwrap();
        let target = graph.add_set(3).unwrap();
        let map = graph.add_map(source, target).unwrap();
        assert_eq!(graph.map(map).unwrap().source, source);
        assert!(graph.add_map(FiniteSetId(9), target).is_err());
    }

    #[test]
    fn public_arena_mutation_is_verified() {
        let graph = FiniteMapGraph {
            sets: vec![FiniteSet { extent: 0 }],
            maps: Vec::new(),
        };
        assert!(graph.verify().is_err());
    }
}
