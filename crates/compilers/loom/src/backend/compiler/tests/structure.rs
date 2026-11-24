// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use super::*;

// The structural baseline: what every definition reads, and where every
// partition boundary falls.
//
// These are the two facts a representation change can move without moving a
// recipe digest, so they are recorded rather than inferred. The tests here say
// what the record must be worth: it has to be non-empty for a program that has
// structure, it has to cover that structure exactly, it has to move when the
// cap moves, and two compiles of one program have to record the same thing.

fn structure_options(statement_cap: Option<usize>) -> CompileOptions {
    CompileOptions {
        max_kernel_statements: statement_cap.and_then(std::num::NonZeroUsize::new),
        record_dependencies_and_partitions: true,
        ..test_options(None)
    }
}

fn structure_of(loom: LiftedProgram, statement_cap: Option<usize>) -> CompileProfile {
    compile_profiled(loom, structure_options(statement_cap))
        .expect("the fixture compiles")
        .1
}

/// A program deep enough to hold shared bindings and wide enough to partition.
fn structured_program() -> LiftedProgram {
    mixed_output_domains_program_with_depth(4)
}

/// The baseline costs a second symbolic expansion per graph, so a compile that
/// did not ask for it records nothing at all.
#[test]
fn nothing_is_recorded_unless_the_compile_asked_for_it() {
    let (_, profile) = compile_profiled(structured_program(), test_options(None)).unwrap();
    assert!(profile.binding_dependencies.is_empty());
    assert!(profile.partition_parts.is_empty());
}

/// Every binding of the evaluation profile is recorded, in binding order, and
/// at least one of them reads another — a record of nothing but empty sets
/// would pass a same-shape comparison against any lowering at all.
#[test]
fn every_definition_states_what_it_reads() {
    let profile = structure_of(structured_program(), None);
    assert!(
        !profile.binding_dependencies.is_empty(),
        "no definition was recorded"
    );

    let mut per_graph: BTreeMap<(usize, &str), Vec<&crate::profile::BindingDependencies>> =
        BTreeMap::new();
    for record in &profile.binding_dependencies {
        per_graph
            .entry((record.graph, record.program.as_str()))
            .or_default()
            .push(record);
    }
    for (graph, records) in &per_graph {
        let positions = records.iter().map(|record| record.binding);
        assert!(
            positions.eq(0..records.len()),
            "{graph:?} skips a binding position"
        );
        for record in records {
            let mut ascending = record.dependencies.clone();
            ascending.sort_unstable();
            ascending.dedup();
            assert_eq!(
                ascending, record.dependencies,
                "{graph:?} definition {} repeats or reorders an entry",
                record.binding
            );
            // A binding reads bindings ordered before it and interned inputs,
            // so no entry it reads can be its own position.
            for &entry in &record.dependencies {
                assert!(
                    entry != record.binding,
                    "{graph:?} definition {} reads itself",
                    record.binding
                );
            }
        }
        assert!(
            records.iter().any(|record| !record.dependencies.is_empty()),
            "{graph:?} recorded no dependency at all"
        );
    }
}

/// A boundary is a partition: within one arm, every element slice a part writes
/// belongs to exactly one part, and every arm of one program writes the same
/// elements — the cap decides how the work is cut up, never which work exists.
#[test]
fn each_arms_boundaries_partition_its_programs_elements() {
    let profile = structure_of(structured_program(), Some(64));
    assert!(
        !profile.partition_parts.is_empty(),
        "no boundary was recorded"
    );

    let mut per_arm: BTreeMap<_, Vec<_>> = BTreeMap::new();
    for record in &profile.partition_parts {
        per_arm
            .entry((
                record.graph,
                record.program.clone(),
                record.order,
                record.statement_cap,
                record.plan_clusters,
                record.placement.clone(),
            ))
            .or_default()
            .push(record);
    }
    let mut covered_per_program: BTreeMap<&String, BTreeSet<BTreeSet<(usize, usize)>>> =
        BTreeMap::new();
    for (arm, records) in &per_arm {
        let positions = records.iter().map(|record| record.part);
        assert!(positions.eq(0..records.len()), "{arm:?} skips a part");
        let mut seen = BTreeSet::new();
        for record in records {
            for (output, element, terms) in &record.slices {
                for &term in terms {
                    assert!(
                        seen.insert((*output, *element, term)),
                        "{arm:?} writes output {output} element {element} term {term} twice"
                    );
                }
            }
        }
        assert!(!seen.is_empty(), "{arm:?} writes nothing");
        covered_per_program.entry(&arm.1).or_default().insert(
            seen.iter()
                .map(|&(output, element, _)| (output, element))
                .collect(),
        );
    }
    assert!(!covered_per_program.is_empty());
    for (program, covered) in covered_per_program {
        assert_eq!(
            covered.len(),
            1,
            "{program}'s arms disagree about what is written"
        );
    }
}

/// The cap is a candidate axis, so one default is not the grid: at a tighter
/// cap some arm is cut into more parts than the same arm at a looser one.
#[test]
fn a_tight_cap_cuts_an_arm_a_loose_one_leaves_whole() {
    let parts_per_arm = |cap: usize| {
        let mut counts: BTreeMap<(String, &'static str, String), usize> = BTreeMap::new();
        for record in
            structure_of(mixed_output_domains_program_with_depth(8), Some(cap)).partition_parts
        {
            *counts
                .entry((record.program, record.order, record.placement))
                .or_default() += 1;
        }
        counts
    };
    let loose = parts_per_arm(48);
    let tight = parts_per_arm(24);
    assert!(!loose.is_empty());
    assert!(
        loose.keys().eq(tight.keys()),
        "the cap changed which arms exist, not only how they are cut"
    );
    assert!(
        tight.iter().any(|(arm, count)| *count > loose[arm]),
        "no arm is cut more finely at the tighter cap: {loose:?} against {tight:?}"
    );
}

/// A part index map had to halve says so, and the halves it produced are the
/// halves of what it was given — the bisection point is in the record, not only
/// the count.
#[test]
fn a_bisected_part_records_the_halving_that_produced_it() {
    let profile = structure_of(mixed_output_domains_program_with_depth(8), Some(24));
    let bisected = profile
        .partition_parts
        .iter()
        .filter(|record| record.bisections > 0)
        .collect::<Vec<_>>();
    assert!(
        !bisected.is_empty(),
        "no part was bisected at a cap this tight"
    );
    for record in bisected {
        assert!(
            record.slices.iter().any(|(_, _, terms)| !terms.is_empty()),
            "a bisected part writes nothing"
        );
    }
}

/// Every program whose boundaries were recorded also has its dependency sets
/// recorded.
///
/// This is the pairing the baseline rests on: a boundary is a cut through a
/// program's bindings, so a boundary whose bindings were never described cannot
/// be compared against a later representation at all. An output-family split is
/// a program in its own right, planned separately, and it is the case the
/// pairing is easiest to lose — so the fixture is asserted to produce one.
#[test]
fn every_program_with_boundaries_also_states_what_its_definitions_read() {
    let profile = structure_of(structured_program(), Some(64));

    let partitioned = profile
        .partition_parts
        .iter()
        .map(|record| (record.graph, record.program.as_str()))
        .collect::<BTreeSet<_>>();
    let described = profile
        .binding_dependencies
        .iter()
        .map(|record| (record.graph, record.program.as_str()))
        .collect::<BTreeSet<_>>();
    assert!(!partitioned.is_empty(), "no boundary was recorded");
    assert_eq!(
        partitioned, described,
        "the programs with boundaries and the programs with dependency sets differ"
    );

    // The root program and its output-family splits are different programs, so
    // the pairing above is over several of them rather than one.
    let families = partitioned
        .iter()
        .filter(|(_, program)| *program != "mixed_domains")
        .count();
    assert!(
        families > 0,
        "the fixture planned no output-family split: {partitioned:?}"
    );
}

/// A program's dependency sets are recorded once, however many arms plan it:
/// they are a property of the program, and the cap × order × placement cross
/// re-plans it many times.
#[test]
fn a_programs_definitions_are_recorded_once_however_many_arms_plan_it() {
    let profile = structure_of(structured_program(), Some(64));

    let arms_per_program = profile.partition_parts.iter().fold(
        BTreeMap::<&str, BTreeSet<(&str, usize, String)>>::new(),
        |mut arms, record| {
            arms.entry(record.program.as_str()).or_default().insert((
                record.order,
                record.statement_cap,
                record.placement.clone(),
            ));
            arms
        },
    );
    assert!(
        arms_per_program.values().any(|arms| arms.len() > 1),
        "no program was planned by more than one arm, so nothing is deduplicated"
    );
    for (program, arms) in &arms_per_program {
        let positions = profile
            .binding_dependencies
            .iter()
            .filter(|record| record.program == *program)
            .map(|record| record.binding)
            .collect::<Vec<_>>();
        assert!(
            positions.iter().copied().eq(0..positions.len()),
            "{program} is described {} times over by its {} arms",
            positions.len(),
            arms.len()
        );
    }
}

/// The record is a baseline, so the same program compiled twice has to produce
/// the same one — otherwise a comparison across a change reads its own noise.
#[test]
fn the_same_program_records_the_same_structure_twice() {
    let first = structure_of(structured_program(), Some(64));
    let second = structure_of(structured_program(), Some(64));

    assert_eq!(first.binding_dependencies, second.binding_dependencies);
    assert_eq!(first.partition_parts, second.partition_parts);
}
