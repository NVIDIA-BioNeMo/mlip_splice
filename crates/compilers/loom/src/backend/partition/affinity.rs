// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(super) trait AffinityCluster: Clone {
    fn fits_with(&self, other: &Self, cap: usize) -> bool;
    fn merge_gain(&self, other: &Self, settings: &super::Settings) -> usize;
    fn absorb(&mut self, other: Self);
}

pub(super) fn agglomerate<T: AffinityCluster>(
    mut tasks: Vec<T>,
    cap: usize,
    settings: &super::Settings,
) -> Vec<T> {
    if tasks.len() > settings.max_affinity_tasks {
        let mut packed: Vec<T> = Vec::new();
        for task in tasks {
            if packed.last().is_some_and(|last| last.fits_with(&task, cap)) {
                packed.last_mut().expect("nonempty pack").absorb(task);
            } else {
                packed.push(task);
            }
        }
        return packed;
    }

    let mut alive = vec![true; tasks.len()];
    let mut gains = vec![vec![(0usize, false); tasks.len()]; tasks.len()];
    let score = |a: &T, b: &T| (a.merge_gain(b, settings), a.fits_with(b, cap));
    for i in 0..tasks.len() {
        for j in i + 1..tasks.len() {
            gains[i][j] = score(&tasks[i], &tasks[j]);
        }
    }
    loop {
        let mut best: Option<(usize, usize, usize)> = None;
        for i in 0..tasks.len() {
            if !alive[i] {
                continue;
            }
            for j in i + 1..tasks.len() {
                let (gain, fits) = gains[i][j];
                if !alive[j] || !fits {
                    continue;
                }
                if best.is_none_or(|(_, _, best_gain)| gain > best_gain) {
                    best = Some((i, j, gain));
                }
            }
        }
        let Some((i, j, _)) = best else { break };
        let absorbed = tasks[j].clone();
        tasks[i].absorb(absorbed);
        alive[j] = false;
        for (k, &is_alive) in alive.iter().enumerate() {
            if k == i || !is_alive {
                continue;
            }
            let (low, high) = (k.min(i), k.max(i));
            gains[low][high] = score(&tasks[low], &tasks[high]);
        }
    }
    tasks
        .into_iter()
        .zip(alive)
        .filter_map(|(task, alive)| alive.then_some(task))
        .collect()
}
