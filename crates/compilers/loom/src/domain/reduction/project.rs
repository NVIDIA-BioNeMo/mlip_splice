// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The kernels that project a fold's bucket coordinate away.
//!
//! One entry per output takes its bucket count and element-tile width at launch,
//! so the executor can measure projection geometry without another cubin.

use gpu_ir::{
    Builder, Carry, DeviceFunction, Launch, Param, ParamTy, Scalar, SharedArr, Ty, Value,
};

use super::{FINALIZE_BLOCK_SIZE, OutputFold, Projection, ProjectionEntries};
use gpu_ir::Counted;
use gpu_ir::Trip;

const WARP_SIZE: u32 = 32;

fn zero(b: &mut Builder, scalar: Scalar) -> Value {
    b.zero(scalar)
}

/// The flat thread index across the grid, as a 64-bit element index.
fn grid_index(b: &mut Builder) -> Value {
    let block = b.ctaid_x();
    let block = b.cast(Scalar::S64, block);
    let width = b.i64(FINALIZE_BLOCK_SIZE as i64);
    let base = b.mul(block, width);
    let thread = b.tid_x();
    let thread = b.cast(Scalar::S64, thread);
    b.add(base, thread)
}

/// This thread's index within its block.
fn thread_index(b: &mut Builder) -> Value {
    let thread = b.tid_x();
    b.cast(Scalar::S32, thread)
}

impl OutputFold {
    /// Every projection entry this output can reach. An output whose only arm
    /// leaves the result in place needs none.
    #[cfg(test)]
    pub(crate) fn projection_kernels(
        &self,
        entries: &ProjectionEntries,
        elements: usize,
    ) -> Result<Vec<Launch>, String> {
        let elements = i64::try_from(elements)
            .map_err(|_| "PTX fold projection element count exceeds i64".to_string())?;
        let mut kernels = Vec::new();
        if self.projection(1) == Some(Projection::Convert) {
            kernels.push(self.convert_kernel(entries.convert.clone(), elements));
        }
        if self.max_buckets() > 1 {
            kernels.push(self.fold_kernel(entries.fold.clone(), elements));
        }
        Ok(kernels)
    }

    pub(crate) fn projection_kernel(
        &self,
        projection: Projection,
        entries: &ProjectionEntries,
        elements: usize,
    ) -> Result<Launch, String> {
        let elements = i64::try_from(elements)
            .map_err(|_| "PTX fold projection element count exceeds i64".to_string())?;
        Ok(match projection {
            Projection::Convert => self.convert_kernel(entries.convert.clone(), elements),
            Projection::Fold { .. } => self.fold_kernel(entries.fold.clone(), elements),
        })
    }

    fn convert_kernel(&self, entry: String, elements: i64) -> Launch {
        let mut b = Builder::new();
        let [input, output] = self.declare_projection(&mut b, None);
        let flat = grid_index(&mut b);
        let index = b.bind_value_as("fold_index", Ty::scalar(Scalar::S64), flat);
        {
            let end = b.i64(elements);
            let past_end = b.ge(index.clone(), end);
            b.if_(past_end, |guarded| guarded.ret());
        }
        let cell = b.offset(input, index.clone());
        let loaded = b.load(cell, 1, false);
        let narrowed = self.narrow(&mut b, loaded);
        let slot = b.offset(output, index);
        b.store(slot, narrowed);
        Launch::new(
            DeviceFunction::new(entry, b.into_function()),
            FINALIZE_BLOCK_SIZE,
            None,
        )
    }

    /// Sums the bucket copies for an element tile. Tile width 256 is one thread
    /// per element, width 1 is one block per element, and powers between them
    /// share the block between elements and bucket groups.
    fn fold_kernel(&self, entry: String, elements: i64) -> Launch {
        let mut b = Builder::new();
        let [input, output] = self.declare_projection(&mut b, None);
        let buckets = b.parameter(Param {
            name: "fold_buckets".to_owned(),
            ty: ParamTy::Scalar(Scalar::S32),
        });
        let tile_log2 = b.parameter(Param {
            name: "fold_tile_log2".to_owned(),
            ty: ParamTy::Scalar(Scalar::S32),
        });
        let partials = b.shared_array(SharedArr {
            name: "fold_partials".to_owned(),
            elem: self.accumulator,
            len: i64::from(FINALIZE_BLOCK_SIZE),
        });

        let (per_element, ()) = b.build_fragment(|f| {
            self.sum_buckets_per_element(f, &input, &output, &buckets, elements);
            f.ret();
        });
        {
            let wide = b.i32(8);
            let one_thread_per_element = b.eq(tile_log2.clone(), wide);
            let per_element = per_element.into_ops();
            b.if_(one_thread_per_element, |arm| arm.place_all(per_element));
        }
        let (per_block, ()) = b.build_fragment(|f| {
            self.sum_buckets_per_block(f, &input, &output, &buckets, &partials, elements);
            f.ret();
        });
        {
            let narrow = b.i32(0);
            let one_block_per_element = b.eq(tile_log2.clone(), narrow);
            let per_block = per_block.into_ops();
            b.if_(one_block_per_element, |arm| arm.place_all(per_block));
        }
        self.sum_buckets_by_tile(
            &mut b, &input, &output, &buckets, &tile_log2, &partials, elements,
        );

        Launch::new(
            DeviceFunction::new(entry, b.into_function()),
            FINALIZE_BLOCK_SIZE,
            None,
        )
    }

    /// One thread per element, folding every bucket of it.
    fn sum_buckets_per_element(
        &self,
        b: &mut Builder,
        input: &Value,
        output: &Value,
        buckets: &Value,
        elements: i64,
    ) {
        let flat = grid_index(b);
        let element = b.bind_value_as("fold_element", Ty::scalar(Scalar::S64), flat);
        {
            let end = b.i64(elements);
            let past_end = b.ge(element.clone(), end);
            b.if_(past_end, |guarded| guarded.ret());
        }
        let start = b.i32(0);
        let step = b.i32(1);
        let value = self.fold_buckets(b, input, start, buckets.clone(), step, &element, elements);
        let narrowed = self.narrow(b, value);
        let slot = b.offset(output.clone(), element);
        b.store(slot, narrowed);
    }

    /// One block per element: the block's threads stride the buckets, then the
    /// warps combine through shared memory.
    fn sum_buckets_per_block(
        &self,
        b: &mut Builder,
        input: &Value,
        output: &Value,
        buckets: &Value,
        partials: &Value,
        elements: i64,
    ) {
        let block = b.ctaid_x();
        let block = b.cast(Scalar::S64, block);
        let element = b.bind_value_as("fold_element", Ty::scalar(Scalar::S64), block);
        let lane = {
            let thread = thread_index(b);
            let width = b.i32(i64::from(WARP_SIZE));
            let lane = b.rem(thread, width);
            b.bind_value_as("fold_lane", Ty::scalar(Scalar::S32), lane)
        };
        let warp = {
            let thread = thread_index(b);
            let width = b.i32(i64::from(WARP_SIZE));
            let warp = b.div(thread, width);
            b.bind_value_as("fold_warp", Ty::scalar(Scalar::S32), warp)
        };
        let start = thread_index(b);
        let step = b.i32(i64::from(FINALIZE_BLOCK_SIZE));
        let mut value =
            self.fold_buckets(b, input, start, buckets.clone(), step, &element, elements);
        for offset in [16, 8, 4, 2, 1] {
            let delta = b.i32(offset);
            let other = b.shfl_down(value.clone(), delta);
            let sum = b.add(value.clone(), other);
            value = b.update(value, sum);
        }
        {
            let zero = b.i32(0);
            let first = b.eq(lane, zero);
            let (deposit, ()) = b.build_fragment(|f| {
                let slot = f.offset(partials.clone(), warp);
                f.store(slot, value.clone());
            });
            let deposit = deposit.into_ops();
            b.if_(first, |arm| arm.place_all(deposit));
        }
        b.barrier();
        // The block's total lands in the accumulator warp 0 already holds, so
        // the combine writes the register it is going to store from. Nothing
        // reads it after the branch, so the branch produces nothing: the last
        // thing it does is the store.
        let (combine, ()) = b.build_fragment(|f| {
            let mut folded = {
                let at = f.i32(0);
                let cell = f.offset(partials.clone(), at);
                let first = f.load(cell, 1, false);
                f.update(value.clone(), first)
            };
            for slot in 1..FINALIZE_BLOCK_SIZE / WARP_SIZE {
                let at = f.i32(i64::from(slot));
                let cell = f.offset(partials.clone(), at);
                let partial = f.load(cell, 1, false);
                let sum = f.add(folded.clone(), partial);
                folded = f.update(folded, sum);
            }
            let narrowed = self.narrow(f, folded);
            let destination = f.offset(output.clone(), element);
            f.store(destination, narrowed);
        });
        let combine = combine.into_ops();
        let thread = thread_index(b);
        let zero = b.i32(0);
        let leader = b.eq(thread, zero);
        b.if_(leader, |arm| arm.place_all(combine));
    }

    /// The general tile: each block folds `tile` elements, its threads split
    /// between elements and bucket groups, and the groups combine through
    /// shared memory.
    #[allow(clippy::too_many_arguments)]
    fn sum_buckets_by_tile(
        &self,
        b: &mut Builder,
        input: &Value,
        output: &Value,
        buckets: &Value,
        tile_log2: &Value,
        partials: &Value,
        elements: i64,
    ) {
        let shift = b.cast(Scalar::U32, tile_log2.clone());
        let tile = {
            let one = b.i32(1);
            let tile = b.shl(one, shift.clone());
            b.bind_value_as("fold_tile", Ty::scalar(Scalar::S32), tile)
        };
        let element = {
            let block = b.ctaid_x();
            let block = b.cast(Scalar::S64, block);
            let width = b.cast(Scalar::S64, tile.clone());
            let base = b.mul(block, width);
            let thread = thread_index(b);
            let one = b.i32(1);
            let mask = b.sub(tile.clone(), one);
            let within = b.and(thread, mask);
            let within = b.cast(Scalar::S64, within);
            let element = b.add(base, within);
            b.bind_value_as("fold_element", Ty::scalar(Scalar::S64), element)
        };
        let bucket_group = {
            let thread = thread_index(b);
            let group = b.shr(thread, shift.clone());
            b.bind_value_as("fold_bucket_group", Ty::scalar(Scalar::S32), group)
        };
        let bucket_groups = {
            let width = b.i32(i64::from(FINALIZE_BLOCK_SIZE));
            let groups = b.shr(width, shift);
            b.bind_value_as("fold_bucket_groups", Ty::scalar(Scalar::S32), groups)
        };
        // Elements past the end contribute zero; the rest fold their buckets.
        // The loop yields what the `if` produces, so it carries the value in the
        // register the `if` already holds it in.
        let end = b.i64(elements);
        let within = b.lt(element.clone(), end);
        b.open_arm();
        let folded = self.fold_buckets(
            b,
            input,
            bucket_group,
            buckets.clone(),
            bucket_groups,
            &element,
            elements,
        );
        let taken = b.close_arm(vec![folded]);
        b.open_arm();
        let empty = zero(b, self.accumulator);
        let untaken = b.close_arm(vec![empty]);
        let value = b.if_arms(
            within,
            &[(Ty::scalar(self.accumulator), None)],
            taken,
            untaken,
        );
        let value = value.into_iter().next().expect("one result");
        {
            let thread = thread_index(b);
            let slot = b.offset(partials.clone(), thread);
            b.store(slot, value);
        }
        b.barrier();
        for offset in [128, 64, 32, 16, 8, 4, 2, 1] {
            let (step, ()) = b.build_fragment(|f| {
                let (combine, ()) = f.build_fragment(|g| {
                    let thread = thread_index(g);
                    let mine = g.offset(partials.clone(), thread.clone());
                    let held = g.load(mine.clone(), 1, false);
                    let away = g.i32(offset);
                    let other = g.add(thread, away);
                    let cell = g.offset(partials.clone(), other);
                    let partial = g.load(cell, 1, false);
                    let sum = g.add(held, partial);
                    g.store(mine, sum);
                });
                let thread = thread_index(f);
                let away = f.i32(offset);
                let active = f.lt(thread, away);
                let combine = combine.into_ops();
                f.if_(active, |arm| arm.place_all(combine));
                f.barrier();
            });
            let away = b.i32(offset);
            let wider = b.ge(away, tile.clone());
            let step = step.into_ops();
            b.if_(wider, |arm| arm.place_all(step));
        }
        let (store, ()) = b.build_fragment(|f| {
            let (inner, ()) = f.build_fragment(|g| {
                let thread = thread_index(g);
                let cell = g.offset(partials.clone(), thread);
                let total = g.load(cell, 1, false);
                let narrowed = self.narrow(g, total);
                let slot = g.offset(output.clone(), element.clone());
                g.store(slot, narrowed);
            });
            let end = f.i64(elements);
            let within = f.lt(element.clone(), end);
            let inner = inner.into_ops();
            f.if_(within, |arm| arm.place_all(inner));
        });
        let thread = thread_index(b);
        let owns = b.lt(thread, tile);
        let store = store.into_ops();
        b.if_(owns, |arm| arm.place_all(store));
    }

    /// `sum over bucket in start..buckets step step of input[bucket * elements +
    /// element]`, carried in one accumulator.
    #[allow(clippy::too_many_arguments)]
    fn fold_buckets(
        &self,
        b: &mut Builder,
        input: &Value,
        start: Value,
        buckets: Value,
        step: Value,
        element: &Value,
        elements: i64,
    ) -> Value {
        let empty = zero(b, self.accumulator);
        let carried = vec![Carry {
            ty: Ty::scalar(self.accumulator),
            init: empty,
            into: None,
        }];
        let input = input.clone();
        let element = element.clone();
        let summed = b.counted_loop_producing(
            Counted {
                index: Scalar::S32,
                start,
                stop: buckets,
                step,
                trip: Trip::PerThread,
            },
            carried,
            move |f, bucket, given| {
                let bucket = f.cast(Scalar::S64, bucket);
                let stride = f.i64(elements);
                let offset = f.mul(bucket, stride);
                let at = f.add(offset, element.clone());
                let cell = f.offset(input.clone(), at);
                let partial = f.load(cell, 1, false);
                vec![f.add(given[0].clone(), partial)]
            },
        );
        summed.into_iter().next().expect("one carried value")
    }

    /// The buffers a projection entry takes: what it reads and what it writes.
    fn declare_projection(&self, b: &mut Builder, _fold: Option<()>) -> [Value; 2] {
        let input = b.parameter(Param {
            name: "fold_in".to_owned(),
            ty: ParamTy::Ptr(self.accumulator),
        });
        let output = b.parameter(Param {
            name: "fold_out".to_owned(),
            ty: ParamTy::Ptr(self.storage),
        });
        [input, output]
    }

    /// The accumulated value in the output's storage dtype.
    fn narrow(&self, b: &mut Builder, value: Value) -> Value {
        b.cast(self.storage, value)
    }
}
