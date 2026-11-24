// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use domain_index::{AddressSource, DomainIndex, IndexBuilder, Slot, SlotKind};
use gpu_ir::{Param, ParamTy, Scalar, Value};

pub(crate) mod reduction;

#[cfg(test)]
mod tests;

/// Where a parameter sits in [`Layout::params`], which is the only place its
/// name lives.
///
/// The index stays inside this module: a slot is read by asking the layout
/// what it is, so nothing outside can index `params` by a number it made up or
/// carry a name that has drifted from the one the kernel declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ParamSlot(usize);

/// The arrays one index map contributes, each as the parameter that carries
/// it. A map walked through its preimage takes the preimage offsets and
/// possibly the source indices those offsets order; one addressed directly
/// takes the target indices.
#[derive(Clone)]
pub(crate) struct RefParams {
    pub target_indices: Option<ParamSlot>,
    pub preimage_offsets: Option<ParamSlot>,
    pub preimage_source_indices: Option<ParamSlot>,
}

/// The `s32` parameter carrying `buckets - 1` for one output whose fold offers
/// more than one bucket count: masking the block index with it picks the copy
/// of the keyspace this block folds into, so one kernel serves every count.
#[derive(Clone)]
pub(crate) struct BucketMask {
    pub output: usize,
    pub param: Param,
}

pub(crate) struct Layout {
    /// Every layout parameter, in the order a launch binds it, and the only
    /// owner of each one's name.
    pub params: Vec<Param>,
    pub operand_params: Vec<ParamSlot>,
    pub used_index_maps: Vec<usize>,
    pub used_views: Vec<usize>,
    pub ref_params: Vec<RefParams>,
    pub view_params: Vec<ParamSlot>,
    /// Trailing parameters, past `params`, that only the kernels writing
    /// through the fold take. Producers leave them off and their launches
    /// bind none.
    pub bucket_masks: Vec<BucketMask>,
}

/// Bind an index array — every DomainIndex bridge array is `s32` — as the next
/// kernel parameter, and hand back where it sits.
fn index_array(params: &mut Vec<Param>, name: String) -> ParamSlot {
    params.push(Param {
        name,
        ty: ParamTy::Ptr(Scalar::S32),
    });
    ParamSlot(params.len() - 1)
}

/// Bind an operand's buffer as the next kernel parameter.
fn operand_array(params: &mut Vec<Param>, name: String, elem: Scalar) -> ParamSlot {
    params.push(Param {
        name,
        ty: ParamTy::Ptr(elem),
    });
    ParamSlot(params.len() - 1)
}

impl Layout {
    /// Index arrays follow `slots` exactly; the graph resolves their map and
    /// view handles, not their membership, array kinds, or order.
    ///
    /// `bucket_selectors` names the outputs whose fold picks its bucket count
    /// at launch; each takes one trailing scalar parameter.
    pub(crate) fn new<I: DomainIndex>(
        index: &I,
        slots: &[Slot],
        num_inputs: usize,
        operand_scalars: &[Scalar],
        bucket_selectors: &[usize],
    ) -> Result<Self, String> {
        if num_inputs > index.num_operands() {
            return Err(format!(
                "num_inputs={num_inputs} exceeds num_operands={}",
                index.num_operands()
            ));
        }
        if operand_scalars.len() != index.num_operands() {
            return Err(format!(
                "got {} operand scalars for {} operands",
                operand_scalars.len(),
                index.num_operands()
            ));
        }

        let mut used_index_maps = Vec::new();
        let mut used_views = Vec::new();
        let in_pfx = "in_";
        let out_pfx = "out_";
        let target_indices_pfx = "target_indices_";
        let preimage_offsets_pfx = "preimage_offsets_";
        let preimage_source_indices_pfx = "preimage_source_indices_";
        let view_pfx = "aligned_target_indices_";
        let mut params = Vec::new();
        let mut operand_params = vec![None; index.num_operands()];

        for (b, slot) in operand_params.iter_mut().enumerate().take(num_inputs) {
            *slot = Some(operand_array(
                &mut params,
                format!("{in_pfx}{b}"),
                operand_scalars[b],
            ));
        }

        let mut ref_params = Vec::new();
        let mut view_params = Vec::new();
        for slot in slots {
            let map_slot = ref_params.len();
            let mut refs = RefParams {
                target_indices: None,
                preimage_offsets: None,
                preimage_source_indices: None,
            };
            match slot.kind {
                SlotKind::TargetIndices | SlotKind::ComposedTargetIndices => {
                    refs.target_indices = Some(index_array(
                        &mut params,
                        format!("{target_indices_pfx}{map_slot}"),
                    ));
                }
                SlotKind::Preimage | SlotKind::PreimageSorted => {
                    refs.preimage_offsets = Some(index_array(
                        &mut params,
                        format!("{preimage_offsets_pfx}{map_slot}"),
                    ));
                    if slot.kind == SlotKind::Preimage {
                        refs.preimage_source_indices = Some(index_array(
                            &mut params,
                            format!("{preimage_source_indices_pfx}{map_slot}"),
                        ));
                    }
                }
                SlotKind::AlignedTargetIndices => {
                    let view_idx = index
                        .graph()
                        .aligned_views
                        .iter()
                        .position(|view| {
                            view.source_id == slot.source_id
                                && view.anchor_target_id == slot.target_id
                                && view.other_target_id == slot.other_target_id
                        })
                        .ok_or_else(|| format!("slot {slot:?} has no aligned view in the graph"))?;
                    if used_views.contains(&view_idx) {
                        return Err(format!("aligned view {view_idx} has multiple slots"));
                    }
                    view_params.push(index_array(
                        &mut params,
                        format!("{view_pfx}{}", view_params.len()),
                    ));
                    used_views.push(view_idx);
                    continue;
                }
            }
            let map_idx = index
                .graph()
                .index_maps
                .iter()
                .position(|map| map.source_id == slot.source_id && map.target_id == slot.target_id)
                .ok_or_else(|| format!("slot {slot:?} has no index map in the graph"))?;
            if used_index_maps.contains(&map_idx) {
                return Err(format!("index map {map_idx} has multiple slots"));
            }
            used_index_maps.push(map_idx);
            ref_params.push(refs);
        }

        for (b, slot) in operand_params.iter_mut().enumerate().skip(num_inputs) {
            *slot = Some(operand_array(
                &mut params,
                format!("{out_pfx}{}", b - num_inputs),
                operand_scalars[b],
            ));
        }
        // The two passes above partition the operands, so every one has a
        // parameter by now.
        let operand_params = operand_params
            .into_iter()
            .map(|slot| slot.expect("every operand is either an input or an output"))
            .collect::<Vec<_>>();

        let mask_pfx = "bmask_";
        let bucket_masks = bucket_selectors
            .iter()
            .map(|&output| BucketMask {
                output,
                param: Param {
                    name: format!("{mask_pfx}{output}"),
                    ty: ParamTy::Scalar(Scalar::S32),
                },
            })
            .collect();

        Ok(Layout {
            params,
            operand_params,
            used_index_maps,
            used_views,
            ref_params,
            view_params,
            bucket_masks,
        })
    }

    /// Where `output`'s mask sits in [`Layout::bucket_masks`], when its fold
    /// picks its bucket count at launch.
    pub(crate) fn bucket_mask_slot(&self, output: usize) -> Option<BucketMaskSlot> {
        self.bucket_masks
            .iter()
            .position(|mask| mask.output == output)
            .map(BucketMaskSlot)
    }

    pub(crate) fn ref_params(&self, map_idx: usize) -> &RefParams {
        let slot = self
            .used_index_maps
            .iter()
            .position(|&candidate| candidate == map_idx)
            .expect("index map belongs to the PTX layout");
        &self.ref_params[slot]
    }

    pub(crate) fn view_param(&self, view_idx: usize) -> ParamSlot {
        let slot = self
            .used_views
            .iter()
            .position(|&candidate| candidate == view_idx)
            .expect("view belongs to the PTX layout");
        self.view_params[slot]
    }

    /// What the kernel calls the parameter in `slot`.
    #[cfg(test)]
    pub(crate) fn param_name(&self, slot: ParamSlot) -> &str {
        &self.params[slot.0].name
    }
}

/// Where a mask sits in [`Layout::bucket_masks`], which is a sequence of its
/// own: a mask trails `params` and is not a layout parameter, so it is not a
/// [`ParamSlot`] and never disturbs the prefix a kernel declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BucketMaskSlot(usize);

/// The handles a kernel bound for the bucket masks it declared, in the order
/// [`Layout::bucket_masks`] holds them. A kernel that writes its own partials
/// rather than through the fold declares none.
pub(crate) struct BucketMaskValues(Vec<Value>);

impl BucketMaskValues {
    /// Declares every mask the layout carries, in order.
    pub(crate) fn declare(body: &mut gpu_ir::Builder, layout: &Layout) -> BucketMaskValues {
        BucketMaskValues(
            layout
                .bucket_masks
                .iter()
                .map(|mask| body.parameter(mask.param.clone()))
                .collect(),
        )
    }

    /// No mask at all, for a kernel that does not write through the fold.
    pub(crate) fn none() -> BucketMaskValues {
        BucketMaskValues(Vec::new())
    }

    pub(crate) fn get(&self, slot: BucketMaskSlot) -> Value {
        self.0
            .get(slot.0)
            .unwrap_or_else(|| {
                panic!(
                    "bucket mask {} is past the {} this kernel declared",
                    slot.0,
                    self.0.len()
                )
            })
            .clone()
    }
}

/// The handles a kernel bound for the layout parameters it declared.
///
/// Every flavour declares a prefix of [`Layout::params`] — outputs go last, so
/// a producer's shorter list is still a prefix — and whatever else it takes is
/// appended past that and is not a layout parameter. So the length says which
/// slots this kernel has, and a slot beyond it is one this kernel never
/// declared.
pub(crate) struct ParameterValues(Vec<Value>);

impl ParameterValues {
    /// Declares `layout.params[..count]` on `body`, in order.
    pub(crate) fn declare(
        body: &mut gpu_ir::Builder,
        layout: &Layout,
        count: usize,
    ) -> ParameterValues {
        ParameterValues(
            layout.params[..count]
                .iter()
                .cloned()
                .map(|param| body.parameter(param))
                .collect(),
        )
    }

    pub(crate) fn get(&self, slot: ParamSlot) -> Value {
        self.0
            .get(slot.0)
            .unwrap_or_else(|| {
                panic!(
                    "parameter slot {} is past the {} this kernel declared",
                    slot.0,
                    self.0.len()
                )
            })
            .clone()
    }
}

/// The domain-index algebra as values of the body the address is built into.
///
/// It borrows the builder because a `gpu_ir` value is a handle into one, unlike
/// a freestanding C++ expression. Every operation is arithmetic over handles the
/// builder owns, and none of them is a statement: an address stays an expression
/// until something binds it.
pub(crate) struct GpuIndex<'a>(pub &'a mut gpu_ir::Builder);

impl IndexBuilder for GpuIndex<'_> {
    type Value = Value;

    fn lit(&mut self, v: i64) -> Value {
        self.0.i64(v)
    }

    fn gather(&mut self, array: Value, idx: Value) -> Value {
        let cell = self.0.offset(array, idx);
        let loaded = self.0.load(cell, 1, false);
        self.0.cast(Scalar::S64, loaded)
    }

    fn max0(&mut self, value: Value) -> Value {
        let zero = self.0.i64(0);
        self.0.max(value, zero)
    }

    fn to_i64(&mut self, value: Value) -> Value {
        self.0.cast(Scalar::S64, value)
    }

    fn mul_lit(&mut self, value: Value, k: i64) -> Value {
        let k = self.0.i64(k);
        self.0.mul(value, k)
    }

    fn add(&mut self, a: Value, b: Value) -> Value {
        self.0.add(a, b)
    }

    fn nonneg(&mut self, array: Value, idx: Value) -> Value {
        let cell = self.0.offset(array, idx);
        let loaded = self.0.load(cell, 1, false);
        let zero = self.0.i32(0);
        self.0.ge(loaded, zero)
    }

    fn nonneg_value(&mut self, value: Value) -> Value {
        let zero = self.0.i64(0);
        self.0.ge(value, zero)
    }

    fn and(&mut self, a: Value, b: Value) -> Value {
        self.0.and(a, b)
    }
}

pub(crate) struct EmitCtx<'a, I> {
    pub index: &'a I,
    pub layout: &'a Layout,
    /// The layout parameters the kernel declared, which is where every address
    /// starts. Held beside the builder rather than in it, so reading a handle
    /// and building with it do not borrow the same thing.
    pub declared: &'a ParameterValues,
    /// The bucket masks the kernel declared, which only a write through the
    /// fold reaches.
    pub bucket_masks: &'a BucketMaskValues,
    pub axis_indices: &'a [Value],
    /// Per view slot (parallel to `layout.view_params`): the fiber
    /// iteration's `view[_k]`, pre-bound by the emitter. Empty outside the
    /// fiber loop.
    pub view_elements: &'a [Value],
    /// The index the fiber loop is on, when this context is inside one.
    pub fiber_index: Option<Value>,
}

impl<'g, I: DomainIndex> AddressSource<GpuIndex<'g>> for EmitCtx<'_, I> {
    type Index = I;

    fn index(&self) -> &I {
        self.index
    }

    fn axis_index(&self, _b: &mut GpuIndex<'g>, axis: usize) -> Value {
        self.axis_indices[axis].clone()
    }

    fn target_indices_array(&self, _b: &mut GpuIndex<'g>, map_idx: usize) -> Value {
        self.declared.get(
            self.layout
                .ref_params(map_idx)
                .target_indices
                .expect("target-indices slot present"),
        )
    }

    fn aligned_view_array(&self, _b: &mut GpuIndex<'g>, view_idx: usize) -> Value {
        self.declared.get(self.layout.view_param(view_idx))
    }

    fn view_element(&self, _b: &mut GpuIndex<'g>, view_idx: usize) -> Option<Value> {
        let slot = self
            .layout
            .used_views
            .iter()
            .position(|&candidate| candidate == view_idx)?;
        self.view_elements.get(slot).cloned()
    }

    fn fiber_index(&self) -> Value {
        self.fiber_index
            .clone()
            .expect("a view is bridged only inside the fiber loop")
    }
}
