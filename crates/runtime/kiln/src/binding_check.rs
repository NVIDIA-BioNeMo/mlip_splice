// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Whether the buffers a caller bound are the ones a recipe states it needs.
//!
//! A recipe's contract states the byte extent of every port, in binding order —
//! whatever else it may state above that. A launch is handed one pointer per port and nothing else, so
//! the kernels address those pointers by the extents the recipe was built from.
//! This is the one place those two statements are compared, and it runs at
//! every boundary a caller reaches — before realization, before tuning, and
//! before a resident replay — because each of them launches kernels.
//!
//! Without it, a buffer smaller than its port is an out-of-bounds device write
//! with no diagnostic: the addresses are formed from the recipe, not from the
//! buffer, so nothing on the device knows the buffer ended.

use recipe::algebra::BytePortContract;

/// Which side of the contract a mismatch is on. A port index means nothing
/// without it: input 0 and output 0 are different buffers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Input,
    Output,
}

impl Side {
    fn name(self) -> &'static str {
        match self {
            Side::Input => "input",
            Side::Output => "output",
        }
    }
}

/// What a caller got wrong about the buffers it bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BindingError {
    /// A different number of buffers than the recipe has ports. Too few leaves
    /// a port unbound; too many means the caller is holding a different
    /// program than the one it loaded.
    Count {
        side: Side,
        bound: usize,
        stated: usize,
    },
    /// A buffer the kernels would address past the end of.
    Undersized {
        side: Side,
        port: usize,
        bound: usize,
        stated: usize,
    },
}

impl std::fmt::Display for BindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BindingError::Count {
                side,
                bound,
                stated,
            } => write!(
                f,
                "recipe states {stated} {} buffer(s), {bound} bound",
                side.name()
            ),
            BindingError::Undersized {
                side,
                port,
                bound,
                stated,
            } => write!(
                f,
                "{} {port} is {bound} bytes, recipe addresses {stated}",
                side.name()
            ),
        }
    }
}

impl std::error::Error for BindingError {}

/// Compares what was bound against the byte extents `contract` states.
///
/// The contract arrives as the execution view of itself and nothing more: names,
/// element types and roles are the producer's business, and a launch is handed
/// one pointer per port whatever they are.
///
/// Counts must match exactly. Extents must be at least what the recipe states:
/// a larger buffer is accepted because a caller may legitimately bind a view
/// into a bigger allocation — a slice of a torch storage, an XLA buffer padded
/// by its layout — and the kernels address only the prefix the recipe named.
/// A smaller one is refused, since that prefix runs off the end.
pub fn check_contract(
    contract: &impl BytePortContract,
    inputs: &[usize],
    outputs: &[usize],
) -> Result<(), BindingError> {
    check_ports(
        Side::Input,
        contract.num_inputs(),
        &|port| contract.input_nbytes(port),
        inputs,
    )?;
    check_ports(
        Side::Output,
        contract.num_outputs(),
        &|port| contract.output_nbytes(port),
        outputs,
    )
}

fn check_ports(
    side: Side,
    ports: usize,
    nbytes: &dyn Fn(usize) -> Option<usize>,
    bound: &[usize],
) -> Result<(), BindingError> {
    let stated: Vec<usize> = (0..ports).map(|port| nbytes(port).unwrap_or(0)).collect();
    check_side(side, &stated, bound)
}

fn check_side(side: Side, stated: &[usize], bound: &[usize]) -> Result<(), BindingError> {
    if bound.len() != stated.len() {
        return Err(BindingError::Count {
            side,
            bound: bound.len(),
            stated: stated.len(),
        });
    }
    for (port, (&bound, &stated)) in bound.iter().zip(stated).enumerate() {
        if bound < stated {
            return Err(BindingError::Undersized {
                side,
                port,
                bound,
                stated,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use recipe::RecipeSignature;

    /// Two inputs of different sizes and one output, so a swap of the sides or
    /// of two ports lands on a different number.
    fn signature() -> RecipeSignature {
        RecipeSignature::new(vec![48, 96], vec![16])
    }

    #[test]
    fn buffers_that_match_the_ports_are_accepted() {
        assert_eq!(check_contract(&signature(), &[48, 96], &[16]), Ok(()));
    }

    #[test]
    fn a_larger_buffer_is_accepted() {
        assert_eq!(
            check_contract(&signature(), &[48, 1 << 20], &[64]),
            Ok(()),
            "a view into a bigger allocation still covers its port"
        );
    }

    #[test]
    fn an_undersized_input_is_refused_by_port() {
        assert_eq!(
            check_contract(&signature(), &[48, 95], &[16]),
            Err(BindingError::Undersized {
                side: Side::Input,
                port: 1,
                bound: 95,
                stated: 96,
            }),
            "one byte short is short"
        );
    }

    #[test]
    fn an_undersized_output_is_refused_by_port() {
        assert_eq!(
            check_contract(&signature(), &[48, 96], &[15]),
            Err(BindingError::Undersized {
                side: Side::Output,
                port: 0,
                bound: 15,
                stated: 16,
            })
        );
    }

    #[test]
    fn too_few_and_too_many_buffers_are_both_refused() {
        assert_eq!(
            check_contract(&signature(), &[48], &[16]),
            Err(BindingError::Count {
                side: Side::Input,
                bound: 1,
                stated: 2,
            })
        );
        assert_eq!(
            check_contract(&signature(), &[48, 96, 8], &[16]),
            Err(BindingError::Count {
                side: Side::Input,
                bound: 3,
                stated: 2,
            })
        );
        assert_eq!(
            check_contract(&signature(), &[48, 96], &[16, 16]),
            Err(BindingError::Count {
                side: Side::Output,
                bound: 2,
                stated: 1,
            })
        );
    }

    /// The sides are checked separately, so an input that happens to match an
    /// output's extent does not cover for a missing one.
    #[test]
    fn the_sides_are_not_interchangeable() {
        let square = RecipeSignature::new(vec![16], vec![16]);
        assert_eq!(check_contract(&square, &[16], &[]), {
            Err(BindingError::Count {
                side: Side::Output,
                bound: 0,
                stated: 1,
            })
        });
    }

    #[test]
    fn a_portless_recipe_accepts_nothing_bound() {
        let empty = RecipeSignature::empty();
        assert_eq!(check_contract(&empty, &[], &[]), Ok(()));
        assert!(check_contract(&empty, &[8], &[]).is_err());
    }

    /// The message names the side, the port, and both numbers — what a caller
    /// needs to find the buffer it got wrong.
    #[test]
    fn the_message_says_which_buffer_and_by_how_much() {
        let error = check_contract(&signature(), &[48, 95], &[16]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "input 1 is 95 bytes, recipe addresses 96"
        );
        let error = check_contract(&signature(), &[48], &[16]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "recipe states 2 input buffer(s), 1 bound"
        );
    }
}
