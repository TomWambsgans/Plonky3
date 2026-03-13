//! Symbolic expression types for AIR constraint representation.
//!
//! Uses a thread-local arena so that all symbolic expression types are `Copy`.

mod builder;
mod expression;
pub(crate) mod expression_ext;
mod variable;

extern crate std;

use alloc::vec::Vec;
use core::cell::RefCell;
use core::ops;

pub use builder::*;
pub use expression::{SymbolicBaseNode, SymbolicExpression};
pub use expression_ext::SymbolicExpressionExt;
use p3_field::{ExtensionField, Field};
pub use variable::{BaseEntry, ExtEntry, SymbolicVariable, SymbolicVariableExt};

/// Operation types for symbolic expression arena nodes.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SymbolicOperation {
    Add,
    Sub,
    Mul,
    Neg,
}

// Thread-local byte arena for symbolic expression nodes.
// This allows SymbolicExpression and SymbolicExpressionExt to be Copy
// by storing tree nodes in the arena and referencing them by byte offset.
std::thread_local! {
    static ARENA: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Allocate a node in the thread-local arena, returning its byte offset.
pub(crate) fn alloc_node<T: Copy>(node: T) -> u32 {
    ARENA.with(|arena| {
        let mut bytes = arena.borrow_mut();
        let node_size = core::mem::size_of::<T>();
        let idx = bytes.len();
        bytes.resize(idx + node_size, 0);
        unsafe {
            core::ptr::write_unaligned(bytes.as_mut_ptr().add(idx) as *mut T, node);
        }
        idx as u32
    })
}

/// Read a node from the thread-local arena at the given byte offset.
pub fn get_node<T: Copy>(idx: u32) -> T {
    ARENA.with(|arena| {
        let bytes = arena.borrow();
        unsafe { core::ptr::read_unaligned(bytes.as_ptr().add(idx as usize) as *const T) }
    })
}

/// Clear the thread-local arena. Call before building symbolic constraints.
pub fn clear_symbolic_arena() {
    ARENA.with(|arena| arena.borrow_mut().clear());
}

// ── SymbolicVariable arithmetic ops ──────────────────────────────────

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Add<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs.into()
    }
}

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Sub<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs.into()
    }
}

impl<F: Field, T: Into<SymbolicExpression<F>>> ops::Mul<T> for SymbolicVariable<F> {
    type Output = SymbolicExpression<F>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs.into()
    }
}

// ── SymbolicVariableExt arithmetic ops ───────────────────────────────

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Add<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn add(self, rhs: T) -> Self::Output {
        Self::Output::from(self) + rhs.into()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Sub<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn sub(self, rhs: T) -> Self::Output {
        Self::Output::from(self) - rhs.into()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<SymbolicExpressionExt<F, EF>>> ops::Mul<T>
    for SymbolicVariableExt<F, EF>
{
    type Output = SymbolicExpressionExt<F, EF>;
    fn mul(self, rhs: T) -> Self::Output {
        Self::Output::from(self) * rhs.into()
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::symbolic::expression_ext::SymbolicExpressionExt;
    use crate::symbolic::variable::{BaseEntry, ExtEntry};

    type F = BabyBear;
    type EF = BinomialExtensionField<BabyBear, 4>;

    #[test]
    fn symbolic_variable_add_produces_add_node() {
        clear_symbolic_arena();
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let expr = SymbolicExpression::from(F::new(5));
        let result = var + expr;
        match result {
            SymbolicExpression::Operation(idx) => {
                let node = get_node::<expression::SymbolicBaseNode<F>>(idx);
                assert_eq!(node.op, SymbolicOperation::Add);
                assert_eq!(node.degree_multiple, 1);
                assert!(matches!(node.lhs, SymbolicExpression::Variable(v) if v.index == 0));
                assert!(matches!(node.rhs, SymbolicExpression::Constant(c) if c == F::new(5)));
            }
            _ => panic!("Expected an Operation node"),
        }
    }

    #[test]
    fn symbolic_variable_sub_produces_sub_node() {
        clear_symbolic_arena();
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let other = SymbolicExpression::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset: 0 },
            1,
        ));
        let result = var - other;
        match result {
            SymbolicExpression::Operation(idx) => {
                let node = get_node::<expression::SymbolicBaseNode<F>>(idx);
                assert_eq!(node.op, SymbolicOperation::Sub);
                assert_eq!(node.degree_multiple, 1);
            }
            _ => panic!("Expected an Operation node"),
        }
    }

    #[test]
    fn symbolic_variable_mul_produces_mul_node() {
        clear_symbolic_arena();
        let var = SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0);
        let other = SymbolicExpression::Variable(SymbolicVariable::new(
            BaseEntry::Main { offset: 0 },
            1,
        ));
        let result = var * other;
        match result {
            SymbolicExpression::Operation(idx) => {
                let node = get_node::<expression::SymbolicBaseNode<F>>(idx);
                assert_eq!(node.op, SymbolicOperation::Mul);
                assert_eq!(node.degree_multiple, 2);
            }
            _ => panic!("Expected an Operation node"),
        }
    }

    #[test]
    fn symbolic_variable_ext_add_produces_add_node() {
        clear_symbolic_arena();
        let var = SymbolicVariableExt::<F, EF>::new(ExtEntry::Permutation { offset: 0 }, 0);
        let expr = SymbolicExpressionExt::<F, EF>::from(F::new(3));
        let result = var + expr;
        match result {
            SymbolicExpressionExt::Operation(idx) => {
                let node =
                    get_node::<expression_ext::SymbolicExtNode<F, EF>>(idx);
                assert_eq!(node.op, SymbolicOperation::Add);
                assert_eq!(node.degree_multiple, 1);
            }
            _ => panic!("Expected an Operation node"),
        }
    }

    #[test]
    fn test_ring_constants() {
        assert!(matches!(
            SymbolicExpression::<F>::ZERO,
            SymbolicExpression::Constant(c) if c == F::ZERO
        ));
        assert!(matches!(
            SymbolicExpression::<F>::ONE,
            SymbolicExpression::Constant(c) if c == F::ONE
        ));
        assert!(matches!(
            SymbolicExpression::<F>::TWO,
            SymbolicExpression::Constant(c) if c == F::TWO
        ));
        assert!(matches!(
            SymbolicExpression::<F>::NEG_ONE,
            SymbolicExpression::Constant(c) if c == F::NEG_ONE
        ));
    }

    #[test]
    fn test_constant_folding() {
        let a = SymbolicExpression::Constant(F::new(3));
        let b = SymbolicExpression::Constant(F::new(4));
        assert!(matches!(a + b, SymbolicExpression::Constant(c) if c == F::new(7)));

        let a = SymbolicExpression::Constant(F::new(10));
        let b = SymbolicExpression::Constant(F::new(4));
        assert!(matches!(a - b, SymbolicExpression::Constant(c) if c == F::new(6)));

        let a = SymbolicExpression::Constant(F::new(3));
        let b = SymbolicExpression::Constant(F::new(5));
        assert!(matches!(a * b, SymbolicExpression::Constant(c) if c == F::new(15)));

        let a = SymbolicExpression::Constant(F::new(7));
        assert!(matches!(-a, SymbolicExpression::Constant(c) if c == F::NEG_ONE * F::new(7)));
    }

    #[test]
    fn test_identity_folding() {
        clear_symbolic_arena();
        let var = SymbolicExpression::Variable(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let zero = SymbolicExpression::<F>::Constant(F::ZERO);
        let one = SymbolicExpression::<F>::Constant(F::ONE);

        // x + 0 = x
        assert!(matches!(var + zero, SymbolicExpression::Variable(_)));
        // 0 + x = x
        assert!(matches!(zero + var, SymbolicExpression::Variable(_)));
        // x - 0 = x
        assert!(matches!(var - zero, SymbolicExpression::Variable(_)));
        // x * 1 = x
        assert!(matches!(var * one, SymbolicExpression::Variable(_)));
        // 1 * x = x
        assert!(matches!(one * var, SymbolicExpression::Variable(_)));
        // x * 0 = 0
        assert!(matches!(var * zero, SymbolicExpression::Constant(c) if c == F::ZERO));
        // 0 * x = 0
        assert!(matches!(zero * var, SymbolicExpression::Constant(c) if c == F::ZERO));
    }

    #[test]
    fn test_sum_and_product() {
        use alloc::vec;
        let exprs = vec![
            SymbolicExpression::Constant(F::new(2)),
            SymbolicExpression::Constant(F::new(3)),
            SymbolicExpression::Constant(F::new(5)),
        ];
        let result: SymbolicExpression<F> = exprs.into_iter().sum();
        assert!(matches!(result, SymbolicExpression::Constant(c) if c == F::new(10)));

        let exprs = vec![
            SymbolicExpression::Constant(F::new(2)),
            SymbolicExpression::Constant(F::new(3)),
            SymbolicExpression::Constant(F::new(4)),
        ];
        let result: SymbolicExpression<F> = exprs.into_iter().product();
        assert!(matches!(result, SymbolicExpression::Constant(c) if c == F::new(24)));
    }

    #[test]
    fn test_degree_tracking() {
        clear_symbolic_arena();
        let a = SymbolicExpression::Variable(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            0,
        ));
        let b = SymbolicExpression::Variable(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            1,
        ));
        let c = SymbolicExpression::Variable(SymbolicVariable::<F>::new(
            BaseEntry::Main { offset: 0 },
            2,
        ));

        let ab = a * b;
        assert_eq!(ab.degree_multiple(), 2);

        let abc = ab * c;
        assert_eq!(abc.degree_multiple(), 3);
    }
}
