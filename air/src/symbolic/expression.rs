use core::iter::{Product, Sum};
use core::ops;

use p3_field::{Algebra, ExtensionField, Field, InjectiveMonomial, PrimeCharacteristicRing};

use crate::symbolic::variable::SymbolicVariable;
use crate::symbolic::{SymbolicOperation, alloc_node, get_node};

/// A node stored in the arena for base-field symbolic expressions.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicBaseNode<F> {
    pub op: SymbolicOperation,
    pub lhs: SymbolicExpression<F>,
    pub rhs: SymbolicExpression<F>, // dummy (ZERO) for Neg
    pub degree_multiple: usize,
}

/// A symbolic expression for base-field AIR constraints.
///
/// All variants are `Copy`. Tree structure is stored in a thread-local arena,
/// referenced by a `u32` byte offset in the `Operation` variant.
#[derive(Copy, Clone, Debug)]
pub enum SymbolicExpression<F> {
    /// A reference to a trace column or public input.
    Variable(SymbolicVariable<F>),
    /// Selector: non-zero only on the first row.
    IsFirstRow,
    /// Selector: non-zero only on the last row.
    IsLastRow,
    /// Selector: zero only on the last row.
    IsTransition,
    /// A constant field element.
    Constant(F),
    /// An arithmetic operation node stored in the arena.
    Operation(u32),
}

impl<F: Field> SymbolicExpression<F> {
    /// Returns the degree multiple of this expression.
    pub fn degree_multiple(&self) -> usize {
        match *self {
            Self::Variable(v) => v.degree_multiple(),
            Self::IsFirstRow | Self::IsLastRow => 1,
            Self::IsTransition | Self::Constant(_) => 0,
            Self::Operation(idx) => get_node::<SymbolicBaseNode<F>>(idx).degree_multiple,
        }
    }

    /// Try to view this expression as a constant field element.
    pub(crate) fn as_const(&self) -> Option<F> {
        match *self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    fn sym_add(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Constant(a + b);
        }
        if self.as_const().map_or(false, |c| c.is_zero()) {
            return rhs;
        }
        if rhs.as_const().map_or(false, |c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Operation(alloc_node(SymbolicBaseNode {
            op: SymbolicOperation::Add,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }

    fn sym_sub(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Constant(a - b);
        }
        if self.as_const().map_or(false, |c| c.is_zero()) {
            return rhs.sym_neg();
        }
        if rhs.as_const().map_or(false, |c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Operation(alloc_node(SymbolicBaseNode {
            op: SymbolicOperation::Sub,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }

    fn sym_neg(self) -> Self {
        if let Some(c) = self.as_const() {
            return Self::Constant(-c);
        }
        let dm = self.degree_multiple();
        Self::Operation(alloc_node(SymbolicBaseNode {
            op: SymbolicOperation::Neg,
            lhs: self,
            rhs: Self::ZERO,
            degree_multiple: dm,
        }))
    }

    fn sym_mul(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Constant(a * b);
        }
        if self.as_const().map_or(false, |c| c.is_zero())
            || rhs.as_const().map_or(false, |c| c.is_zero())
        {
            return Self::Constant(F::ZERO);
        }
        if self.as_const().map_or(false, |c| c.is_one()) {
            return rhs;
        }
        if rhs.as_const().map_or(false, |c| c.is_one()) {
            return self;
        }
        let dm = self.degree_multiple() + rhs.degree_multiple();
        Self::Operation(alloc_node(SymbolicBaseNode {
            op: SymbolicOperation::Mul,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }
}

// ── Trait implementations ────────────────────────────────────────────

impl<F: Field> Default for SymbolicExpression<F> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field> PrimeCharacteristicRing for SymbolicExpression<F> {
    type PrimeSubfield = F::PrimeSubfield;

    const ZERO: Self = Self::Constant(F::ZERO);
    const ONE: Self = Self::Constant(F::ONE);
    const TWO: Self = Self::Constant(F::TWO);
    const NEG_ONE: Self = Self::Constant(F::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::Constant(F::from_prime_subfield(f))
    }
}

// ── From conversions ─────────────────────────────────────────────────

impl<F: Field, EF: ExtensionField<F>> From<SymbolicVariable<F>> for SymbolicExpression<EF> {
    fn from(var: SymbolicVariable<F>) -> Self {
        Self::Variable(SymbolicVariable::new(var.entry, var.index))
    }
}

impl<F: Field, EF: ExtensionField<F>> From<F> for SymbolicExpression<EF> {
    fn from(f: F) -> Self {
        Self::Constant(f.into())
    }
}

// ── Algebra / monomial ───────────────────────────────────────────────

impl<F: Field> Algebra<F> for SymbolicExpression<F> {}
impl<F: Field> Algebra<SymbolicVariable<F>> for SymbolicExpression<F> {}
impl<F: Field + InjectiveMonomial<N>, const N: u64> InjectiveMonomial<N>
    for SymbolicExpression<F>
{
}

// ── Arithmetic operators ─────────────────────────────────────────────

impl<F: Field, T: Into<Self>> ops::Add<T> for SymbolicExpression<F> {
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        self.sym_add(rhs.into())
    }
}

impl<F: Field, T: Into<Self>> ops::Sub<T> for SymbolicExpression<F> {
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        self.sym_sub(rhs.into())
    }
}

impl<F: Field> ops::Neg for SymbolicExpression<F> {
    type Output = Self;
    fn neg(self) -> Self {
        self.sym_neg()
    }
}

impl<F: Field, T: Into<Self>> ops::Mul<T> for SymbolicExpression<F> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        self.sym_mul(rhs.into())
    }
}

impl<F: Field, T: Into<Self>> ops::AddAssign<T> for SymbolicExpression<F> {
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl<F: Field, T: Into<Self>> ops::SubAssign<T> for SymbolicExpression<F> {
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs.into();
    }
}

impl<F: Field, T: Into<Self>> ops::MulAssign<T> for SymbolicExpression<F> {
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl<F: Field, T: Into<Self>> Sum<T> for SymbolicExpression<F> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a + b)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, T: Into<Self>> Product<T> for SymbolicExpression<F> {
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a * b)
            .unwrap_or(Self::ONE)
    }
}
