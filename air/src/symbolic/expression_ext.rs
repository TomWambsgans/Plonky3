use core::iter::{Product, Sum};
use core::ops;

use p3_field::extension::BinomialExtensionField;
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};

use crate::symbolic::expression::{SymbolicBaseNode, SymbolicExpression};
use crate::symbolic::variable::{SymbolicVariable, SymbolicVariableExt};
use crate::symbolic::{SymbolicOperation, alloc_node, get_node};

/// A node stored in the arena for extension-field symbolic expressions.
#[derive(Copy, Clone, Debug)]
pub struct SymbolicExtNode<F, EF> {
    pub op: SymbolicOperation,
    pub lhs: SymbolicExpressionExt<F, EF>,
    pub rhs: SymbolicExpressionExt<F, EF>, // dummy (ZERO) for Neg
    pub degree_multiple: usize,
}

/// A symbolic expression for extension-field AIR constraints.
///
/// All variants are `Copy`. Tree structure is stored in a thread-local arena.
#[derive(Copy, Clone, Debug)]
pub enum SymbolicExpressionExt<F, EF> {
    /// A lifted base-field expression.
    Base(SymbolicExpression<F>),
    /// An extension-field variable (permutation column or challenge).
    ExtVariable(SymbolicVariableExt<F, EF>),
    /// An extension-field constant.
    ExtConstant(EF),
    /// An arithmetic operation node stored in the arena.
    Operation(u32),
}

impl<F: Field, EF: ExtensionField<F>> SymbolicExpressionExt<F, EF> {
    /// Returns the degree multiple of this expression.
    pub fn degree_multiple(&self) -> usize {
        match *self {
            Self::Base(e) => e.degree_multiple(),
            Self::ExtVariable(v) => v.degree_multiple(),
            Self::ExtConstant(_) => 0,
            Self::Operation(idx) => get_node::<SymbolicExtNode<F, EF>>(idx).degree_multiple,
        }
    }

    /// Try to view this expression as a base-field constant.
    fn as_const(&self) -> Option<F> {
        match *self {
            Self::Base(SymbolicExpression::Constant(c)) => Some(c),
            Self::ExtConstant(ef) if ef.is_in_basefield() => {
                Some(ef.as_basis_coefficients_slice()[0])
            }
            _ => None,
        }
    }

    /// Try to lower this extension expression to a base-field expression.
    ///
    /// Returns `None` if the tree contains any extension-only nodes.
    pub fn to_base(&self) -> Option<SymbolicExpression<F>> {
        match *self {
            Self::Base(e) => Some(e),
            Self::ExtVariable(_) | Self::ExtConstant(_) => None,
            Self::Operation(idx) => {
                let node: SymbolicExtNode<F, EF> = get_node(idx);
                let lhs = node.lhs.to_base()?;
                let rhs = node.rhs.to_base()?;
                Some(SymbolicExpression::Operation(alloc_node(
                    SymbolicBaseNode {
                        op: node.op,
                        lhs,
                        rhs,
                        degree_multiple: node.degree_multiple,
                    },
                )))
            }
        }
    }

    fn sym_add(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Base(SymbolicExpression::Constant(a + b));
        }
        if self.as_const().map_or(false, |c| c.is_zero()) {
            return rhs;
        }
        if rhs.as_const().map_or(false, |c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Operation(alloc_node(SymbolicExtNode {
            op: SymbolicOperation::Add,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }

    fn sym_sub(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Base(SymbolicExpression::Constant(a - b));
        }
        if self.as_const().map_or(false, |c| c.is_zero()) {
            return rhs.sym_neg();
        }
        if rhs.as_const().map_or(false, |c| c.is_zero()) {
            return self;
        }
        let dm = self.degree_multiple().max(rhs.degree_multiple());
        Self::Operation(alloc_node(SymbolicExtNode {
            op: SymbolicOperation::Sub,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }

    fn sym_neg(self) -> Self {
        if let Some(c) = self.as_const() {
            return Self::Base(SymbolicExpression::Constant(-c));
        }
        let dm = self.degree_multiple();
        Self::Operation(alloc_node(SymbolicExtNode {
            op: SymbolicOperation::Neg,
            lhs: self,
            rhs: Self::ZERO,
            degree_multiple: dm,
        }))
    }

    fn sym_mul(self, rhs: Self) -> Self {
        if let (Some(a), Some(b)) = (self.as_const(), rhs.as_const()) {
            return Self::Base(SymbolicExpression::Constant(a * b));
        }
        if self.as_const().map_or(false, |c| c.is_zero())
            || rhs.as_const().map_or(false, |c| c.is_zero())
        {
            return Self::Base(SymbolicExpression::Constant(F::ZERO));
        }
        if self.as_const().map_or(false, |c| c.is_one()) {
            return rhs;
        }
        if rhs.as_const().map_or(false, |c| c.is_one()) {
            return self;
        }
        let dm = self.degree_multiple() + rhs.degree_multiple();
        Self::Operation(alloc_node(SymbolicExtNode {
            op: SymbolicOperation::Mul,
            lhs: self,
            rhs,
            degree_multiple: dm,
        }))
    }
}

// ── Trait implementations ────────────────────────────────────────────

impl<F: Field, EF: ExtensionField<F>> Default for SymbolicExpressionExt<F, EF> {
    fn default() -> Self {
        Self::ZERO
    }
}

impl<F: Field, EF: ExtensionField<F>> PrimeCharacteristicRing for SymbolicExpressionExt<F, EF> {
    type PrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::Base(SymbolicExpression::ZERO);
    const ONE: Self = Self::Base(SymbolicExpression::ONE);
    const TWO: Self = Self::Base(SymbolicExpression::TWO);
    const NEG_ONE: Self = Self::Base(SymbolicExpression::NEG_ONE);

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        Self::Base(SymbolicExpression::from_prime_subfield(f))
    }
}

// ── From conversions ─────────────────────────────────────────────────

impl<F: Field, EF> From<SymbolicExpression<F>> for SymbolicExpressionExt<F, EF> {
    fn from(expr: SymbolicExpression<F>) -> Self {
        Self::Base(expr)
    }
}

impl<F: Field, EF> From<SymbolicVariable<F>> for SymbolicExpressionExt<F, EF> {
    fn from(var: SymbolicVariable<F>) -> Self {
        Self::Base(SymbolicExpression::Variable(var))
    }
}

impl<F, EF> From<SymbolicVariableExt<F, EF>> for SymbolicExpressionExt<F, EF> {
    fn from(var: SymbolicVariableExt<F, EF>) -> Self {
        Self::ExtVariable(var)
    }
}

impl<F: Field, EF> From<F> for SymbolicExpressionExt<F, EF> {
    fn from(f: F) -> Self {
        Self::Base(SymbolicExpression::Constant(f))
    }
}

/// Concrete [`From`] for [`BinomialExtensionField`] constants.
impl<F, const D: usize> From<BinomialExtensionField<F, D>>
    for SymbolicExpressionExt<F, BinomialExtensionField<F, D>>
where
    F: Field,
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
    fn from(ef: BinomialExtensionField<F, D>) -> Self {
        Self::ExtConstant(ef)
    }
}

// ── Algebra impls ────────────────────────────────────────────────────

impl<F: Field, EF: ExtensionField<F>> Algebra<F> for SymbolicExpressionExt<F, EF> {}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicExpression<F>>
    for SymbolicExpressionExt<F, EF>
{
}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicVariable<F>>
    for SymbolicExpressionExt<F, EF>
{
}

impl<F: Field, EF: ExtensionField<F>> Algebra<SymbolicVariableExt<F, EF>>
    for SymbolicExpressionExt<F, EF>
{
}

impl<F: Field, const D: usize> Algebra<BinomialExtensionField<F, D>>
    for SymbolicExpressionExt<F, BinomialExtensionField<F, D>>
where
    BinomialExtensionField<F, D>: ExtensionField<F>,
{
}

// ── Arithmetic operators ─────────────────────────────────────────────

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::Add<T>
    for SymbolicExpressionExt<F, EF>
{
    type Output = Self;
    fn add(self, rhs: T) -> Self {
        self.sym_add(rhs.into())
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::Sub<T>
    for SymbolicExpressionExt<F, EF>
{
    type Output = Self;
    fn sub(self, rhs: T) -> Self {
        self.sym_sub(rhs.into())
    }
}

impl<F: Field, EF: ExtensionField<F>> ops::Neg for SymbolicExpressionExt<F, EF> {
    type Output = Self;
    fn neg(self) -> Self {
        self.sym_neg()
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::Mul<T>
    for SymbolicExpressionExt<F, EF>
{
    type Output = Self;
    fn mul(self, rhs: T) -> Self {
        self.sym_mul(rhs.into())
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::AddAssign<T>
    for SymbolicExpressionExt<F, EF>
{
    fn add_assign(&mut self, rhs: T) {
        *self = *self + rhs.into();
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::SubAssign<T>
    for SymbolicExpressionExt<F, EF>
{
    fn sub_assign(&mut self, rhs: T) {
        *self = *self - rhs.into();
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> ops::MulAssign<T>
    for SymbolicExpressionExt<F, EF>
{
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs.into();
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> Sum<T> for SymbolicExpressionExt<F, EF> {
    fn sum<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a + b)
            .unwrap_or(Self::ZERO)
    }
}

impl<F: Field, EF: ExtensionField<F>, T: Into<Self>> Product<T>
    for SymbolicExpressionExt<F, EF>
{
    fn product<I: Iterator<Item = T>>(iter: I) -> Self {
        iter.map(Into::into)
            .reduce(|a, b| a * b)
            .unwrap_or(Self::ONE)
    }
}
