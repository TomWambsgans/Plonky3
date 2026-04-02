//! Degree-3 extension field using the trinomial `X^3 - X - 1`.
//!
//! This extension requires that `X^3 - X - 1` is irreducible over the base field.
//! Currently used for Goldilocks where irreducibility has been verified.
//!
//! Reduction identity: `X^3 = X + 1`
//! (and consequently `X^4 = X^2 + X`)

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use num_bigint::BigUint;
use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::packed_cubic_extension::PackedCubicTrinomialExtensionField;
use super::{CubicExtendableAlgebra, HasFrobenius, HasTwoAdicCubicExtension};
use crate::extension::CubicTrinomialExtendable;
use crate::field::Field;
use crate::{
    Algebra, BasedVectorSpace, ExtensionField, Packable, PackedFieldExtension,
    PrimeCharacteristicRing, RawDataSerializable, TwoAdicField, field_to_array,
};

/// A degree-3 extension field using the trinomial `X^3 - X - 1`.
///
/// Elements are represented as `a_0 + a_1*X + a_2*X^2`.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Required for safe memory layout casts.
#[must_use]
pub struct CubicTrinomialExtensionField<F, A = F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "A: Serialize", deserialize = "A: Deserialize<'de>")
    )]
    pub(crate) value: [A; 3],
    _phantom: PhantomData<F>,
}

impl<F, A> CubicTrinomialExtensionField<F, A> {
    /// Create an extension field element from coefficient array.
    ///
    /// The coefficients represent the polynomial `value[0] + value[1]*X + value[2]*X^2`.
    #[inline]
    pub const fn new(value: [A; 3]) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }
}

impl<F: Copy> CubicTrinomialExtensionField<F, F> {
    /// Convert a `[[F; 3]; N]` array to an array of extension field elements.
    ///
    /// # Panics
    /// Panics if `N == 0`.
    #[inline]
    pub const fn new_array<const N: usize>(input: [[F; 3]; N]) -> [Self; N] {
        const { assert!(N > 0) }
        let mut output = [Self::new(input[0]); N];
        let mut i = 1;
        while i < N {
            output[i] = Self::new(input[i]);
            i += 1;
        }
        output
    }
}

impl<F: Field, A: Algebra<F>> Default for CubicTrinomialExtensionField<F, A> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| A::ZERO))
    }
}

impl<F: Field, A: Algebra<F>> From<A> for CubicTrinomialExtensionField<F, A> {
    fn from(x: A) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F, A> From<[A; 3]> for CubicTrinomialExtensionField<F, A> {
    #[inline]
    fn from(x: [A; 3]) -> Self {
        Self {
            value: x,
            _phantom: PhantomData,
        }
    }
}

impl<F: CubicTrinomialExtendable> Packable for CubicTrinomialExtensionField<F> {}

impl<F: CubicTrinomialExtendable, A: Algebra<F>> BasedVectorSpace<A>
    for CubicTrinomialExtensionField<F, A>
{
    const DIMENSION: usize = 3;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[A] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> A>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = A>>(mut iter: I) -> Option<Self> {
        (iter.len() == 3).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<A> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 3]`.
        unsafe { flatten_to_base::<A, Self>(vec) }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<A>) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 3]`.
        unsafe { reconstitute_from_base::<A, Self>(vec) }
    }
}

impl<F: CubicTrinomialExtendable> ExtensionField<F> for CubicTrinomialExtensionField<F>
where
    F::Packing: CubicExtendableAlgebra<F>,
    PackedCubicTrinomialExtensionField<F, F::Packing>: PackedFieldExtension<F, Self>,
{
    type ExtensionPacking = PackedCubicTrinomialExtensionField<F, F::Packing>;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<F> {
        <Self as ExtensionField<F>>::is_in_basefield(self).then(|| self.value[0])
    }
}

impl<F: CubicTrinomialExtendable> HasFrobenius<F> for CubicTrinomialExtensionField<F>
where
    F::Packing: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn frobenius(&self) -> Self {
        let a = &self.value;
        let fc = &F::FROBENIUS_COEFFS;

        // phi(a) = a_0 + a_1 * X^p + a_2 * X^{2p}
        // where X^p = fc[0] and X^{2p} = fc[1]
        let a_tail = &[a[1], a[2]];
        let c0 = a[0] + F::dot_product::<2>(a_tail, &[fc[0][0], fc[1][0]]);
        let c1 = F::dot_product::<2>(a_tail, &[fc[0][1], fc[1][1]]);
        let c2 = F::dot_product::<2>(a_tail, &[fc[0][2], fc[1][2]]);

        Self::new([c0, c1, c2])
    }

    /// Apply Frobenius `count` times: `x -> x^{p^count}`.
    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        match count % 3 {
            0 => *self,
            _ => {
                let mut result = *self;
                for _ in 0..(count % 3) {
                    result = result.frobenius();
                }
                result
            }
        }
    }

    /// Compute pseudo-inverse using Frobenius automorphism.
    ///
    /// Returns `0` if `self == 0`, and `1/self` otherwise.
    ///
    /// Uses the identity: `a^{-1} = ProdConj(a) * Norm(a)^{-1}` where
    /// - `ProdConj(a) = a^{p + p^2}`,
    /// - `Norm(a) = a * ProdConj(a)` is in the base field.
    #[inline]
    fn pseudo_inv(&self) -> Self {
        if self.is_zero() {
            return Self::ZERO;
        }

        // Compute ProdConj(a) = a^{p + p^2} = a^p * a^{p^2}
        let a_p = self.frobenius();
        let a_p2 = a_p.frobenius();
        let prod_conj = a_p * a_p2;

        // Norm(a) = a * ProdConj(a) lies in the base field.
        // Compute only the constant coefficient.
        let norm = self.compute_norm_with_prod_conj(&prod_conj);
        debug_assert_eq!(Self::from(norm), *self * prod_conj);

        prod_conj * norm.inverse()
    }
}

impl<F: CubicTrinomialExtendable> CubicTrinomialExtensionField<F> {
    /// Compute the norm given pre-computed product of conjugates.
    ///
    /// The norm `Norm(a) = a * prod_conj` lies in the base field.
    /// This computes only the constant coefficient for efficiency.
    #[inline]
    fn compute_norm_with_prod_conj(&self, prod_conj: &Self) -> F {
        let a = &self.value;
        let b = &prod_conj.value;

        // For trinomial X^3 - X - 1, the constant term of a*b is:
        // c_0 + c_3 where c_k = sum_{i+j=k} a_i*b_j
        // c_0 = a_0*b_0
        // c_3 = a_1*b_2 + a_2*b_1 (reduced: x^3 = x + 1, contributes 1 to constant)
        a[0] * b[0] + a[1] * b[2] + a[2] * b[1]
    }
}

impl<F, A> PrimeCharacteristicRing for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    type PrimeSubfield = <A as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([A::ZERO; 3]);
    const ONE: Self = Self::new(field_to_array(A::ONE));
    const TWO: Self = Self::new(field_to_array(A::TWO));
    const NEG_ONE: Self = Self::new(field_to_array(A::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <A as PrimeCharacteristicRing>::from_prime_subfield(f).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(array::from_fn(|i| self.value[i].halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        A::cubic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(array::from_fn(|i| self.value[i].mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(array::from_fn(|i| self.value[i].div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: `Self` is `repr(transparent)` over `[A; 3]`.
        unsafe { reconstitute_from_base(A::zero_vec(len * 3)) }
    }
}

impl<F: CubicTrinomialExtendable> Algebra<F> for CubicTrinomialExtensionField<F> {}

impl<F: CubicTrinomialExtendable> RawDataSerializable for CubicTrinomialExtensionField<F> {
    const NUM_BYTES: usize = F::NUM_BYTES * 3;

    #[inline]
    fn into_bytes(self) -> impl IntoIterator<Item = u8> {
        self.value.into_iter().flat_map(|x| x.into_bytes())
    }

    #[inline]
    fn into_byte_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u8> {
        F::into_byte_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u32_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u32> {
        F::into_u32_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_u64_stream(input: impl IntoIterator<Item = Self>) -> impl IntoIterator<Item = u64> {
        F::into_u64_stream(input.into_iter().flat_map(|x| x.value))
    }

    #[inline]
    fn into_parallel_byte_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u8; N]> {
        F::into_parallel_byte_streams(
            input
                .into_iter()
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        F::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        F::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..3).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
}

impl<F: CubicTrinomialExtendable> Field for CubicTrinomialExtensionField<F> {
    type Packing = Self;

    const GENERATOR: Self = Self::new(F::EXT_GENERATOR);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        // Direct inversion using norm/adjugate method.
        // For x^3 - x - 1, the multiplication-by-a matrix is:
        //   M = | a0    a2      a1      |
        //       | a1    a0+a2   a1+a2   |
        //       | a2    a1      a0+a2   |
        //
        // We compute a^{-1} = adj(M) * e_1 / det(M).
        let [a0, a1, a2] = self.value;

        let a0_sq = a0.square();
        let a1_sq = a1.square();
        let a2_sq = a2.square();
        let a0a1 = a0 * a1;
        let a0a2 = a0 * a2;
        let a1a2 = a1 * a2;

        // Cofactor numerators (negated cofactors of first row of M):
        // n0 = a1*a2 + a1^2 - a0^2 - 2*a0*a2 - a2^2
        let n0 = a1a2 + a1_sq - a0_sq - a0a2.double() - a2_sq;
        // n1 = a0*a1 - a2^2
        let n1 = a0a1 - a2_sq;
        // n2 = a0*a2 + a2^2 - a1^2
        let n2 = a0a2 + a2_sq - a1_sq;

        // Norm t = a0*n0 + a2*n1 + a1*n2  (= -det(M))
        let t = a0 * n0 + a2 * n1 + a1 * n2;

        let t_inv = t.try_inverse()?;

        Some(Self::new([n0 * t_inv, n1 * t_inv, n2 * t_inv]))
    }

    #[inline]
    fn add_slices(slice_1: &mut [Self], slice_2: &[Self]) {
        // SAFETY: `Self` is `repr(transparent)` over `[F; 3]`.
        // Addition is F-linear, so we can operate on base field slices.
        unsafe {
            let base_slice_1 = as_base_slice_mut(slice_1);
            let base_slice_2 = as_base_slice(slice_2);
            F::add_slices(base_slice_1, base_slice_2);
        }
    }

    #[inline]
    fn order() -> BigUint {
        F::order().pow(3)
    }
}

impl<F: CubicTrinomialExtendable> Display for CubicTrinomialExtensionField<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            write!(f, "0")
        } else {
            let str = self
                .value
                .iter()
                .enumerate()
                .filter(|(_, x)| !x.is_zero())
                .map(|(i, x)| match (i, x.is_one()) {
                    (0, _) => format!("{x}"),
                    (1, true) => "X".to_string(),
                    (1, false) => format!("{x} X"),
                    (_, true) => format!("X^{i}"),
                    (_, false) => format!("{x} X^{i}"),
                })
                .join(" + ");
            write!(f, "{str}")
        }
    }
}

impl<F, A> Neg for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(A::neg))
    }
}

impl<F, A> Add for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self::new(A::cubic_add(&self.value, &rhs.value))
    }
}

impl<F, A> Add<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: A) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, A> AddAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.value = A::cubic_add(&self.value, &rhs.value);
    }
}

impl<F, A> AddAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: A) {
        self.value[0] += rhs;
    }
}

impl<F, A> Sum for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, A> Sub for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self::new(A::cubic_sub(&self.value, &rhs.value))
    }
}

impl<F, A> Sub<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: A) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl<F, A> SubAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.value = A::cubic_sub(&self.value, &rhs.value);
    }
}

impl<F, A> SubAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: Algebra<F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: A) {
        self.value[0] -= rhs;
    }
}

impl<F, A> Mul for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let mut res = Self::default();
        A::cubic_mul(&self.value, &rhs.value, &mut res.value);
        res
    }
}

impl<F, A> Mul<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: A) -> Self {
        Self::new(A::cubic_base_mul(self.value, rhs))
    }
}

impl<F, A> MulAssign for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> MulAssign<A> for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: A) {
        *self = self.clone() * rhs;
    }
}

impl<F, A> Product for CubicTrinomialExtensionField<F, A>
where
    F: CubicTrinomialExtendable,
    A: CubicExtendableAlgebra<F> + Copy,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F> Div for CubicTrinomialExtensionField<F>
where
    F: CubicTrinomialExtendable,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F> DivAssign for CubicTrinomialExtensionField<F>
where
    F: CubicTrinomialExtendable,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: CubicTrinomialExtendable> Distribution<CubicTrinomialExtensionField<F>>
    for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> CubicTrinomialExtensionField<F> {
        CubicTrinomialExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: CubicTrinomialExtendable + HasTwoAdicCubicExtension> TwoAdicField
    for CubicTrinomialExtensionField<F>
{
    const TWO_ADICITY: usize = F::EXT_TWO_ADICITY;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        Self::new(F::ext_two_adic_generator(bits))
    }
}

/// Multiply two elements in the cubic trinomial extension field (X^3 - X - 1).
///
/// Uses Karatsuba: 6 base-field multiplications instead of 9 schoolbook.
/// Reduction: X^3 = X + 1, X^4 = X^2 + X
#[inline]
pub fn trinomial_cubic_mul<R: PrimeCharacteristicRing>(a: &[R; 3], b: &[R; 3], res: &mut [R; 3]) {
    // Karatsuba: compute 3 diagonal and 3 cross products.
    let m0 = a[0].dup() * b[0].dup();
    let m1 = a[1].dup() * b[1].dup();
    let m2 = a[2].dup() * b[2].dup();

    let t01 = (a[0].dup() + a[1].dup()) * (b[0].dup() + b[1].dup());
    let t02 = (a[0].dup() + a[2].dup()) * (b[0].dup() + b[2].dup());
    let t12 = (a[1].dup() + a[2].dup()) * (b[1].dup() + b[2].dup());

    // c3 = a1*b2 + a2*b1 = t12 - m1 - m2 (shared subexpression)
    let c3 = t12 - m1.dup() - m2.dup();

    // Apply reduction: X^3 = X + 1, X^4 = X^2 + X
    // r0 = m0 + c3
    res[0] = m0.dup() + c3.dup();
    // r1 = (t01 - m0 - m1) + c3 + m2
    res[1] = t01 + c3 + m2 - m0.dup() - m1.dup();
    // r2 = (t02 - m0 - m2 + m1) + m2 = t02 - m0 + m1
    res[2] = t02 - m0 + m1;
}

/// Square an element in the cubic extension field.
///
/// Reduction: X^3 = X + 1, X^4 = X^2 + X
#[inline]
pub(super) fn cubic_square<R: PrimeCharacteristicRing>(a: &[R; 3], res: &mut [R; 3]) {
    // Precompute doubled coefficients for cross terms
    let a0_2 = a[0].double();

    // Convolution coefficients
    // c0 = a0^2
    let c0 = a[0].square();
    // c1 = 2*a0*a1
    let c1 = a0_2.dup() * a[1].dup();
    // c2 = 2*a0*a2 + a1^2
    let c2 = R::dot_product::<2>(&[a0_2, a[1].dup()], &[a[2].dup(), a[1].dup()]);
    // c3 = 2*a1*a2
    let c3 = a[1].dup() * a[2].double();
    // c4 = a2^2
    let c4 = a[2].square();

    // Apply reduction: X^3 = X + 1, X^4 = X^2 + X
    res[0] = c0 + c3.dup();
    res[1] = c1 + c3 + c4.dup();
    res[2] = c2 + c4;
}
