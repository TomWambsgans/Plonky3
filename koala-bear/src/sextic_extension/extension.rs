use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;
use core::array;
use core::fmt::{self, Debug, Display, Formatter};
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use num_bigint::BigUint;
use p3_field::extension::HasFrobenius;
use p3_field::{
    Algebra, BasedVectorSpace, ExtensionField, Field, Packable, PrimeCharacteristicRing,
    RawDataSerializable, TwoAdicField, field_to_array,
};
use p3_util::{as_base_slice, as_base_slice_mut, flatten_to_base, reconstitute_from_base};
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};

use super::packed_extension::PackedSexticExtensionField;
use crate::SexticExtendable;

/// Sextic Extension Field (degree 6), specifically designed for Koala-Bear.
/// Irreducible polynomial: w^6 - 2w^3 - 2
/// Reduction rule: w^6 = 2w^3 + 2
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
#[must_use]
pub struct SexticExtensionField<F> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>")
    )]
    pub(crate) value: [F; 6],
}

impl<F> SexticExtensionField<F> {
    pub(crate) const fn new(value: [F; 6]) -> Self {
        Self { value }
    }
}

impl<F: Field> Default for SexticExtensionField<F> {
    fn default() -> Self {
        Self::new(array::from_fn(|_| F::ZERO))
    }
}

impl<F: Field> From<F> for SexticExtensionField<F> {
    fn from(x: F) -> Self {
        Self::new(field_to_array(x))
    }
}

impl<F: SexticExtendable> Packable for SexticExtensionField<F> {}

impl<F: SexticExtendable> BasedVectorSpace<F> for SexticExtensionField<F> {
    const DIMENSION: usize = 6;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[F] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> F>(f: Fn) -> Self {
        Self::new(array::from_fn(f))
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = F>>(mut iter: I) -> Option<Self> {
        (iter.len() == 6).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<F> {
        unsafe {
            // Safety: Self is repr(transparent), stored identically to [F; 6].
            flatten_to_base::<F, Self>(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<F>) -> Vec<Self> {
        unsafe {
            // Safety: Self is repr(transparent), stored identically to [F; 6].
            reconstitute_from_base::<F, Self>(vec)
        }
    }
}

impl<F: SexticExtendable> ExtensionField<F> for SexticExtensionField<F> {
    type ExtensionPacking = PackedSexticExtensionField<F, F::Packing>;

    #[inline]
    fn is_in_basefield(&self) -> bool {
        self.value[1..].iter().all(F::is_zero)
    }

    #[inline]
    fn as_base(&self) -> Option<F> {
        <Self as ExtensionField<F>>::is_in_basefield(self).then(|| self.value[0])
    }
}

impl<F: SexticExtendable> HasFrobenius<F> for SexticExtensionField<F> {
    /// Frobenius automorphism: x -> x^p, where p is the order of the base field.
    #[inline]
    fn frobenius(&self) -> Self {
        let mut res = Self::ZERO;
        res.value[0] = self.value[0];
        for i in 0..5 {
            for j in 0..6 {
                res.value[j] += self.value[i + 1] * F::FROBENIUS_MATRIX[i][j];
            }
        }
        res
    }

    /// Repeated Frobenius automorphisms: x -> x^(p^count).
    #[inline]
    fn repeated_frobenius(&self, count: usize) -> Self {
        if count == 0 {
            return *self;
        } else if count >= 6 {
            // x |-> x^(p^6) is the identity, so x^(p^count) == x^(p^(count % 6))
            return self.repeated_frobenius(count % 6);
        }

        let mut res = self.frobenius();
        for _ in 1..count {
            res = res.frobenius();
        }
        res
    }

    #[inline]
    fn pseudo_inv(&self) -> Self {
        unimplemented!()
    }
}

impl<F> PrimeCharacteristicRing for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type PrimeSubfield = <F as PrimeCharacteristicRing>::PrimeSubfield;

    const ZERO: Self = Self::new([F::ZERO; 6]);

    const ONE: Self = Self::new(field_to_array(F::ONE));

    const TWO: Self = Self::new(field_to_array(F::TWO));

    const NEG_ONE: Self = Self::new(field_to_array(F::NEG_ONE));

    #[inline]
    fn from_prime_subfield(f: Self::PrimeSubfield) -> Self {
        <F as PrimeCharacteristicRing>::from_prime_subfield(f).into()
    }

    #[inline]
    fn halve(&self) -> Self {
        Self::new(self.value.map(|x| x.halve()))
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        sextic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn mul_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.map(|x| x.mul_2exp_u64(exp)))
    }

    #[inline]
    fn div_2exp_u64(&self, exp: u64) -> Self {
        Self::new(self.value.map(|x| x.div_2exp_u64(exp)))
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(F::zero_vec(len * 6)) }
    }
}

impl<F: SexticExtendable> Algebra<F> for SexticExtensionField<F> {}

impl<F: SexticExtendable> RawDataSerializable for SexticExtensionField<F> {
    const NUM_BYTES: usize = F::NUM_BYTES * 6;

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
                .flat_map(|x| (0..6).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u32_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u32; N]> {
        F::into_parallel_u32_streams(
            input
                .into_iter()
                .flat_map(|x| (0..6).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }

    #[inline]
    fn into_parallel_u64_streams<const N: usize>(
        input: impl IntoIterator<Item = [Self; N]>,
    ) -> impl IntoIterator<Item = [u64; N]> {
        F::into_parallel_u64_streams(
            input
                .into_iter()
                .flat_map(|x| (0..6).map(move |i| array::from_fn(|j| x[j].value[i]))),
        )
    }
}

impl<F: SexticExtendable> Field for SexticExtensionField<F> {
    type Packing = Self;

    const GENERATOR: Self = Self::new(F::EXT_GENERATOR);

    fn try_inverse(&self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }

        Some(sextic_inv(self))
    }

    #[inline]
    fn add_slices(slice_1: &mut [Self], slice_2: &[Self]) {
        // By construction, Self is repr(transparent) over [F; 6].
        // Additionally, addition is F-linear. Hence we can cast
        // everything to F and use F's add_slices.
        unsafe {
            let base_slice_1 = as_base_slice_mut(slice_1);
            let base_slice_2 = as_base_slice(slice_2);

            F::add_slices(base_slice_1, base_slice_2);
        }
    }

    #[inline]
    fn order() -> BigUint {
        F::order().pow(6)
    }
}

impl<F> Display for SexticExtensionField<F>
where
    F: SexticExtendable,
{
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

impl<F> Neg for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self::new(self.value.map(F::neg))
    }
}

impl<F> Add for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = F::sextic_add(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F> Add<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: F) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F> AddAssign for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F> AddAssign<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn add_assign(&mut self, rhs: F) {
        self.value[0] += rhs;
    }
}

impl<F> Sum for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F> Sub for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = F::sextic_sub(&self.value, &rhs.value);
        Self::new(value)
    }
}

impl<F> Sub<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self::new(res)
    }
}

impl<F> SubAssign for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.value[i] -= rhs.value[i];
        }
    }
}

impl<F> SubAssign<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn sub_assign(&mut self, rhs: F) {
        self.value[0] -= rhs;
    }
}

impl<F> Mul for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();

        F::sextic_mul(&a, &b, &mut res.value);

        res
    }
}

impl<F> Mul<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F) -> Self {
        Self::new(F::sextic_base_mul(self.value, rhs))
    }
}

impl<F> MulAssign for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F> MulAssign<F> for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}

impl<F> Product for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ONE)
    }
}

impl<F> Div for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)]
    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl<F> DivAssign for SexticExtensionField<F>
where
    F: SexticExtendable,
{
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<F: SexticExtendable> Distribution<SexticExtensionField<F>> for StandardUniform
where
    Self: Distribution<F>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> SexticExtensionField<F> {
        SexticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: TwoAdicField + SexticExtendable> TwoAdicField for SexticExtensionField<F> {
    // The two-adicity of p^6-1 is v_2(p-1) + v_2(p+1) = TWO_ADICITY + 1
    // because v_2(p^6-1) = v_2(p^2-1) + v_2(p^4+p^2+1)
    // and v_2(p^2-1) = v_2(p-1) + v_2(p+1) = TWO_ADICITY + 1
    // and p^4+p^2+1 is odd.
    const TWO_ADICITY: usize = F::TWO_ADICITY + 1;

    #[inline]
    fn two_adic_generator(bits: usize) -> Self {
        if bits <= F::TWO_ADICITY {
            F::two_adic_generator(bits).into()
        } else {
            assert_eq!(bits, F::TWO_ADICITY + 1);
            // Precomputed element of order 2^(TWO_ADICITY+1) in the sextic extension.
            Self::new(F::TWO_ADIC_SEXTIC_GENERATOR)
        }
    }
}

// ===== Core arithmetic functions =====

/// Schoolbook multiplication in F_p[w]/(w^6 - 2w^3 - 2).
///
/// Computes the product of two degree-5 polynomials, then reduces
/// using the relation w^6 = 2w^3 + 2.
pub(crate) fn sextic_mul<F, R, R2>(a: &[R; 6], b: &[R2; 6], res: &mut [R; 6])
where
    F: Field,
    R: Algebra<F> + Algebra<R2>,
    R2: Algebra<F>,
{
    // Convert b to R type for computation
    let b_r: [R; 6] = array::from_fn(|i| b[i].clone().into());

    // Compute the 11 unreduced coefficients of the product polynomial (degree 0..10).
    // c[k] = sum_{i+j=k} a[i] * b[j]
    let mut c: [R; 11] = array::from_fn(|_| R::ZERO);
    for i in 0..6 {
        for j in 0..6 {
            c[i + j] = c[i + j].clone() + a[i].clone() * b_r[j].clone();
        }
    }

    // Reduce using w^6 = 2w^3 + 2.
    // For k >= 6: w^k = 2*w^{k-3} + 2*w^{k-6}
    // Process from highest to lowest so each reduction only feeds into lower indices.
    for k in (6..=10).rev() {
        let two_ck = c[k].double();
        c[k - 3] = c[k - 3].clone() + two_ck.clone();
        c[k - 6] = c[k - 6].clone() + two_ck;
    }

    // Copy the reduced coefficients to the result.
    for i in 0..6 {
        res[i] = c[i].clone();
    }
}

/// Schoolbook squaring in F_p[w]/(w^6 - 2w^3 - 2).
///
/// Exploits the symmetry a[i]*a[j] = a[j]*a[i] to reduce multiplications.
#[inline]
pub(crate) fn sextic_square<F, R>(a: &[R; 6], res: &mut [R; 6])
where
    F: Field,
    R: Algebra<F>,
{
    // Compute the 11 unreduced coefficients.
    // Diagonal terms: c[2i] += a[i]^2
    // Off-diagonal terms: c[i+j] += 2*a[i]*a[j] for i < j
    let mut c: [R; 11] = array::from_fn(|_| R::ZERO);

    for i in 0..6 {
        c[2 * i] = c[2 * i].clone() + a[i].square();
        for j in (i + 1)..6 {
            let cross = (a[i].clone() * a[j].clone()).double();
            c[i + j] = c[i + j].clone() + cross;
        }
    }

    // Reduce using w^6 = 2w^3 + 2.
    for k in (6..=10).rev() {
        let two_ck = c[k].double();
        c[k - 3] = c[k - 3].clone() + two_ck.clone();
        c[k - 6] = c[k - 6].clone() + two_ck;
    }

    for i in 0..6 {
        res[i] = c[i].clone();
    }
}

// ===== Tower-based inversion =====
//
// The tower construction:
//   F_p^2 = F_p[u]/(u^2 - 3)        (quadratic non-residue: 3)
//   F_p^6 = F_p^2[w]/(w^3 - (u+1))  (cubic non-residue: u+1)
//
// Tower representation: B0 + B1*w + B2*w^2 where Bi = (bi0, bi1) in F_p^2
//
// Conversion between direct [A0..A5] and tower [(a00,a01), (a10,a11), (a20,a21)]:
//   ToTower:   a00 = A0+A3, a01 = A3, a10 = A1+A4, a11 = A4, a20 = A2+A5, a21 = A5
//   FromTower: A0 = a00-a01, A1 = a10-a11, A2 = a20-a21, A3 = a01, A4 = a11, A5 = a21

/// F_p^2 multiplication: (a0 + a1*u)(b0 + b1*u) where u^2 = 3
/// Uses Karatsuba: 3 base muls instead of 4.
#[inline]
fn fp2_mul<F: Field>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    let t0 = a0 * b0;
    let t1 = a1 * b1;
    let t2 = (a0 + a1) * (b0 + b1);
    // real part: a0*b0 + 3*a1*b1
    let real = t0 + t1.double() + t1;
    // imag part: (a0+a1)(b0+b1) - a0*b0 - a1*b1
    let imag = t2 - t0 - t1;
    (real, imag)
}

/// F_p^2 squaring: (a0 + a1*u)^2 where u^2 = 3
#[inline]
fn fp2_square<F: Field>(a0: F, a1: F) -> (F, F) {
    let t0 = a0 * a1;
    // (a0 + a1*u)^2 = a0^2 + 3*a1^2 + 2*a0*a1*u
    let a1_sq = a1.square();
    let real = a0.square() + a1_sq.double() + a1_sq;
    let imag = t0.double();
    (real, imag)
}

/// F_p^2 inversion: 1/(a0 + a1*u) = (a0 - a1*u)/(a0^2 - 3*a1^2)
#[inline]
fn fp2_inv<F: Field>(a0: F, a1: F) -> (F, F) {
    let a1_sq = a1.square();
    let norm = a0.square() - a1_sq.double() - a1_sq; // a0^2 - 3*a1^2
    let norm_inv = norm.inverse();
    (a0 * norm_inv, -a1 * norm_inv)
}

/// F_p^2 addition
#[inline]
fn fp2_add<F: Field>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    (a0 + b0, a1 + b1)
}

/// F_p^2 subtraction
#[inline]
fn fp2_sub<F: Field>(a0: F, a1: F, b0: F, b1: F) -> (F, F) {
    (a0 - b0, a1 - b1)
}

/// Multiply F_p^2 element by the cubic non-residue β = (1 + u).
/// (1+u)(a0 + a1*u) = (a0 + 3*a1) + (a0 + a1)*u
#[inline]
fn mul_by_cubic_nonresidue<F: Field>(a0: F, a1: F) -> (F, F) {
    let imag = a0 + a1;
    let real = a1.double() + imag; // a0 + 3*a1 = a0 + a1 + 2*a1
    (real, imag)
}

/// Convert direct representation to tower representation.
/// ToTower: a00 = A0+A3, a01 = A3, a10 = A1+A4, a11 = A4, a20 = A2+A5, a21 = A5
#[inline]
fn to_tower<F: Field>(x: &[F; 6]) -> [(F, F); 3] {
    [
        (x[0] + x[3], x[3]),
        (x[1] + x[4], x[4]),
        (x[2] + x[5], x[5]),
    ]
}

/// Convert tower representation to direct representation.
/// FromTower: A0 = a00-a01, A1 = a10-a11, A2 = a20-a21, A3 = a01, A4 = a11, A5 = a21
#[inline]
fn from_tower<F: Field>(t: &[(F, F); 3]) -> [F; 6] {
    [
        t[0].0 - t[0].1,
        t[1].0 - t[1].1,
        t[2].0 - t[2].1,
        t[0].1,
        t[1].1,
        t[2].1,
    ]
}

/// Chung-Hasan Algorithm 17: Inversion in F_p^6 = F_p^2[w]/(w^3 - β)
/// where β = u + 1 is the cubic non-residue.
///
/// Input: x = (x0, x1, x2) where xi in F_p^2
/// Output: x^{-1} = (z0, z1, z2)
#[inline]
fn tower_inv<F: Field>(x: &[(F, F); 3]) -> [(F, F); 3] {
    let (x00, x01) = (x[0].0, x[0].1);
    let (x10, x11) = (x[1].0, x[1].1);
    let (x20, x21) = (x[2].0, x[2].1);

    // t0 = x0^2
    let (t00, t01) = fp2_square(x00, x01);
    // t1 = x1^2
    let (t10, t11) = fp2_square(x10, x11);
    // t2 = x2^2
    let (t20, t21) = fp2_square(x20, x21);
    // t3 = x0 * x1
    let (t30, t31) = fp2_mul(x00, x01, x10, x11);
    // t4 = x0 * x2
    let (t40, t41) = fp2_mul(x00, x01, x20, x21);
    // t5 = x1 * x2
    let (t50, t51) = fp2_mul(x10, x11, x20, x21);

    // c0 = t0 - β*t5
    let (bt50, bt51) = mul_by_cubic_nonresidue(t50, t51);
    let (c00, c01) = fp2_sub(t00, t01, bt50, bt51);

    // c1 = β*t2 - t3
    let (bt20, bt21) = mul_by_cubic_nonresidue(t20, t21);
    let (c10, c11) = fp2_sub(bt20, bt21, t30, t31);

    // c2 = t1 - t4
    let (c20, c21) = fp2_sub(t10, t11, t40, t41);

    // t6 = x0*c0 + β*(x2*c1 + x1*c2)
    let (xc00, xc01) = fp2_mul(x00, x01, c00, c01);
    let (xc10, xc11) = fp2_mul(x20, x21, c10, c11);
    let (xc20, xc21) = fp2_mul(x10, x11, c20, c21);
    let (s0, s1) = fp2_add(xc10, xc11, xc20, xc21);
    let (bs0, bs1) = mul_by_cubic_nonresidue(s0, s1);
    let (t60, t61) = fp2_add(xc00, xc01, bs0, bs1);

    // t6 = t6^{-1}
    let (t6_inv0, t6_inv1) = fp2_inv(t60, t61);

    // z0 = c0 * t6^{-1}
    let (z00, z01) = fp2_mul(c00, c01, t6_inv0, t6_inv1);
    // z1 = c1 * t6^{-1}
    let (z10, z11) = fp2_mul(c10, c11, t6_inv0, t6_inv1);
    // z2 = c2 * t6^{-1}
    let (z20, z21) = fp2_mul(c20, c21, t6_inv0, t6_inv1);

    [(z00, z01), (z10, z11), (z20, z21)]
}

/// Compute the inverse of an element in the sextic extension field.
/// Uses the tower representation for efficient inversion.
#[inline]
fn sextic_inv<F: SexticExtendable>(a: &SexticExtensionField<F>) -> SexticExtensionField<F> {
    // Convert to tower representation
    let tower = to_tower(&a.value);
    // Invert in tower
    let tower_inv = tower_inv(&tower);
    // Convert back to direct representation
    SexticExtensionField::new(from_tower(&tower_inv))
}

// ===== Naive multiplication for testing =====

/// Naive schoolbook multiplication for cross-checking.
/// This computes the same result as `sextic_mul` but in a more straightforward way.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn sextic_mul_naive<F: Field>(a: &[F; 6], b: &[F; 6]) -> [F; 6] {
    let mut d = [F::ZERO; 11];
    for i in 0..6 {
        for j in 0..6 {
            d[i + j] += a[i] * b[j];
        }
    }

    // Reduce: w^6 = 2w^3 + 2
    for k in (6..=10).rev() {
        let two_dk = d[k].double();
        d[k - 3] += two_dk;
        d[k - 6] += two_dk;
    }

    [d[0], d[1], d[2], d[3], d[4], d[5]]
}
