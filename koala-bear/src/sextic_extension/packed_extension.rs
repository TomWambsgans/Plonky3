use alloc::vec::Vec;
use core::array;
use core::fmt::Debug;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use itertools::Itertools;
use p3_field::extension::{vector_add, vector_sub};
use p3_field::{
    Algebra, BasedVectorSpace, Field, PackedField, PackedFieldExtension, PackedValue, Powers,
    PrimeCharacteristicRing, field_to_array,
};
use p3_util::{flatten_to_base, reconstitute_from_base};
use rand::distr::{Distribution, StandardUniform};
use serde::{Deserialize, Serialize};

use super::extension::SexticExtensionField;
use crate::SexticExtendable;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize, PartialOrd, Ord)]
#[repr(transparent)] // Needed to make various casts safe.
pub struct PackedSexticExtensionField<F: Field, PF: PackedField<Scalar = F>> {
    #[serde(
        with = "p3_util::array_serialization",
        bound(serialize = "PF: Serialize", deserialize = "PF: Deserialize<'de>")
    )]
    pub(crate) value: [PF; 6],
}

impl<F: Field, PF: PackedField<Scalar = F>> PackedSexticExtensionField<F, PF> {
    const fn new(value: [PF; 6]) -> Self {
        Self { value }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Default for PackedSexticExtensionField<F, PF> {
    #[inline]
    fn default() -> Self {
        Self {
            value: array::from_fn(|_| PF::ZERO),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<SexticExtensionField<F>>
    for PackedSexticExtensionField<F, PF>
{
    #[inline]
    fn from(x: SexticExtensionField<F>) -> Self {
        Self {
            value: x.value.map(Into::into),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> From<PF> for PackedSexticExtensionField<F, PF> {
    #[inline]
    fn from(x: PF) -> Self {
        Self {
            value: field_to_array(x),
        }
    }
}

impl<F: Field, PF: PackedField<Scalar = F>> Distribution<PackedSexticExtensionField<F, PF>>
    for StandardUniform
where
    Self: Distribution<PF>,
{
    #[inline]
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> PackedSexticExtensionField<F, PF> {
        PackedSexticExtensionField::new(array::from_fn(|_| self.sample(rng)))
    }
}

impl<F: SexticExtendable, PF: PackedField<Scalar = F>> Algebra<SexticExtensionField<F>>
    for PackedSexticExtensionField<F, PF>
{
}

impl<F: SexticExtendable, PF: PackedField<Scalar = F>> Algebra<PF>
    for PackedSexticExtensionField<F, PF>
{
}

impl<F, PF> PrimeCharacteristicRing for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type PrimeSubfield = PF::PrimeSubfield;

    const ZERO: Self = Self {
        value: [PF::ZERO; 6],
    };

    const ONE: Self = Self {
        value: field_to_array(PF::ONE),
    };

    const TWO: Self = Self {
        value: field_to_array(PF::TWO),
    };

    const NEG_ONE: Self = Self {
        value: field_to_array(PF::NEG_ONE),
    };

    #[inline]
    fn from_prime_subfield(val: Self::PrimeSubfield) -> Self {
        PF::from_prime_subfield(val).into()
    }

    #[inline]
    fn from_bool(b: bool) -> Self {
        PF::from_bool(b).into()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        let mut res = Self::default();
        super::extension::sextic_square(&self.value, &mut res.value);
        res
    }

    #[inline]
    fn zero_vec(len: usize) -> Vec<Self> {
        // SAFETY: this is a repr(transparent) wrapper around an array.
        unsafe { reconstitute_from_base(PF::zero_vec(len * 6)) }
    }
}

impl<F, PF> BasedVectorSpace<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    const DIMENSION: usize = 6;

    #[inline]
    fn as_basis_coefficients_slice(&self) -> &[PF] {
        &self.value
    }

    #[inline]
    fn from_basis_coefficients_fn<Fn: FnMut(usize) -> PF>(f: Fn) -> Self {
        Self {
            value: array::from_fn(f),
        }
    }

    #[inline]
    fn from_basis_coefficients_iter<I: ExactSizeIterator<Item = PF>>(mut iter: I) -> Option<Self> {
        (iter.len() == 6).then(|| Self::new(array::from_fn(|_| iter.next().unwrap())))
    }

    #[inline]
    fn flatten_to_base(vec: Vec<Self>) -> Vec<PF> {
        unsafe {
            // Safety: Self is repr(transparent), stored identically to [PF; 6].
            flatten_to_base(vec)
        }
    }

    #[inline]
    fn reconstitute_from_base(vec: Vec<PF>) -> Vec<Self> {
        unsafe {
            // Safety: Self is repr(transparent), stored identically to [PF; 6].
            reconstitute_from_base(vec)
        }
    }
}

impl<F> PackedFieldExtension<F, SexticExtensionField<F>>
    for PackedSexticExtensionField<F, F::Packing>
where
    F: SexticExtendable,
{
    #[inline]
    fn from_ext_slice(ext_slice: &[SexticExtensionField<F>]) -> Self {
        let width = F::Packing::WIDTH;
        assert_eq!(ext_slice.len(), width);

        let res = array::from_fn(|i| F::Packing::from_fn(|j| ext_slice[j].value[i]));
        Self::new(res)
    }

    #[inline]
    fn to_ext_iter(
        iter: impl IntoIterator<Item = Self>,
    ) -> impl Iterator<Item = SexticExtensionField<F>> {
        let width = F::Packing::WIDTH;
        iter.into_iter().flat_map(move |x| {
            (0..width).map(move |i| {
                let values = array::from_fn(|j| x.value[j].as_slice()[i]);
                SexticExtensionField::new(values)
            })
        })
    }

    #[inline]
    fn packed_ext_powers(base: SexticExtensionField<F>) -> p3_field::Powers<Self> {
        let width = F::Packing::WIDTH;
        let powers = base.powers().take(width + 1).collect_vec();
        // Transpose first WIDTH powers
        let current = Self::from_ext_slice(&powers[..width]);

        // Broadcast self^WIDTH
        let multiplier = powers[width].into();

        Powers {
            base: multiplier,
            current,
        }
    }
}

impl<F, PF> Neg for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self {
            value: self.value.map(PF::neg),
        }
    }
}

impl<F, PF> Add for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Add<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: SexticExtensionField<F>) -> Self {
        let value = vector_add(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Add<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: PF) -> Self {
        self.value[0] += rhs;
        self
    }
}

impl<F, PF> AddAssign for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for i in 0..6 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: SexticExtensionField<F>) {
        for i in 0..6 {
            self.value[i] += rhs.value[i];
        }
    }
}

impl<F, PF> AddAssign<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn add_assign(&mut self, rhs: PF) {
        self.value[0] += rhs;
    }
}

impl<F, PF> Sum for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc + x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> Sub for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Sub<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: SexticExtensionField<F>) -> Self {
        let value = vector_sub(&self.value, &rhs.value);
        Self { value }
    }
}

impl<F, PF> Sub<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: PF) -> Self {
        let mut res = self.value;
        res[0] -= rhs;
        Self { value: res }
    }
}

impl<F, PF> SubAssign for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: SexticExtensionField<F>) {
        *self = *self - rhs;
    }
}

impl<F, PF> SubAssign<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn sub_assign(&mut self, rhs: PF) {
        *self = *self - rhs;
    }
}

impl<F, PF> Mul for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        super::extension::sextic_mul::<F, PF, PF>(&a, &b, &mut res.value);
        res
    }
}

impl<F, PF> Mul<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: SexticExtensionField<F>) -> Self {
        let a = self.value;
        let b = rhs.value;
        let mut res = Self::default();
        super::extension::sextic_mul(&a, &b, &mut res.value);

        res
    }
}

impl<F, PF> Mul<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: PF) -> Self {
        Self {
            value: self.value.map(|x| x * rhs),
        }
    }
}

impl<F, PF> Product for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|acc, x| acc * x).unwrap_or(Self::ZERO)
    }
}

impl<F, PF> MulAssign for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<SexticExtensionField<F>> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: SexticExtensionField<F>) {
        *self = *self * rhs;
    }
}

impl<F, PF> MulAssign<PF> for PackedSexticExtensionField<F, PF>
where
    F: SexticExtendable,
    PF: PackedField<Scalar = F>,
{
    #[inline]
    fn mul_assign(&mut self, rhs: PF) {
        *self = *self * rhs;
    }
}
