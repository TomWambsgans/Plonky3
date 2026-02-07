use p3_field::{Algebra, Field, PrimeCharacteristicRing, packed_mod_add, packed_mod_sub};
use p3_monty_31::{MontyParameters, base_mul_packed, monty_add, monty_sub};

use crate::sextic_extension::extension::SexticExtensionField;
use crate::sextic_extension::packed_extension::PackedSexticExtensionField;
use crate::sextic_extension::packing::sextic_mul_packed;
use crate::{KoalaBear, KoalaBearParameters};

pub(crate) mod extension;
pub(crate) mod packed_extension;
pub(crate) mod packing;

pub type SexticExtensionFieldKB = SexticExtensionField<KoalaBear>;
pub type PackedSexticExtensionFieldKB =
    PackedSexticExtensionField<KoalaBear, <KoalaBear as Field>::Packing>;

impl SexticExtendable for KoalaBear {
    // Frobenius matrix: FROBENIUS_MATRIX[i][j] = coefficient j of w^((i+1)*p)
    const FROBENIUS_MATRIX: [[Self; 6]; 5] = [
        [
            Self::new(0),
            Self::new(0),
            Self::new(178695960),
            Self::new(0),
            Self::new(0),
            Self::new(285003076),
        ],
        [
            Self::new(0),
            Self::new(1652631425),
            Self::new(0),
            Self::new(0),
            Self::new(283040797),
            Self::new(0),
        ],
        [
            Self::new(2),
            Self::new(0),
            Self::new(0),
            Self::new(2130706432),
            Self::new(0),
            Self::new(0),
        ],
        [
            Self::new(0),
            Self::new(0),
            Self::new(1918092201),
            Self::new(0),
            Self::new(0),
            Self::new(1952010473),
        ],
        [
            Self::new(0),
            Self::new(608474823),
            Self::new(0),
            Self::new(0),
            Self::new(478075008),
            Self::new(0),
        ],
    ];

    const EXT_GENERATOR: [Self; 6] = Self::new_array([3, 1, 1, 0, 0, 0]);

    const TWO_ADIC_SEXTIC_GENERATOR: [Self; 6] =
        Self::new_array([1759267465, 0, 0, 371438968, 0, 0]);
}

impl SexticExtendableAlgebra<KoalaBear> for KoalaBear {
    #[inline(always)]
    fn sextic_mul(a: &[Self; 6], b: &[Self; 6], res: &mut [Self; 6]) {
        sextic_mul_packed(a, b, res);
    }

    #[inline(always)]
    fn sextic_add(a: &[Self; 6], b: &[Self; 6]) -> [Self; 6] {
        let mut res = [Self::ZERO; 6];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 6] = &*(a.as_ptr() as *const [u32; 6]);
            let b: &[u32; 6] = &*(b.as_ptr() as *const [u32; 6]);
            let res: &mut [u32; 6] = &mut *(res.as_mut_ptr() as *mut [u32; 6]);

            packed_mod_add(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                monty_add::<KoalaBearParameters>,
            );
        }
        res
    }

    #[inline(always)]
    fn sextic_sub(a: &[Self; 6], b: &[Self; 6]) -> [Self; 6] {
        let mut res = [Self::ZERO; 6];
        unsafe {
            // Safe as Self is repr(transparent) and stores a single u32.
            let a: &[u32; 6] = &*(a.as_ptr() as *const [u32; 6]);
            let b: &[u32; 6] = &*(b.as_ptr() as *const [u32; 6]);
            let res: &mut [u32; 6] = &mut *(res.as_mut_ptr() as *mut [u32; 6]);

            packed_mod_sub(
                a,
                b,
                res,
                KoalaBearParameters::PRIME,
                monty_sub::<KoalaBearParameters>,
            );
        }
        res
    }

    #[inline(always)]
    fn sextic_base_mul(lhs: [Self; 6], rhs: Self) -> [Self; 6] {
        let mut res = [Self::ZERO; 6];
        base_mul_packed(lhs, rhs, &mut res);
        res
    }
}

/// Trait for fields that support sextic extension: F[w]/(w^6 - 2w^3 - 2)
pub trait SexticExtendable: Field + SexticExtendableAlgebra<Self> {
    const FROBENIUS_MATRIX: [[Self; 6]; 5];

    /// A generator for the extension field, expressed as 6 base field coefficients.
    const EXT_GENERATOR: [Self; 6];

    /// Generator of order 2^(TWO_ADICITY+1) in the sextic extension.
    const TWO_ADIC_SEXTIC_GENERATOR: [Self; 6];
}

pub trait SexticExtendableAlgebra<F: Field>: Algebra<F> {
    /// Multiplication in the algebra extension ring A[w] / (w^6 - 2w^3 - 2).
    fn sextic_mul(a: &[Self; 6], b: &[Self; 6], res: &mut [Self; 6]);

    /// Addition in the sextic extension ring.
    #[must_use]
    fn sextic_add(a: &[Self; 6], b: &[Self; 6]) -> [Self; 6];

    /// Subtraction in the sextic extension ring.
    #[must_use]
    fn sextic_sub(a: &[Self; 6], b: &[Self; 6]) -> [Self; 6];

    fn sextic_base_mul(lhs: [Self; 6], rhs: Self) -> [Self; 6];
}
