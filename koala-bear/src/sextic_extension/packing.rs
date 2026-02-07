use crate::KoalaBear;

/// Multiplication in the sextic extension field F_p[w]/(w^6 - 2w^3 - 2).
///
/// Currently uses the generic schoolbook multiplication for all architectures.
/// TODO: Implement Montgomery 6-term Karatsuba for SIMD targets.
#[inline]
pub(crate) fn sextic_mul_packed(
    a: &[KoalaBear; 6],
    b: &[KoalaBear; 6],
    res: &mut [KoalaBear; 6],
) {
    super::extension::sextic_mul(a, b, res);
}
