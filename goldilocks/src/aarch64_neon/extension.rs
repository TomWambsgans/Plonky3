use p3_field::extension::CubicExtendableAlgebra;

use super::PackedGoldilocksNeon;
use crate::Goldilocks;

/// NEON-optimized cubic extension arithmetic for `PackedGoldilocksNeon`.
///
/// Uses Karatsuba multiplication (6 base-field muls instead of 9)
/// operating directly on `PackedGoldilocksNeon` values which pack 2
/// independent Goldilocks field elements per NEON `uint64x2_t` register.
impl CubicExtendableAlgebra<Goldilocks> for PackedGoldilocksNeon {
    #[inline(always)]
    fn cubic_mul(a: &[Self; 3], b: &[Self; 3], res: &mut [Self; 3]) {
        // Karatsuba: 6 multiplications instead of 9.
        // Each multiplication here operates on 2 independent Goldilocks
        // products in parallel via NEON uint64x2_t vectors.
        let m0 = a[0] * b[0];
        let m1 = a[1] * b[1];
        let m2 = a[2] * b[2];

        let t01 = (a[0] + a[1]) * (b[0] + b[1]);
        let t02 = (a[0] + a[2]) * (b[0] + b[2]);
        let t12 = (a[1] + a[2]) * (b[1] + b[2]);

        // c3 = a1*b2 + a2*b1 = t12 - m1 - m2
        let c3 = t12 - m1 - m2;

        // Reduction: X^3 = X + 1, X^4 = X^2 + X
        res[0] = m0 + c3;
        res[1] = t01 + c3 + m2 - m0 - m1;
        res[2] = t02 - m0 + m1;
    }

    #[inline(always)]
    fn cubic_square(a: &[Self; 3], res: &mut [Self; 3]) {
        // Optimized squaring: 3 squares + 3 multiplications.
        // Each operation processes 2 independent Goldilocks squarings in parallel.
        let a0_sq = a[0] * a[0];
        let a2_sq = a[2] * a[2];
        let a1a2_2 = (a[1] + a[1]) * a[2]; // 2*a1*a2

        // r0 = a0^2 + 2*a1*a2
        res[0] = a0_sq + a1a2_2;
        // r1 = 2*a1*(a0 + a2) + a2^2
        res[1] = (a[1] + a[1]) * (a[0] + a[2]) + a2_sq;
        // r2 = 2*a0*a2 + a1^2 + a2^2
        res[2] = (a[0] + a[0]) * a[2] + a[1] * a[1] + a2_sq;
    }
}
