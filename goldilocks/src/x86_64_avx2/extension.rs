//! AVX2-optimized cubic extension field arithmetic for Goldilocks.
//!
//! The lane-wise SIMD ops (`+ - * square`) on `PackedGoldilocksAVX2` are
//! already efficient. The wins versus the default `trinomial_cubic_mul`
//! come from rearranging the result-stage tree to minimise dep-chain depth
//! (the bench is latency-bound on a wide pipeline) and from routing the
//! diagonal `a_i^2` terms through `square()` rather than `mul()`.

use p3_field::PrimeCharacteristicRing;
use p3_field::extension::CubicExtendableAlgebra;

use super::PackedGoldilocksAVX2;
use crate::Goldilocks;

impl CubicExtendableAlgebra<Goldilocks> for PackedGoldilocksAVX2 {
    /// Karatsuba: 6 SIMD muls. Output combines arranged for chain depth 3
    /// (vs depth 6 in the default `c3`-shared formulation).
    #[inline]
    fn cubic_mul(a: &[Self; 3], b: &[Self; 3], res: &mut [Self; 3]) {
        let m0 = a[0] * b[0];
        let m1 = a[1] * b[1];
        let m2 = a[2] * b[2];

        let p01 = (a[0] + a[1]) * (b[0] + b[1]);
        let p02 = (a[0] + a[2]) * (b[0] + b[2]);
        let p12 = (a[1] + a[2]) * (b[1] + b[2]);

        // Reduction (X^3 = X + 1, X^4 = X^2 + X) yields:
        //   r0 = m0 + p12 - m1 - m2
        //   r1 = p01 + p12 - m0 - 2*m1
        //   r2 = p02 - m0 + m1
        // Trees below are arranged for minimum dep-chain depth.

        // (p12 - m1) is shared between r0 and r1.
        let p12_m1 = p12 - m1;

        // r0 = (p12 - m1) + (m0 - m2): chain depth 2
        res[0] = p12_m1 + (m0 - m2);
        // r1 = (p01 - m1) + (p12 - m1) - m0: chain depth 3
        res[1] = ((p01 - m1) + p12_m1) - m0;
        // r2 = (p02 - m0) + m1: chain depth 2
        res[2] = (p02 - m0) + m1;
    }

    /// 3 squares + 3 muls. Default routes one of the squares through
    /// `dot_product` which calls plain mul; we use `square()` instead so
    /// every diagonal term hits the faster `square64` SIMD routine.
    #[inline]
    fn cubic_square(a: &[Self; 3], res: &mut [Self; 3]) {
        let a0sq = a[0].square();
        let a1sq = a[1].square();
        let a2sq = a[2].square();
        let a0a1 = a[0] * a[1];
        let a0a2 = a[0] * a[2];
        let a1a2 = a[1] * a[2];

        //   r0 = a0^2 + 2*a1*a2
        //   r1 = 2*a0*a1 + 2*a1*a2 + a2^2
        //   r2 = 2*a0*a2 + a1^2 + a2^2

        // 2*a1*a2 is shared between r0 and r1.
        let dbl_a1a2 = a1a2 + a1a2;

        res[0] = a0sq + dbl_a1a2;
        res[1] = (a0a1 + a0a1) + (dbl_a1a2 + a2sq);
        res[2] = (a0a2 + a0a2) + (a1sq + a2sq);
    }
}
