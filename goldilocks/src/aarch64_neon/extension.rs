//! NEON-optimized cubic extension field arithmetic for Goldilocks.
//!
//! Key optimization: operates entirely in scalar `u64` space for additions
//! and subtractions (avoiding the expensive NEON vector canonicalize/shift
//! pipeline), while using the efficient interleaved dual-lane ASM for
//! multiplications. The Karatsuba algorithm reduces multiplications from 9 to 6.

use p3_field::extension::CubicExtendableAlgebra;

use super::packing::mul_reduce_dual_asm;
use super::PackedGoldilocksNeon;
use crate::{Goldilocks, P};

const EPSILON: u64 = P.wrapping_neg(); // 2^32 - 1

/// Goldilocks scalar addition: (a + b) mod P.
/// Handles u64 overflow via EPSILON correction.
#[inline(always)]
const fn gadd(a: u64, b: u64) -> u64 {
    let (sum, overflow) = a.overflowing_add(b);
    let (res, _) = sum.overflowing_add(if overflow { EPSILON } else { 0 });
    res
}

/// Goldilocks scalar subtraction: (a - b) mod P.
/// Handles u64 underflow via EPSILON correction.
#[inline(always)]
const fn gsub(a: u64, b: u64) -> u64 {
    let (diff, borrow) = a.overflowing_sub(b);
    let (res, _) = diff.overflowing_sub(if borrow { EPSILON } else { 0 });
    res
}

impl CubicExtendableAlgebra<Goldilocks> for PackedGoldilocksNeon {
    /// Karatsuba multiplication operating in scalar u64 space.
    ///
    /// Each `PackedGoldilocksNeon` packs 2 independent Goldilocks elements.
    /// This function:
    /// 1. Extracts all 12 scalar values (6 inputs × 2 lanes)
    /// 2. Computes 6 Karatsuba sums using fast scalar add (~3 instr each)
    /// 3. Performs 6 interleaved dual-lane multiplications via ASM
    /// 4. Reduces using fast scalar add/sub
    /// 5. Packs the 6 output scalars back into 3 vectors
    ///
    /// This bypasses the NEON add/sub pipeline (which requires canonicalize +
    /// shift + signed-compare per operation, ~11 instructions each) in favor
    /// of simple scalar overflowing_add/sub with EPSILON correction (~3 instr).
    #[inline(always)]
    fn cubic_mul(a: &[Self; 3], b: &[Self; 3], res: &mut [Self; 3]) {
        unsafe {
            // Read scalar lanes directly from `[Goldilocks; 2]` storage to
            // skip the umov NEON->GPR detour.
            let a00 = a[0].0[0].value;
            let a01 = a[0].0[1].value;
            let a10 = a[1].0[0].value;
            let a11 = a[1].0[1].value;
            let a20 = a[2].0[0].value;
            let a21 = a[2].0[1].value;
            let b00 = b[0].0[0].value;
            let b01 = b[0].0[1].value;
            let b10 = b[1].0[0].value;
            let b11 = b[1].0[1].value;
            let b20 = b[2].0[0].value;
            let b21 = b[2].0[1].value;

            // Karatsuba sums (scalar add, ~3 instr each).
            let sa01_0 = gadd(a00, a10);
            let sa01_1 = gadd(a01, a11);
            let sa02_0 = gadd(a00, a20);
            let sa02_1 = gadd(a01, a21);
            let sa12_0 = gadd(a10, a20);
            let sa12_1 = gadd(a11, a21);
            let sb01_0 = gadd(b00, b10);
            let sb01_1 = gadd(b01, b11);
            let sb02_0 = gadd(b00, b20);
            let sb02_1 = gadd(b01, b21);
            let sb12_0 = gadd(b10, b20);
            let sb12_1 = gadd(b11, b21);

            // 6 Karatsuba multiplications via interleaved dual-lane ASM.
            // Each call processes both lanes simultaneously for ILP.
            let (m0_0, m0_1) = mul_reduce_dual_asm(a00, b00, a01, b01);
            let (m1_0, m1_1) = mul_reduce_dual_asm(a10, b10, a11, b11);
            let (m2_0, m2_1) = mul_reduce_dual_asm(a20, b20, a21, b21);
            let (t01_0, t01_1) = mul_reduce_dual_asm(sa01_0, sb01_0, sa01_1, sb01_1);
            let (t02_0, t02_1) = mul_reduce_dual_asm(sa02_0, sb02_0, sa02_1, sb02_1);
            let (t12_0, t12_1) = mul_reduce_dual_asm(sa12_0, sb12_0, sa12_1, sb12_1);

            // Reduction: X^3 = X + 1, X^4 = X^2 + X
            // c3 = t12 - m1 - m2
            let c3_0 = gsub(gsub(t12_0, m1_0), m2_0);
            let c3_1 = gsub(gsub(t12_1, m1_1), m2_1);

            // r0 = m0 + c3
            let r0_0 = gadd(m0_0, c3_0);
            let r0_1 = gadd(m0_1, c3_1);

            // r1 = t01 + c3 + m2 - m0 - m1
            let r1_0 = gsub(gsub(gadd(gadd(t01_0, c3_0), m2_0), m0_0), m1_0);
            let r1_1 = gsub(gsub(gadd(gadd(t01_1, c3_1), m2_1), m0_1), m1_1);

            // r2 = t02 - m0 + m1
            let r2_0 = gadd(gsub(t02_0, m0_0), m1_0);
            let r2_1 = gadd(gsub(t02_1, m0_1), m1_1);

            // Write straight into `[Goldilocks; 2]` storage to skip the
            // ins GPR->NEON detour.
            res[0] = Self([Goldilocks::new(r0_0), Goldilocks::new(r0_1)]);
            res[1] = Self([Goldilocks::new(r1_0), Goldilocks::new(r1_1)]);
            res[2] = Self([Goldilocks::new(r2_0), Goldilocks::new(r2_1)]);
        }
    }

    /// Coefficient-wise add in scalar u64 space.
    ///
    /// Reaches into the `[Goldilocks; 2]` storage directly so the compiler can
    /// keep all 12 input lanes in GPRs and avoid the NEON↔GPR umov/ins bounce
    /// that the generic `vector_add` path produces here.
    #[inline(always)]
    fn cubic_add(a: &[Self; 3], b: &[Self; 3]) -> [Self; 3] {
        let r0_0 = gadd(a[0].0[0].value, b[0].0[0].value);
        let r0_1 = gadd(a[0].0[1].value, b[0].0[1].value);
        let r1_0 = gadd(a[1].0[0].value, b[1].0[0].value);
        let r1_1 = gadd(a[1].0[1].value, b[1].0[1].value);
        let r2_0 = gadd(a[2].0[0].value, b[2].0[0].value);
        let r2_1 = gadd(a[2].0[1].value, b[2].0[1].value);
        [
            Self([Goldilocks::new(r0_0), Goldilocks::new(r0_1)]),
            Self([Goldilocks::new(r1_0), Goldilocks::new(r1_1)]),
            Self([Goldilocks::new(r2_0), Goldilocks::new(r2_1)]),
        ]
    }

    /// Coefficient-wise sub in scalar u64 space (mirror of `cubic_add`).
    #[inline(always)]
    fn cubic_sub(a: &[Self; 3], b: &[Self; 3]) -> [Self; 3] {
        let r0_0 = gsub(a[0].0[0].value, b[0].0[0].value);
        let r0_1 = gsub(a[0].0[1].value, b[0].0[1].value);
        let r1_0 = gsub(a[1].0[0].value, b[1].0[0].value);
        let r1_1 = gsub(a[1].0[1].value, b[1].0[1].value);
        let r2_0 = gsub(a[2].0[0].value, b[2].0[0].value);
        let r2_1 = gsub(a[2].0[1].value, b[2].0[1].value);
        [
            Self([Goldilocks::new(r0_0), Goldilocks::new(r0_1)]),
            Self([Goldilocks::new(r1_0), Goldilocks::new(r1_1)]),
            Self([Goldilocks::new(r2_0), Goldilocks::new(r2_1)]),
        ]
    }

    /// Squaring in scalar u64 space.
    ///
    /// Uses 3 squares + 3 multiplications with scalar add/sub for reduction.
    #[inline(always)]
    fn cubic_square(a: &[Self; 3], res: &mut [Self; 3]) {
        unsafe {
            // Read scalar lanes directly from storage (no umov).
            let a00 = a[0].0[0].value;
            let a01 = a[0].0[1].value;
            let a10 = a[1].0[0].value;
            let a11 = a[1].0[1].value;
            let a20 = a[2].0[0].value;
            let a21 = a[2].0[1].value;

            // a0^2, a1^2, a2^2
            let (a0sq_0, a0sq_1) = mul_reduce_dual_asm(a00, a00, a01, a01);
            let (a1sq_0, a1sq_1) = mul_reduce_dual_asm(a10, a10, a11, a11);
            let (a2sq_0, a2sq_1) = mul_reduce_dual_asm(a20, a20, a21, a21);

            // Cross products
            let (a0a1_0, a0a1_1) = mul_reduce_dual_asm(a00, a10, a01, a11);
            let (a0a2_0, a0a2_1) = mul_reduce_dual_asm(a00, a20, a01, a21);
            let (a1a2_0, a1a2_1) = mul_reduce_dual_asm(a10, a20, a11, a21);

            // Reduction: X^3 = X + 1, X^4 = X^2 + X
            // r0 = a0^2 + 2*a1*a2
            let r0_0 = gadd(a0sq_0, gadd(a1a2_0, a1a2_0));
            let r0_1 = gadd(a0sq_1, gadd(a1a2_1, a1a2_1));

            // r1 = 2*a0*a1 + 2*a1*a2 + a2^2
            let r1_0 = gadd(gadd(gadd(a0a1_0, a0a1_0), gadd(a1a2_0, a1a2_0)), a2sq_0);
            let r1_1 = gadd(gadd(gadd(a0a1_1, a0a1_1), gadd(a1a2_1, a1a2_1)), a2sq_1);

            // r2 = 2*a0*a2 + a1^2 + a2^2
            let r2_0 = gadd(gadd(gadd(a0a2_0, a0a2_0), a1sq_0), a2sq_0);
            let r2_1 = gadd(gadd(gadd(a0a2_1, a0a2_1), a1sq_1), a2sq_1);

            res[0] = Self([Goldilocks::new(r0_0), Goldilocks::new(r0_1)]);
            res[1] = Self([Goldilocks::new(r1_0), Goldilocks::new(r1_1)]);
            res[2] = Self([Goldilocks::new(r2_0), Goldilocks::new(r2_1)]);
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_field::extension::{CubicExtendableAlgebra, CubicTrinomialExtensionField};
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;

    type EF = CubicTrinomialExtensionField<Goldilocks>;

    fn pack(lane0: [Goldilocks; 3], lane1: [Goldilocks; 3]) -> [PackedGoldilocksNeon; 3] {
        core::array::from_fn(|i| PackedGoldilocksNeon([lane0[i], lane1[i]]))
    }

    fn unpack(p: &[PackedGoldilocksNeon; 3]) -> ([Goldilocks; 3], [Goldilocks; 3]) {
        (
            [p[0].0[0], p[1].0[0], p[2].0[0]],
            [p[0].0[1], p[1].0[1], p[2].0[1]],
        )
    }

    fn ext_to_array(x: EF) -> [Goldilocks; 3] {
        let s = x.as_basis_coefficients_slice();
        [s[0], s[1], s[2]]
    }

    fn check_mul(a0: [Goldilocks; 3], a1: [Goldilocks; 3], b0: [Goldilocks; 3], b1: [Goldilocks; 3]) {
        let a = pack(a0, a1);
        let b = pack(b0, b1);
        let mut got = [PackedGoldilocksNeon::ZERO; 3];
        PackedGoldilocksNeon::cubic_mul(&a, &b, &mut got);

        let want0 = ext_to_array(EF::new(a0) * EF::new(b0));
        let want1 = ext_to_array(EF::new(a1) * EF::new(b1));
        let (g0, g1) = unpack(&got);
        assert_eq!(g0, want0, "cubic_mul lane 0 mismatch");
        assert_eq!(g1, want1, "cubic_mul lane 1 mismatch");
    }

    fn check_square(a0: [Goldilocks; 3], a1: [Goldilocks; 3]) {
        let a = pack(a0, a1);
        let mut got = [PackedGoldilocksNeon::ZERO; 3];
        PackedGoldilocksNeon::cubic_square(&a, &mut got);

        let want0 = ext_to_array(EF::new(a0).square());
        let want1 = ext_to_array(EF::new(a1).square());
        let (g0, g1) = unpack(&got);
        assert_eq!(g0, want0, "cubic_square lane 0 mismatch");
        assert_eq!(g1, want1, "cubic_square lane 1 mismatch");
    }

    #[test]
    fn cubic_mul_matches_scalar() {
        // A few hand-picked values, including 0, 1, P-1, and the boundary value
        // P = 0xFFFF_FFFF_0000_0001 (which canonicalises to 0).
        let max = Goldilocks::new(P - 1);
        let p_redundant = Goldilocks::new(P);
        let cases: &[([Goldilocks; 3], [Goldilocks; 3], [Goldilocks; 3], [Goldilocks; 3])] = &[
            (
                [Goldilocks::new(3), Goldilocks::new(5), Goldilocks::new(7)],
                [Goldilocks::new(11), Goldilocks::new(13), Goldilocks::new(17)],
                [Goldilocks::new(19), Goldilocks::new(23), Goldilocks::new(29)],
                [Goldilocks::new(31), Goldilocks::new(37), Goldilocks::new(41)],
            ),
            (
                [Goldilocks::ZERO, Goldilocks::ONE, max],
                [max, max, max],
                [Goldilocks::ONE, Goldilocks::ZERO, Goldilocks::ONE],
                [p_redundant, max, Goldilocks::ZERO],
            ),
        ];
        for (a0, a1, b0, b1) in cases {
            check_mul(*a0, *a1, *b0, *b1);
        }

        // Random fuzz.
        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        for _ in 0..1024 {
            let a0 = core::array::from_fn(|_| rng.random());
            let a1 = core::array::from_fn(|_| rng.random());
            let b0 = core::array::from_fn(|_| rng.random());
            let b1 = core::array::from_fn(|_| rng.random());
            check_mul(a0, a1, b0, b1);
        }
    }

    #[test]
    fn cubic_square_matches_scalar() {
        let max = Goldilocks::new(P - 1);
        let p_redundant = Goldilocks::new(P);
        let cases: &[([Goldilocks; 3], [Goldilocks; 3])] = &[
            (
                [Goldilocks::new(3), Goldilocks::new(5), Goldilocks::new(7)],
                [Goldilocks::new(11), Goldilocks::new(13), Goldilocks::new(17)],
            ),
            (
                [Goldilocks::ZERO, Goldilocks::ONE, max],
                [max, p_redundant, Goldilocks::ZERO],
            ),
        ];
        for (a0, a1) in cases {
            check_square(*a0, *a1);
        }

        // Random fuzz.
        let mut rng = SmallRng::seed_from_u64(0xBADC0DE);
        for _ in 0..1024 {
            let a0 = core::array::from_fn(|_| rng.random());
            let a1 = core::array::from_fn(|_| rng.random());
            check_square(a0, a1);
        }
    }

    #[test]
    fn cubic_mul_matches_square() {
        // x*x must equal cubic_square(x).
        let mut rng = SmallRng::seed_from_u64(0xFEEDFACE);
        for _ in 0..256 {
            let a0: [Goldilocks; 3] = core::array::from_fn(|_| rng.random());
            let a1: [Goldilocks; 3] = core::array::from_fn(|_| rng.random());
            let a = pack(a0, a1);
            let mut via_mul = [PackedGoldilocksNeon::ZERO; 3];
            let mut via_sq = [PackedGoldilocksNeon::ZERO; 3];
            PackedGoldilocksNeon::cubic_mul(&a, &a, &mut via_mul);
            PackedGoldilocksNeon::cubic_square(&a, &mut via_sq);
            assert_eq!(via_mul, via_sq);
        }
    }
}
