//! NEON-tuned cubic-extension multiplication for Goldilocks under `X^3 - 2`.
//!
//! Operates entirely in scalar `u64` space: each `PackedGoldilocksNeon` packs 2 lanes,
//! so we read both lanes as scalars (skipping the NEON↔GPR `umov`/`ins` bounce),
//! run 6-mul Karatsuba per lane in pure Rust so LLVM can interleave the 12
//! lane-products across each other, then write results back as scalars.

use super::PackedGoldilocksNeon;
use super::packing::{gadd, gdouble, gsub, mul_reduce};
use crate::Goldilocks;

/// Multiplication in `Fp[X]/(X^3 - 2)` on `[PackedGoldilocksNeon; 3]`.
///
/// Reduction: `X^3 = 2`, `X^4 = 2X`. Karatsuba gives 6 base muls per lane (12 total).
#[inline(always)]
pub(crate) fn cubic_packed_mul(
    a: &[PackedGoldilocksNeon; 3],
    b: &[PackedGoldilocksNeon; 3],
    res: &mut [PackedGoldilocksNeon; 3],
) {
    // Pull both lanes into scalar GPRs.
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

    // Karatsuba sums for the cross-product inputs.
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

    // 12 lane-products as plain Rust so the compiler can interleave them.
    let m0_0 = mul_reduce(a00, b00);
    let m0_1 = mul_reduce(a01, b01);
    let m1_0 = mul_reduce(a10, b10);
    let m1_1 = mul_reduce(a11, b11);
    let m2_0 = mul_reduce(a20, b20);
    let m2_1 = mul_reduce(a21, b21);
    let t01_0 = mul_reduce(sa01_0, sb01_0);
    let t01_1 = mul_reduce(sa01_1, sb01_1);
    let t02_0 = mul_reduce(sa02_0, sb02_0);
    let t02_1 = mul_reduce(sa02_1, sb02_1);
    let t12_0 = mul_reduce(sa12_0, sb12_0);
    let t12_1 = mul_reduce(sa12_1, sb12_1);

    // Reduction with X^3 = 2, X^4 = 2X:
    //   r0 = m0 + 2*(t12 - m1 - m2)
    //   r1 = (t01 - m0 - m1) + 2*m2
    //   r2 = (t02 - m0 - m2) + m1

    // r0: q = (t12 - m1) - m2; r0 = m0 + q + q. Chain depth ~4.
    let q_0 = gsub(gsub(t12_0, m1_0), m2_0);
    let q_1 = gsub(gsub(t12_1, m1_1), m2_1);
    let r0_0 = gadd(m0_0, gdouble(q_0));
    let r0_1 = gadd(m0_1, gdouble(q_1));

    // r1: A = (t01 - m0) - m1; r1 = A + 2*m2. Chain depth 3.
    let r1_0 = gadd(gsub(gsub(t01_0, m0_0), m1_0), gdouble(m2_0));
    let r1_1 = gadd(gsub(gsub(t01_1, m0_1), m1_1), gdouble(m2_1));

    // r2: r2 = (t02 - m0) + (m1 - m2). Chain depth 2.
    let r2_0 = gadd(gsub(t02_0, m0_0), gsub(m1_0, m2_0));
    let r2_1 = gadd(gsub(t02_1, m0_1), gsub(m1_1, m2_1));

    res[0] = PackedGoldilocksNeon([Goldilocks::new(r0_0), Goldilocks::new(r0_1)]);
    res[1] = PackedGoldilocksNeon([Goldilocks::new(r1_0), Goldilocks::new(r1_1)]);
    res[2] = PackedGoldilocksNeon([Goldilocks::new(r2_0), Goldilocks::new(r2_1)]);
}

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;
    use p3_field::extension::BinomialExtensionField;
    use rand::rngs::SmallRng;
    use rand::{RngExt, SeedableRng};

    use super::*;
    use crate::P;

    type EF = BinomialExtensionField<Goldilocks, 3>;

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
        use p3_field::BasedVectorSpace;
        let s = x.as_basis_coefficients_slice();
        [s[0], s[1], s[2]]
    }

    fn check_mul(
        a0: [Goldilocks; 3],
        a1: [Goldilocks; 3],
        b0: [Goldilocks; 3],
        b1: [Goldilocks; 3],
    ) {
        let a = pack(a0, a1);
        let b = pack(b0, b1);
        let mut got = [PackedGoldilocksNeon::ZERO; 3];
        cubic_packed_mul(&a, &b, &mut got);

        let want0 = ext_to_array(EF::from(<EF as p3_field::BasedVectorSpace<Goldilocks>>::from_basis_coefficients_slice(&a0).unwrap()) * EF::from(<EF as p3_field::BasedVectorSpace<Goldilocks>>::from_basis_coefficients_slice(&b0).unwrap()));
        let want1 = ext_to_array(EF::from(<EF as p3_field::BasedVectorSpace<Goldilocks>>::from_basis_coefficients_slice(&a1).unwrap()) * EF::from(<EF as p3_field::BasedVectorSpace<Goldilocks>>::from_basis_coefficients_slice(&b1).unwrap()));
        let (g0, g1) = unpack(&got);
        assert_eq!(g0, want0, "cubic_packed_mul lane 0 mismatch");
        assert_eq!(g1, want1, "cubic_packed_mul lane 1 mismatch");
    }

    #[test]
    fn cubic_packed_mul_matches_scalar() {
        let max = Goldilocks::new(P - 1);
        let p_redundant = Goldilocks::new(P);
        let cases: &[(
            [Goldilocks; 3],
            [Goldilocks; 3],
            [Goldilocks; 3],
            [Goldilocks; 3],
        )] = &[
            (
                [Goldilocks::new(3), Goldilocks::new(5), Goldilocks::new(7)],
                [
                    Goldilocks::new(11),
                    Goldilocks::new(13),
                    Goldilocks::new(17),
                ],
                [
                    Goldilocks::new(19),
                    Goldilocks::new(23),
                    Goldilocks::new(29),
                ],
                [
                    Goldilocks::new(31),
                    Goldilocks::new(37),
                    Goldilocks::new(41),
                ],
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

        let mut rng = SmallRng::seed_from_u64(0xC0FFEE);
        for _ in 0..1024 {
            let a0 = core::array::from_fn(|_| rng.random());
            let a1 = core::array::from_fn(|_| rng.random());
            let b0 = core::array::from_fn(|_| rng.random());
            let b1 = core::array::from_fn(|_| rng.random());
            check_mul(a0, a1, b0, b1);
        }
    }
}
