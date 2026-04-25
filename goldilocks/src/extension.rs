use p3_field::extension::{
    BinomiallyExtendable, BinomiallyExtendableAlgebra, CubicExtendableAlgebra,
    CubicTrinomialExtendable, HasTwoAdicBinomialExtension, HasTwoAdicCubicExtension,
};
use p3_field::{PrimeCharacteristicRing, TwoAdicField, field_to_array};

use crate::Goldilocks;

impl BinomiallyExtendableAlgebra<Self, 2> for Goldilocks {}

impl BinomiallyExtendable<2> for Goldilocks {
    // Verifiable in Sage with
    // `R.<x> = GF(p)[]; assert (x^2 - 7).is_irreducible()`.
    const W: Self = Self::new(7);

    // DTH_ROOT = W^((p - 1)/2).
    const DTH_ROOT: Self = Self::new(18446744069414584320);

    const EXT_GENERATOR: [Self; 2] = [
        Self::new(18081566051660590251),
        Self::new(16121475356294670766),
    ];
}

impl HasTwoAdicBinomialExtension<2> for Goldilocks {
    const EXT_TWO_ADICITY: usize = 33;

    fn ext_two_adic_generator(bits: usize) -> [Self; 2] {
        assert!(bits <= 33);

        if bits == 33 {
            [Self::ZERO, Self::new(15659105665374529263)]
        } else {
            [Self::two_adic_generator(bits), Self::ZERO]
        }
    }
}

impl BinomiallyExtendableAlgebra<Self, 5> for Goldilocks {}

impl BinomiallyExtendable<5> for Goldilocks {
    // Verifiable via:
    //  ```sage
    //  # Define Fp
    //  p = 2**64 - 2**32 + 1
    //  F = GF(p)

    //  # Define Fp[z]
    //  R.<z> = PolynomialRing(F)

    //  # The polynomial x^5-3 is irreducible
    //  assert(R(z^5-3).is_irreducible())
    //  ```
    const W: Self = Self::new(3);

    // 5-th root = w^((p - 1)/5)
    const DTH_ROOT: Self = Self::new(1041288259238279555);

    // Generator of the extension field
    // Obtained by finding the smallest Hamming weight vector
    // with appropriate order, starting at [0,1,0,0,0]
    const EXT_GENERATOR: [Self; 5] = [Self::TWO, Self::ONE, Self::ZERO, Self::ZERO, Self::ZERO];
}

impl HasTwoAdicBinomialExtension<5> for Goldilocks {
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 5] {
        assert!(bits <= 32);

        field_to_array(Self::two_adic_generator(bits))
    }
}

impl CubicExtendableAlgebra<Self> for Goldilocks {}

impl CubicTrinomialExtendable for Goldilocks {
    // Verifiable via:
    //  ```sage
    //  p = 2**64 - 2**32 + 1
    //  F = GF(p)
    //  R.<x> = PolynomialRing(F)
    //  assert R(x^3 - x - 1).is_irreducible()
    //
    //  K.<a> = F.extension(x^3 - x - 1)
    //  # Frobenius coefficients: x^p and x^{2p} mod (x^3-x-1)
    //  xp = R(a^p)
    //  x2p = R(a^(2*p))
    //  print([int(c) for c in xp.list()])
    //  print([int(c) for c in x2p.list()])
    //  ```
    //
    // Computed by the `compute_cubic_frobenius_coeffs` test; verifiable via Sage.
    const FROBENIUS_COEFFS: [[Self; 3]; 2] = [
        // x^p mod (x^3 - x - 1)
        [
            Self::new(10615703402128488253),
            Self::new(10050274602728160328),
            Self::new(11746561000929144102),
        ],
        // x^{2p} mod (x^3 - x - 1)
        [
            Self::new(6700183068485440220),
            Self::new(14531223735771536287),
            Self::new(8396469466686423992),
        ],
    ];

    // Placeholder generator -- will be verified/replaced by test
    const EXT_GENERATOR: [Self; 3] = [Self::TWO, Self::ONE, Self::ZERO];
}

impl HasTwoAdicCubicExtension for Goldilocks {
    // v_2(p^3 - 1) = v_2((p-1)(p^2+p+1)) = v_2(p-1) + v_2(p^2+p+1)
    // p-1 = 2^32 * (2^32 - 1), so v_2(p-1) = 32
    // p^2+p+1 is odd (since p is odd, p^2+p+1 = p(p+1)+1, p+1 is even, p(p+1) is even, +1 is odd)
    // So v_2(p^3-1) = 32
    const EXT_TWO_ADICITY: usize = 32;

    fn ext_two_adic_generator(bits: usize) -> [Self; 3] {
        assert!(bits <= 32);
        field_to_array(Self::two_adic_generator(bits))
    }
}

#[cfg(test)]
mod test_quadratic_extension {

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 2>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^2 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(2u8), 33),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(7u8), 1),
            (BigUint::from(17u8), 1),
            (BigUint::from(179u8), 1),
            (BigUint::from(257u16), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(7361031152998637u64), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 2);
}

#[cfg(test)]
mod test_quintic_extension {

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = BinomialExtensionField<F, 5>;

    // There is a redundant representation of zero but we already tested it
    // when testing the base field.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^5 - 1.
    fn multiplicative_group_prime_factorization() -> [(num_bigint::BigUint, u32); 10] {
        [
            (BigUint::from(2u8), 32),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 2),
            (BigUint::from(17u8), 1),
            (BigUint::from(257u16), 1),
            (BigUint::from(45971u16), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(255006435240067831u64), 1),
            (BigUint::from(280083648770327405561u128), 1),
            (BigUint::from(7053197395277272939628824863222181u128), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
    p3_field_testing::test_packed_binomial_extension_division!(F, 5);
}

#[cfg(test)]
mod test_cubic_extension {
    use num_bigint::BigUint;
    use p3_field::extension::CubicTrinomialExtensionField;
    use p3_field::{ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_frobenius, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = CubicTrinomialExtensionField<F>;

    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Prime factorization of p^3 - 1
    // p^3 - 1 = (p-1)(p^2+p+1)
    //   p-1 = 2^32 * 3 * 5 * 17 * 257 * 65537
    //   p^2+p+1 = 3 * 937 * 724723 * 167034643597991036904547663171
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(2u8), 32),
            (BigUint::from(3u8), 2),
            (BigUint::from(5u8), 1),
            (BigUint::from(17u8), 1),
            (BigUint::from(257u16), 1),
            (BigUint::from(937u16), 1),
            (BigUint::from(65537u32), 1),
            (BigUint::from(724723u32), 1),
            (BigUint::from(167034643597991036904547663171u128), 1),
        ]
    }

    test_field!(
        super::EF,
        &super::ZEROS,
        &super::ONES,
        &super::multiplicative_group_prime_factorization()
    );

    test_extension_field!(super::F, super::EF);
    test_two_adic_extension_field!(super::F, super::EF);
    test_frobenius!(super::F, super::EF);

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(
        super::F,
        super::EF,
        super::Pef,
        &super::PACKED_ZEROS,
        &super::PACKED_ONES
    );
}

#[cfg(test)]
mod test_cubic_extension_arithmetic {
    use p3_field::extension::{CubicTrinomialExtendable, CubicTrinomialExtensionField, HasFrobenius};
    use p3_field::{Field, PrimeCharacteristicRing};

    use crate::Goldilocks;

    type F = Goldilocks;
    type EF = CubicTrinomialExtensionField<F>;

    const P: u64 = 0xFFFF_FFFF_0000_0001;

    #[test]
    fn verify_cubic_frobenius_coeffs() {
        // Verify Frobenius coefficients match x^p and x^{2p} computed via exponentiation.
        let x = EF::new([F::ZERO, F::ONE, F::ZERO]);
        let x_p = x.exp_u64(P);
        let x_2p = x_p.square();

        assert_eq!(x_p, EF::new(F::FROBENIUS_COEFFS[0]));
        assert_eq!(x_2p, EF::new(F::FROBENIUS_COEFFS[1]));

        // Frobenius is a ring homomorphism
        let a = EF::new([F::new(3), F::new(5), F::new(7)]);
        let b = EF::new([F::new(11), F::new(13), F::new(17)]);
        assert_eq!((a * b).frobenius(), a.frobenius() * b.frobenius());
        assert_eq!((a + b).frobenius(), a.frobenius() + b.frobenius());

        // Frobenius fixes base field
        assert_eq!(EF::from(F::new(42)).frobenius(), EF::from(F::new(42)));
    }

    #[test]
    fn verify_cubic_inversion() {
        let a = EF::new([F::new(3), F::new(5), F::new(7)]);
        assert_eq!(a * a.inverse(), EF::ONE);

        // x^-1 = -1 + x^2 (since x(-1+x^2) = -x + x^3 = -x + x+1 = 1)
        let x = EF::new([F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(x.inverse(), EF::new([F::NEG_ONE, F::ZERO, F::ONE]));
    }

    #[test]
    fn verify_reduction_rules() {
        let x = EF::new([F::ZERO, F::ONE, F::ZERO]);
        assert_eq!(x * x, EF::new([F::ZERO, F::ZERO, F::ONE]));         // x^2
        assert_eq!(x * x * x, EF::new([F::ONE, F::ONE, F::ZERO]));      // x^3 = x + 1
        assert_eq!(x * x * x * x, EF::new([F::ZERO, F::ONE, F::ONE]));  // x^4 = x^2 + x
    }

    #[test]
    fn verify_cubic_square() {
        let a = EF::new([F::new(3), F::new(5), F::new(7)]);
        assert_eq!(a.square(), a * a);
    }
}
