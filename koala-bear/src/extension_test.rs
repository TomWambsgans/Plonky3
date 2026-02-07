#[cfg(test)]
mod test_quartic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::KoalaBear;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 4>;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of P^4 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 7] {
        [
            (BigUint::from(2u8), 26),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(127u8), 1),
            (BigUint::from(283u16), 1),
            (BigUint::from(1254833u32), 1),
            (BigUint::from(453990990362758349u64), 1),
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

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[F::TWO, F::ONE, F::ZERO, F::TWO]).unwrap()
            ),
            "2 + X + 2 X^3"
        );
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}

#[cfg(test)]
mod test_octic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::KoalaBear;

    type F = KoalaBear;
    type EF = BinomialExtensionField<F, 8>;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of p^8 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 10] {
        [
            (BigUint::from(2u8), 27),
            (BigUint::from(3u8), 1),
            (BigUint::from(5u8), 1),
            (BigUint::from(17u8), 2),
            (BigUint::from(127u8), 1),
            (BigUint::from(137u8), 1),
            (BigUint::from(283u16), 1),
            (BigUint::from(1254833u32), 1),
            (BigUint::from(453990990362758349u64), 1),
            (BigUint::from(260283155268050089696848485460377u128), 1),
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

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[
                    F::TWO,
                    F::ONE,
                    F::ZERO,
                    F::TWO,
                    F::ZERO,
                    F::TWO,
                    F::TWO,
                    F::ZERO
                ])
                .unwrap()
            ),
            "2 + X + 2 X^3 + 2 X^5 + 2 X^6"
        );
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}

#[cfg(test)]
mod test_sextic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::{BasedVectorSpace, ExtensionField, Field, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::{KoalaBear, SexticExtensionFieldKB};

    type F = KoalaBear;
    type EF = SexticExtensionFieldKB;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Prime factorization of p^6 - 1.
    // p^6-1 = (p-1)(p+1)(p^2+p+1)(p^2-p+1)
    //       = 2^25 * 3^2 * 7 * 67 * 127 * 283 * 1254833 * 9679978477096567 * 1513303300498959019
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 9] {
        [
            (BigUint::from(2u8), 25),
            (BigUint::from(3u8), 2),
            (BigUint::from(7u8), 1),
            (BigUint::from(67u8), 1),
            (BigUint::from(127u8), 1),
            (BigUint::from(283u16), 1),
            (BigUint::from(1254833u32), 1),
            (BigUint::from(9679978477096567u64), 1),
            (BigUint::from(1513303300498959019u64), 1),
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

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[
                    F::TWO,
                    F::ONE,
                    F::ZERO,
                    F::TWO,
                    F::ZERO,
                    F::ZERO,
                ])
                .unwrap()
            ),
            "2 + X + 2 X^3"
        );
    }

    #[test]
    fn test_defining_relation() {
        // Verify w^6 = 2w^3 + 2
        let w = EF::new([F::ZERO, F::ONE, F::ZERO, F::ZERO, F::ZERO, F::ZERO]);
        let w3 = w * w * w;
        let w6 = w3 * w3;
        let expected = EF::new([F::TWO, F::ZERO, F::ZERO, F::TWO, F::ZERO, F::ZERO]);
        assert_eq!(w6, expected, "w^6 should equal 2w^3 + 2");
    }

    #[test]
    fn test_inverse_consistency() {
        // Test that x * x^{-1} = 1 for several elements
        let test_cases: &[[u32; 6]] = &[
            [1, 2, 3, 4, 5, 6],
            [7, 11, 13, 17, 19, 23],
            [100, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ];

        for coeffs in test_cases {
            let x = EF::new(F::new_array(*coeffs));
            let x_inv = x.inverse();
            assert_eq!(x * x_inv, EF::ONE, "x * x^-1 should be 1 for {:?}", coeffs);
        }
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}

#[cfg(test)]
mod test_quintic_extension {
    use alloc::format;

    use num_bigint::BigUint;
    use p3_field::{BasedVectorSpace, ExtensionField, PrimeCharacteristicRing};
    use p3_field_testing::{
        test_extension_field, test_field, test_packed_extension_field,
        test_two_adic_extension_field,
    };

    use crate::{KoalaBear, QuinticExtensionFieldKB};

    type F = KoalaBear;
    type EF = QuinticExtensionFieldKB;

    // MontyField31's have no redundant representations.
    const ZEROS: [EF; 1] = [EF::ZERO];
    const ONES: [EF; 1] = [EF::ONE];

    // Get the prime factorization of the order of the multiplicative group.
    // i.e. the prime factorization of p^5 - 1.
    fn multiplicative_group_prime_factorization() -> [(BigUint, u32); 7] {
        [
            (BigUint::from(2u8), 24),
            (BigUint::from(11u8), 2),
            (BigUint::from(71u8), 1),
            (BigUint::from(127u8), 1),
            (BigUint::from(181u8), 1),
            (BigUint::from(344859791u32), 1),
            (BigUint::from(38435241482589294665521u128), 1),
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

    #[test]
    fn display() {
        assert_eq!(format!("{}", EF::ZERO), "0");
        assert_eq!(format!("{}", EF::ONE), "1");
        assert_eq!(format!("{}", EF::TWO), "2");

        assert_eq!(
            format!(
                "{}",
                EF::from_basis_coefficients_slice(&[F::TWO, F::ONE, F::ZERO, F::TWO, F::ZERO,])
                    .unwrap()
            ),
            "2 + X + 2 X^3"
        );
    }

    type Pef = <EF as ExtensionField<F>>::ExtensionPacking;
    const PACKED_ZEROS: [Pef; 1] = [Pef::ZERO];
    const PACKED_ONES: [Pef; 1] = [Pef::ONE];
    test_packed_extension_field!(super::Pef, &super::PACKED_ZEROS, &super::PACKED_ONES);
}
