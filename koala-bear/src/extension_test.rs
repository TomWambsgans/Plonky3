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
