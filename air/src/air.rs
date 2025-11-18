use core::ops::{Add, Mul, Sub};

use alloc::vec::Vec;
use p3_field::PrimeCharacteristicRing;

pub trait Air: Send + Sync + 'static {
    fn degree() -> usize;

    fn n_columns_f() -> usize;
    fn n_columns_ef() -> usize;

    fn n_columns() -> usize {
        Self::n_columns_f() + Self::n_columns_ef()
    }

    fn n_constraints() -> usize;

    fn down_column_indexes() -> Vec<usize>;

    fn eval<AB: AirBuilder>(&self, builder: &mut AB);
}

pub trait AirBuilder: Sized {
    type F: PrimeCharacteristicRing + 'static;
    type EF: PrimeCharacteristicRing
        + 'static
        + Add<Self::F, Output = Self::EF>
        + Mul<Self::F, Output = Self::EF>
        + Sub<Self::F, Output = Self::EF>
        + From<Self::F>;

    fn up_f(&self) -> &[Self::F];
    fn down_f(&self) -> &[Self::F];
    fn up_ef(&self) -> &[Self::EF];
    fn down_ef(&self) -> &[Self::EF];

    fn assert_zero(&mut self, x: Self::F);
    fn assert_zero_ef(&mut self, x: Self::EF);

    fn eval_custom(&mut self, x: Self::EF);

    fn assert_eq(&mut self, x: Self::F, y: Self::F) {
        self.assert_zero(x - y);
    }

    fn assert_bool(&mut self, x: Self::F) {
        self.assert_zero(x.bool_check());
    }

    fn assert_eq_ef(&mut self, x: Self::EF, y: Self::EF) {
        self.assert_zero_ef(x - y);
    }

    fn assert_bool_ef(&mut self, x: Self::EF) {
        self.assert_zero_ef(x.bool_check());
    }
}
