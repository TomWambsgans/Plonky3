use alloc::vec::Vec;
use p3_field::{Algebra, PrimeCharacteristicRing};

pub trait Air: Send + Sync + 'static {
    fn width(&self) -> usize;

    fn degree(&self) -> usize;

    fn columns_with_shift(&self) -> Vec<usize>;

    fn eval<AB: AirBuilder>(&self, builder: &mut AB);

    fn eval_custom<AB: AirBuilder>(&self, inputs: &[AB::Expr]) -> AB::FinalOutput;
}


pub trait AirBuilder: Sized {
    type F: PrimeCharacteristicRing + Sync;

    type Expr: Algebra<Self::F> + 'static;

    // the final type, after batching with the random challenges (extension field, in practice)
    type FinalOutput: 'static;

    fn main(&self) -> &[Self::Expr];

    /// Assert that the given element is zero.
    ///
    /// Where possible, batching multiple assert_zero calls
    /// into a single assert_zeros call will improve performance.
    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I);

    /// Assert that every element of a given array is 0.
    ///
    /// This should be preferred over calling `assert_zero` multiple times.
    fn assert_zeros<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        for elem in array {
            self.assert_zero(elem);
        }
    }

    /// Assert that a given array consists of only boolean values.
    fn assert_bools<const N: usize, I: Into<Self::Expr>>(&mut self, array: [I; N]) {
        let zero_array = array.map(|x| x.into().bool_check());
        self.assert_zeros(zero_array);
    }

    /// Assert that `x` element is equal to `1`.
    fn assert_one<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into() - Self::Expr::ONE);
    }

    /// Assert that the given elements are equal.
    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        self.assert_zero(x.into() - y.into());
    }

    /// Assert that `x` is a boolean, i.e. either `0` or `1`.
    ///
    /// Where possible, batching multiple assert_bool calls
    /// into a single assert_bools call will improve performance.
    fn assert_bool<I: Into<Self::Expr>>(&mut self, x: I) {
        self.assert_zero(x.into().bool_check());
    }

    fn add_custom(&mut self, value: Self::FinalOutput);
}
