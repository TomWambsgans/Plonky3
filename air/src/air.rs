use core::ops::{Add, Mul, Sub};

use alloc::vec::Vec;
use p3_field::{Algebra, ExtensionField, Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

/// The underlying structure of an AIR.
pub trait BaseAir<F>: Sync {
    /// The number of columns (a.k.a. registers) in this AIR.
    fn width(&self) -> usize;

    fn degree(&self) -> usize;

    /// Returns the list of columns such that the row "down" is used within the transition constraints.
    fn columns_with_shift(&self) -> Vec<usize>;
}

/// An extension of `BaseAir` that includes support for public values.
pub trait BaseAirWithPublicValues<F>: BaseAir<F> {
    /// Return the number of expected public values.
    fn num_public_values(&self) -> usize {
        0
    }
}

/// An algebraic intermediate representation (AIR) definition.
///
/// Contains an evaluation function for computing the constraints of the AIR.
/// This function can be applied to an evaluation trace in which case each
/// constraint will compute a particular value or it can be applied symbolically
/// with each constraint computing a symbolic expression.
pub trait Air<AB: AirBuilder>: BaseAir<AB::F> {
    /// Evaluate all AIR constraints using the provided builder.
    ///
    /// The builder provides both the trace on which the constraints
    /// are evaluated on as well as the method of accumulating the
    /// constraint evaluations.
    ///
    /// # Arguments
    /// - `builder`: Mutable reference to an `AirBuilder` for defining constraints.
    fn eval(&self, builder: &mut AB);
}

/// A builder which contains both a trace on which AIR constraints can be evaluated as well as a method of accumulating the AIR constraint evaluations.
///
/// Supports both symbolic cases where the constraints are treated as polynomials and collected into a vector
/// as well cases where the constraints are evaluated on an evaluation trace and combined using randomness.
pub trait AirBuilder: Sized {
    /// Underlying field type.
    ///
    /// This should usually implement `Field` but there are a few edge cases (mostly involving `PackedFields`) where
    /// it may only implement `PrimeCharacteristicRing`.
    type F: PrimeCharacteristicRing + Sync;

    /// Serves as the output type for an AIR constraint evaluation.
    type Expr: Algebra<Self::F> + Algebra<Self::Var>;

    /// The type of the variable appearing in the trace matrix.
    ///
    /// Serves as the input type for an AIR constraint evaluation.
    type Var: Into<Self::Expr>
        + Clone
        + Send
        + Sync
        + Add<Self::F, Output = Self::Expr>
        + Add<Self::Var, Output = Self::Expr>
        + Add<Self::Expr, Output = Self::Expr>
        + Sub<Self::F, Output = Self::Expr>
        + Sub<Self::Var, Output = Self::Expr>
        + Sub<Self::Expr, Output = Self::Expr>
        + Mul<Self::F, Output = Self::Expr>
        + Mul<Self::Var, Output = Self::Expr>
        + Mul<Self::Expr, Output = Self::Expr>;

    /// Return the matrix representing the main (primary) trace registers.
    fn main(&self) -> &[Self::Var];

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
}

/// Extension trait for `AirBuilder` providing access to public values.
pub trait AirBuilderWithPublicValues: AirBuilder {
    /// Type representing a public variable.
    type PublicVar: Into<Self::Expr> + Copy;

    /// Return the list of public variables.
    fn public_values(&self) -> &[Self::PublicVar];
}

/// Extension of `AirBuilder` for working over extension fields.
pub trait ExtensionBuilder: AirBuilder<F: Field> {
    /// Extension field type.
    type EF: ExtensionField<Self::F>;

    /// Expression type over extension field elements.
    type ExprEF: Algebra<Self::Expr> + Algebra<Self::EF>;

    /// Variable type over extension field elements.
    type VarEF: Into<Self::ExprEF> + Copy + Send + Sync;

    /// Assert that an extension field expression is zero.
    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>;

    /// Assert that two extension field expressions are equal.
    fn assert_eq_ext<I1, I2>(&mut self, x: I1, y: I2)
    where
        I1: Into<Self::ExprEF>,
        I2: Into<Self::ExprEF>,
    {
        self.assert_zero_ext(x.into() - y.into());
    }

    /// Assert that an extension field expression is equal to one.
    fn assert_one_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.assert_eq_ext(x, Self::ExprEF::ONE)
    }
}

/// Trait for builders supporting permutation arguments (e.g., for lookup constraints).
pub trait PermutationAirBuilder: ExtensionBuilder {
    /// Matrix type over extension field variables representing a permutation.
    type MP: Matrix<Self::VarEF>;

    /// Randomness variable type used in permutation commitments.
    type RandomVar: Into<Self::ExprEF> + Copy;

    /// Return the matrix representing permutation registers.
    fn permutation(&self) -> Self::MP;

    /// Return the list of randomness values for permutation argument.
    fn permutation_randomness(&self) -> &[Self::RandomVar];
}
