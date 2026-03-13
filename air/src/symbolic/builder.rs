use alloc::vec;
use alloc::vec::Vec;
use core::borrow::Borrow;

use p3_field::{Algebra, ExtensionField, Field};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::symbolic::expression::SymbolicExpression;
use crate::symbolic::expression_ext::SymbolicExpressionExt;
use crate::symbolic::variable::{BaseEntry, ExtEntry, SymbolicVariableExt};
use crate::symbolic::clear_symbolic_arena;
use crate::{
    Air, AirBuilder, ExtensionBuilder, PeriodicAirBuilder, PermutationAirBuilder,
    SymbolicVariable, WindowAccess,
};

/// Describes the shape of an AIR for symbolic constraint evaluation.
#[derive(Debug, Clone, Copy, Default)]
pub struct AirLayout {
    pub preprocessed_width: usize,
    pub main_width: usize,
    pub num_public_values: usize,
    pub permutation_width: usize,
    pub num_permutation_challenges: usize,
    pub num_permutation_values: usize,
    pub num_periodic_columns: usize,
}

#[instrument(skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A, layout: AirLayout) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_max_constraint_degree_extension(air, layout)
}

#[instrument(
    name = "infer base and extension constraint degree",
    skip_all,
    level = "debug"
)]
pub fn get_max_constraint_degree_extension<F, EF, A>(air: &A, layout: AirLayout) -> usize
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    let (base_constraints, extension_constraints) = get_all_symbolic_constraints(air, layout);

    let base_degree = base_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);

    let extension_degree = extension_constraints
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0);
    base_degree.max(extension_degree)
}

#[instrument(
    name = "evaluate base constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints<F, A>(air: &A, layout: AirLayout) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    clear_symbolic_arena();
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.base_constraints()
}

#[instrument(
    name = "evaluate extension constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_symbolic_constraints_extension<F, EF, A>(
    air: &A,
    layout: AirLayout,
) -> Vec<SymbolicExpressionExt<F, EF>>
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    clear_symbolic_arena();
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.extension_constraints()
}

#[instrument(
    name = "evaluate all constraints symbolically",
    skip_all,
    level = "debug"
)]
pub fn get_all_symbolic_constraints<F, EF, A>(
    air: &A,
    layout: AirLayout,
) -> (
    Vec<SymbolicExpression<F>>,
    Vec<SymbolicExpressionExt<F, EF>>,
)
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
{
    clear_symbolic_arena();
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    (builder.base_constraints(), builder.extension_constraints())
}

/// An [`AirBuilder`] for evaluating constraints symbolically.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field, EF: ExtensionField<F> = F> {
    preprocessed: RowMajorMatrix<SymbolicVariable<F>>,
    main: RowMajorMatrix<SymbolicVariable<F>>,
    public_values: Vec<SymbolicVariable<F>>,
    periodic: Vec<SymbolicVariable<F>>,
    base_constraints: Vec<SymbolicExpression<F>>,
    permutation: RowMajorMatrix<SymbolicVariableExt<F, EF>>,
    permutation_challenges: Vec<SymbolicVariableExt<F, EF>>,
    permutation_values: Vec<SymbolicVariableExt<F, EF>>,
    extension_constraints: Vec<SymbolicExpressionExt<F, EF>>,
    constraint_types: Vec<ConstraintType>,
}

impl<F: Field, EF: ExtensionField<F>> SymbolicAirBuilder<F, EF> {
    pub fn new(layout: AirLayout) -> Self {
        let AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width,
            num_permutation_challenges,
            num_permutation_values,
            num_periodic_columns,
        } = layout;
        let prep_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..preprocessed_width).map(move |index| {
                    SymbolicVariable::new(BaseEntry::Preprocessed { offset }, index)
                })
            })
            .collect();
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..main_width)
                    .map(move |index| SymbolicVariable::new(BaseEntry::Main { offset }, index))
            })
            .collect();
        let public_values = (0..num_public_values)
            .map(move |index| SymbolicVariable::new(BaseEntry::Public, index))
            .collect();
        let periodic = (0..num_periodic_columns)
            .map(|index| SymbolicVariable::new(BaseEntry::Periodic, index))
            .collect();
        let perm_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                (0..permutation_width).map(move |index| {
                    SymbolicVariableExt::new(ExtEntry::Permutation { offset }, index)
                })
            })
            .collect();
        let permutation = RowMajorMatrix::new(perm_values, permutation_width);
        let permutation_challenges = (0..num_permutation_challenges)
            .map(|index| SymbolicVariableExt::new(ExtEntry::Challenge, index))
            .collect();
        let permutation_values = (0..num_permutation_values)
            .map(|index| SymbolicVariableExt::new(ExtEntry::PermutationValue, index))
            .collect();
        Self {
            preprocessed: RowMajorMatrix::new(prep_values, preprocessed_width),
            main: RowMajorMatrix::new(main_values, main_width),
            public_values,
            periodic,
            base_constraints: vec![],
            permutation,
            permutation_challenges,
            permutation_values,
            extension_constraints: vec![],
            constraint_types: vec![],
        }
    }

    pub fn constraint_layout(&self) -> ConstraintLayout {
        let mut base_indices = Vec::new();
        let mut ext_indices = Vec::new();
        for (idx, kind) in self.constraint_types.iter().enumerate() {
            match kind {
                ConstraintType::Base => base_indices.push(idx),
                ConstraintType::Ext => ext_indices.push(idx),
            }
        }
        ConstraintLayout {
            base_indices,
            ext_indices,
        }
    }

    pub fn extension_constraints(&self) -> Vec<SymbolicExpressionExt<F, EF>> {
        self.extension_constraints.clone()
    }

    pub fn base_constraints(&self) -> Vec<SymbolicExpression<F>> {
        self.base_constraints.clone()
    }
}

/// Implement `WindowAccess` for `RowMajorMatrix` treating it as a two-row window.
impl<T: Clone + Send + Sync> WindowAccess<T> for RowMajorMatrix<T> {
    fn current_slice(&self) -> &[T] {
        assert_eq!(
            self.height(),
            2,
            "WindowAccess for RowMajorMatrix requires exactly 2 rows, got {}",
            self.height()
        );
        let values: &[T] = self.values.borrow();
        &values[..self.width]
    }

    fn next_slice(&self) -> &[T] {
        assert_eq!(
            self.height(),
            2,
            "WindowAccess for RowMajorMatrix requires exactly 2 rows, got {}",
            self.height()
        );
        let values: &[T] = self.values.borrow();
        &values[self.width..]
    }
}

impl<F: Field, EF: ExtensionField<F>> AirBuilder for SymbolicAirBuilder<F, EF> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type PreprocessedWindow = RowMajorMatrix<Self::Var>;
    type MainWindow = RowMajorMatrix<Self::Var>;
    type PublicVar = SymbolicVariable<F>;

    fn main(&self) -> Self::MainWindow {
        self.main.clone()
    }

    fn preprocessed(&self) -> &Self::PreprocessedWindow {
        &self.preprocessed
    }

    fn is_first_row(&self) -> Self::Expr {
        SymbolicExpression::IsFirstRow
    }

    fn is_last_row(&self) -> Self::Expr {
        SymbolicExpression::IsLastRow
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            SymbolicExpression::IsTransition
        } else {
            panic!("uni-stark only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.base_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Base);
    }

    fn public_values(&self) -> &[Self::PublicVar] {
        &self.public_values
    }
}

impl<F: Field, EF: ExtensionField<F>> ExtensionBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type EF = EF;
    type ExprEF = SymbolicExpressionExt<F, EF>;
    type VarEF = SymbolicVariableExt<F, EF>;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        self.extension_constraints.push(x.into());
        self.constraint_types.push(ConstraintType::Ext);
    }
}

impl<F: Field, EF: ExtensionField<F>> PermutationAirBuilder for SymbolicAirBuilder<F, EF>
where
    SymbolicExpressionExt<F, EF>: Algebra<EF>,
{
    type MP = RowMajorMatrix<Self::VarEF>;

    type RandomVar = SymbolicVariableExt<F, EF>;

    type PermutationVar = SymbolicVariableExt<F, EF>;

    fn permutation(&self) -> Self::MP {
        self.permutation.clone()
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        &self.permutation_challenges
    }

    fn permutation_values(&self) -> &[Self::PermutationVar] {
        &self.permutation_values
    }
}

impl<F: Field, EF: ExtensionField<F>> PeriodicAirBuilder for SymbolicAirBuilder<F, EF> {
    type PeriodicVar = SymbolicVariable<F>;

    fn periodic_values(&self) -> &[Self::PeriodicVar] {
        &self.periodic
    }
}

/// Tracks whether a constraint was emitted via `assert_zero` (base) or `assert_zero_ext` (ext).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConstraintType {
    Base,
    Ext,
}

/// Maps between global constraint indices and the separated base/ext streams.
#[derive(Debug, Default)]
pub struct ConstraintLayout {
    pub base_indices: Vec<usize>,
    pub ext_indices: Vec<usize>,
}

impl ConstraintLayout {
    pub const fn total_constraints(&self) -> usize {
        self.base_indices.len() + self.ext_indices.len()
    }

    pub fn decompose_alpha<F: Field, EF: ExtensionField<F>>(
        &self,
        alpha: EF,
    ) -> (Vec<Vec<F>>, Vec<EF>) {
        let total = self.total_constraints();

        let mut alpha_powers = alpha.powers().collect_n(total);
        alpha_powers.reverse();

        let base_alpha_powers = (0..EF::DIMENSION)
            .map(|d| {
                self.base_indices
                    .iter()
                    .map(|&idx| alpha_powers[idx].as_basis_coefficients_slice()[d])
                    .collect()
            })
            .collect();

        let ext_alpha_powers = self
            .ext_indices
            .iter()
            .map(|&idx| alpha_powers[idx])
            .collect();

        (base_alpha_powers, ext_alpha_powers)
    }
}

/// Evaluate the AIR symbolically and return the constraint layout.
#[instrument(name = "compute constraint layout", skip_all, level = "debug")]
pub fn get_constraint_layout<F, EF, A>(air: &A, layout: AirLayout) -> ConstraintLayout
where
    F: Field,
    EF: ExtensionField<F>,
    A: Air<SymbolicAirBuilder<F, EF>>,
    SymbolicExpression<EF>: Algebra<SymbolicExpression<F>>,
{
    clear_symbolic_arena();
    let mut builder = SymbolicAirBuilder::new(layout);
    air.eval(&mut builder);
    builder.constraint_layout()
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;
    use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};

    use super::*;
    use crate::BaseAir;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[derive(Debug)]
    struct MockAir {
        constraints: Vec<SymbolicVariable<F>>,
        width: usize,
    }

    impl BaseAir<F> for MockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<F>> for MockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F>) {
            for constraint in &self.constraints {
                builder.assert_zero(*constraint);
            }
        }
    }

    const fn layout(
        preprocessed_width: usize,
        main_width: usize,
        num_public_values: usize,
        num_periodic_columns: usize,
    ) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width: 0,
            num_permutation_challenges: 0,
            num_permutation_values: 0,
            num_periodic_columns,
        }
    }

    const fn layout_with_perm(
        preprocessed_width: usize,
        main_width: usize,
        num_public_values: usize,
        permutation_width: usize,
        num_permutation_challenges: usize,
        num_periodic_columns: usize,
    ) -> AirLayout {
        AirLayout {
            preprocessed_width,
            main_width,
            num_public_values,
            permutation_width,
            num_permutation_challenges,
            num_permutation_values: 0,
            num_periodic_columns,
        }
    }

    #[test]
    fn test_get_max_constraint_degree_no_constraints() {
        let air = MockAir {
            constraints: vec![],
            width: 4,
        };
        let l = layout(3, air.width, air.num_public_values(), 0);
        let max_degree = get_max_constraint_degree(&air, l);
        assert_eq!(max_degree, 0);
    }

    #[test]
    fn test_get_max_constraint_degree_multiple_constraints() {
        let air = MockAir {
            constraints: vec![
                SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0),
                SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1),
                SymbolicVariable::new(BaseEntry::Main { offset: 2 }, 2),
            ],
            width: 4,
        };
        let l = layout(3, air.width, air.num_public_values(), 0);
        let max_degree = get_max_constraint_degree(&air, l);
        assert_eq!(max_degree, 1);
    }

    #[test]
    fn test_get_symbolic_constraints() {
        let c1 = SymbolicVariable::new(BaseEntry::Main { offset: 0 }, 0);
        let c2 = SymbolicVariable::new(BaseEntry::Main { offset: 1 }, 1);

        let air = MockAir {
            constraints: vec![c1, c2],
            width: 4,
        };

        let l = layout(3, air.width, air.num_public_values(), 0);
        let constraints = get_symbolic_constraints(&air, l);

        assert_eq!(constraints.len(), 2);
        assert!(constraints.iter().any(
            |x| matches!(x, SymbolicExpression::Variable(v) if v.index == c1.index && v.entry == c1.entry)
        ));
        assert!(constraints.iter().any(
            |x| matches!(x, SymbolicExpression::Variable(v) if v.index == c2.index && v.entry == c2.entry)
        ));
    }

    #[test]
    fn test_symbolic_air_builder_initialization() {
        let builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));

        let expected_main = [
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 0 }, 3),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 0),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 1),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 2),
            SymbolicVariable::<F>::new(BaseEntry::Main { offset: 1 }, 3),
        ];

        let builder_main = &builder.main.values;
        assert_eq!(builder_main.len(), expected_main.len());
        for (expected, actual) in expected_main.iter().zip(builder_main.iter()) {
            assert_eq!(expected.index, actual.index);
            assert_eq!(expected.entry, actual.entry);
        }
    }

    #[test]
    fn test_symbolic_air_builder_is_first_last_row() {
        let builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));
        assert!(matches!(builder.is_first_row(), SymbolicExpression::IsFirstRow));
        assert!(matches!(builder.is_last_row(), SymbolicExpression::IsLastRow));
    }

    #[test]
    fn test_symbolic_air_builder_assert_zero() {
        let mut builder = SymbolicAirBuilder::<F>::new(layout(2, 4, 3, 0));
        let expr = SymbolicExpression::Constant(F::new(5));
        builder.assert_zero(expr);

        let constraints = builder.base_constraints();
        assert_eq!(constraints.len(), 1);
        assert!(constraints
            .iter()
            .any(|x| matches!(x, SymbolicExpression::Constant(val) if *val == F::new(5))));
    }

    #[test]
    fn test_is_transition_window_size_2() {
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 2, 0, 0));
        let expr = builder.is_transition_window(2);
        assert!(matches!(expr, SymbolicExpression::IsTransition));
    }

    #[test]
    #[should_panic(expected = "uni-stark only supports a window size of 2")]
    fn test_is_transition_window_size_3_panics() {
        let builder = SymbolicAirBuilder::<F>::new(layout(0, 2, 0, 0));
        let _ = builder.is_transition_window(3);
    }

    #[derive(Debug)]
    struct ExtMockAir {
        width: usize,
    }

    impl BaseAir<F> for ExtMockAir {
        fn width(&self) -> usize {
            self.width
        }
    }

    impl Air<SymbolicAirBuilder<F, EF>> for ExtMockAir {
        fn eval(&self, builder: &mut SymbolicAirBuilder<F, EF>) {
            let main = builder.main();
            builder.assert_zero(main.values[0]);
            let perm = builder.permutation();
            builder.assert_zero_ext(perm.values[0]);
        }
    }

    #[test]
    fn test_get_all_symbolic_constraints() {
        let air = ExtMockAir { width: 2 };
        let l = layout_with_perm(0, air.width, air.num_public_values(), 3, 1, 0);
        let (base, ext) = get_all_symbolic_constraints::<F, EF, _>(&air, l);
        assert_eq!(base.len(), 1);
        assert_eq!(ext.len(), 1);
    }

    #[test]
    fn test_decompose_alpha_empty_layout() {
        let layout = ConstraintLayout::default();
        let alpha = EF::TWO;
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);
        assert_eq!(base.len(), <EF as BasedVectorSpace<F>>::DIMENSION);
        for col in &base {
            assert!(col.is_empty());
        }
        assert!(ext.is_empty());
    }

    #[test]
    fn test_decompose_alpha_interleaved_base_and_ext() {
        let layout = ConstraintLayout {
            base_indices: vec![0, 2],
            ext_indices: vec![1],
        };
        let alpha = EF::TWO;
        let (base, ext) = layout.decompose_alpha::<F, EF>(alpha);

        let alpha_sq = alpha * alpha;
        let power_base_0 = alpha_sq;
        let power_base_2 = EF::ONE;
        let power_ext_1 = alpha;

        let coeffs_0 = power_base_0.as_basis_coefficients_slice();
        let coeffs_2 = power_base_2.as_basis_coefficients_slice();
        for (d, col) in base.iter().enumerate() {
            assert_eq!(col.len(), 2);
            assert_eq!(col[0], coeffs_0[d]);
            assert_eq!(col[1], coeffs_2[d]);
        }
        assert_eq!(ext, vec![power_ext_1]);
    }
}
