use alloc::vec;
use alloc::vec::Vec;

use p3_air::{Air, AirBuilder};
use p3_field::Field;
use p3_util::log2_ceil_usize;
use tracing::instrument;

use crate::Entry;
use crate::symbolic_expression::SymbolicExpression;
use crate::symbolic_variable::SymbolicVariable;

#[instrument(name = "infer log of constraint degree", skip_all)]
pub fn get_log_quotient_degree<F, A>(air: &A, is_zk: usize) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    assert!(is_zk <= 1, "is_zk must be either 0 or 1");
    // We pad to at least degree 2, since a quotient argument doesn't make sense with smaller degrees.
    let constraint_degree = (get_max_constraint_degree(air) + is_zk).max(2);

    // The quotient's actual degree is approximately (max_constraint_degree - 1) n,
    // where subtracting 1 comes from division by the vanishing polynomial.
    // But we pad it to a power of two so that we can efficiently decompose the quotient.
    log2_ceil_usize(constraint_degree - 1)
}

#[instrument(name = "infer constraint degree", skip_all, level = "debug")]
pub fn get_max_constraint_degree<F, A>(air: &A) -> usize
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    get_symbolic_constraints(air)
        .iter()
        .map(|c| c.degree_multiple())
        .max()
        .unwrap_or(0)
}

#[instrument(name = "evaluate constraints symbolically", skip_all, level = "debug")]
pub fn get_symbolic_constraints<F, A>(air: &A) -> Vec<SymbolicExpression<F>>
where
    F: Field,
    A: Air<SymbolicAirBuilder<F>>,
{
    let mut builder = SymbolicAirBuilder::new(air.width(), air.columns_with_shift().len());
    air.eval(&mut builder);
    builder.constraints()
}

/// An `AirBuilder` for evaluating constraints symbolically, and recording them for later use.
#[derive(Debug)]
pub struct SymbolicAirBuilder<F: Field> {
    main: Vec<SymbolicVariable<F>>,
    constraints: Vec<SymbolicExpression<F>>,
}

impl<F: Field> SymbolicAirBuilder<F> {
    pub(crate) fn new(n_columns: usize, n_columns_with_shift: usize) -> Self {
        let main_values = [0, 1]
            .into_iter()
            .flat_map(|offset| {
                let width = if offset == 0 {
                    n_columns
                } else {
                    n_columns_with_shift
                };
                (0..width).map(move |index| SymbolicVariable::new(Entry::Main { offset }, index))
            })
            .collect();
        Self {
            main: main_values,
            constraints: vec![],
        }
    }

    pub(crate) fn constraints(self) -> Vec<SymbolicExpression<F>> {
        self.constraints
    }
}

impl<F: Field> AirBuilder for SymbolicAirBuilder<F> {
    type F = F;
    type Expr = SymbolicExpression<F>;
    type Var = SymbolicVariable<F>;
    type FinalOutput = ();

    fn main(&self) -> &[Self::Var] {
        &self.main
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        self.constraints.push(x.into());
    }

    fn add_custom(&mut self, value: Self::FinalOutput) {
        let _ = value;
    }
}
