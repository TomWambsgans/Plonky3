//! A minimal univariate STARK framework.

#![no_std]

extern crate alloc;

mod config;
mod proof;
mod symbolic_builder;
mod symbolic_expression;
mod symbolic_variable;

pub use config::*;
pub use proof::*;
pub use symbolic_builder::*;
pub use symbolic_expression::*;
pub use symbolic_variable::*;
