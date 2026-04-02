use p3_field::extension::CubicExtendableAlgebra;

use super::PackedGoldilocksAVX2;
use crate::Goldilocks;

impl CubicExtendableAlgebra<Goldilocks> for PackedGoldilocksAVX2 {}
