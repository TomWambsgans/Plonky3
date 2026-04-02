use p3_field::extension::CubicExtendableAlgebra;

use super::PackedGoldilocksAVX512;
use crate::Goldilocks;

impl CubicExtendableAlgebra<Goldilocks> for PackedGoldilocksAVX512 {}
