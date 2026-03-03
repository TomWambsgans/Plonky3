extern crate std;

use core::hint::black_box;
use std::println;
use std::time::Instant;

use p3_field::{Field, PackedValue, PrimeCharacteristicRing};
use p3_mds::coset_mds::CosetMds;
use p3_poseidon::{
    Poseidon, PoseidonExternalLayerGeneric, PoseidonInternalLayerGeneric,
};
use p3_symmetric::Permutation;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use crate::{KoalaBear, default_koalabear_poseidon2_16};

type F = KoalaBear;
type FPacking = <KoalaBear as Field>::Packing;
const PACKING_WIDTH: usize = <FPacking as PackedValue>::WIDTH;

type Mds16 = CosetMds<F, 16>;
type Poseidon1KoalaBear16 = Poseidon<
    F,
    PoseidonExternalLayerGeneric<F, Mds16, 16>,
    PoseidonInternalLayerGeneric<F, 16>,
    16,
    3,
>;

fn make_poseidon1() -> Poseidon1KoalaBear16 {
    let mut rng = SmallRng::seed_from_u64(1);
    let half_num_full_rounds = 4;
    let num_partial_rounds = 20;
    Poseidon1KoalaBear16::new_from_rng(
        half_num_full_rounds,
        num_partial_rounds,
        &Mds16::default(),
        &mut rng,
    )
}

/*
RUSTFLAGS='-C target-cpu=native' cargo test --release --package p3-koala-bear --lib -- poseidon1_vs_poseidon2_test::bench_koalabear_poseidon1_vs_poseidon2 --exact --nocapture
*/
#[test]
fn bench_koalabear_poseidon1_vs_poseidon2() {
    let n = 1 << 22;
    let poseidon1 = make_poseidon1();
    let poseidon2 = default_koalabear_poseidon2_16();

    // // --- Single-threaded, no SIMD ---

    // let time = Instant::now();
    // let mut state = [F::ZERO; 16];
    // for _ in 0..n {
    //     poseidon1.permute_mut(&mut state);
    // }
    // let _ = black_box(state);
    // let time_p1 = time.elapsed();
    // println!(
    //     "Poseidon1, single-threaded, no SIMD: {:.2}M hashes/s",
    //     n as f64 / time_p1.as_secs_f64() / 1_000_000.0
    // );

    // let time = Instant::now();
    // let mut state = [F::ZERO; 16];
    // for _ in 0..n {
    //     poseidon2.permute_mut(&mut state);
    // }
    // let _ = black_box(state);
    // let time_p2 = time.elapsed();
    // println!(
    //     "Poseidon2, single-threaded, no SIMD: {:.2}M hashes/s ({:.1}x faster than Poseidon1)",
    //     n as f64 / time_p2.as_secs_f64() / 1_000_000.0,
    //     time_p1.as_secs_f64() / time_p2.as_secs_f64()
    // );

    // --- Single-threaded, SIMD ---

    let time = Instant::now();
    let mut state = [FPacking::ZERO; 16];
    for _ in 0..n / PACKING_WIDTH {
        poseidon1.permute_mut(&mut state);
    }
    let _ = black_box(state);
    let time_p1_simd = time.elapsed();
    println!(
        "Poseidon1, single-threaded, SIMD: {:.2}M hashes/s",
        n as f64 / time_p1_simd.as_secs_f64() / 1_000_000.0
    );

    let time = Instant::now();
    let mut state = [FPacking::ZERO; 16];
    for _ in 0..n / PACKING_WIDTH {
        poseidon2.permute_mut(&mut state);
    }
    let _ = black_box(state);
    let time_p2_simd = time.elapsed();
    println!(
        "Poseidon2, single-threaded, SIMD: {:.2}M hashes/s ({:.1}x faster than Poseidon1)",
        n as f64 / time_p2_simd.as_secs_f64() / 1_000_000.0,
        time_p1_simd.as_secs_f64() / time_p2_simd.as_secs_f64()
    );
}
