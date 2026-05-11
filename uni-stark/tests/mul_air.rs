use core::fmt::Debug;
use core::marker::PhantomData;
use std::hint::black_box;

use itertools::Itertools;
use p3_air::{Air, AirBuilder, BaseAir, WindowAccess};
use p3_baby_bear::{BabyBear, Poseidon2BabyBear};
use p3_challenger::{DuplexChallenger, HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_commit::testing::TrivialPcs;
use p3_dft::Radix2DitParallel;
use p3_field::extension::{
    BinomialExtensionField, CubicTrinomialExtensionField, QuinticTrinomialExtensionField,
};
use p3_field::{ExtensionField, PackedValue};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_fri::{FriParameters, HidingFriPcs, TwoAdicFriPcs};
use p3_goldilocks::Goldilocks;
use p3_keccak::Keccak256Hash;
use p3_koala_bear::KoalaBear;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::{MerkleTreeHidingMmcs, MerkleTreeMmcs};
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{
    CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher, TruncatedPermutation,
};
use p3_uni_stark::{StarkConfig, StarkGenericConfig, Val, prove, verify};
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

/// How many `a * b = c` operations to do per row in the AIR.
const REPETITIONS: usize = 20; // This should be < 255 so it can fit into a u8.
const TRACE_WIDTH: usize = REPETITIONS * 3;

/*
In its basic form, asserts a^(self.degree-1) * b = c
(so that the total constraint degree is self.degree)


If `uses_transition_constraints`, checks that on transition rows, the first a = row number
*/
pub struct MulAir {
    degree: u64,
    uses_boundary_constraints: bool,
    uses_transition_constraints: bool,
}

impl Default for MulAir {
    fn default() -> Self {
        Self {
            degree: 3,
            uses_boundary_constraints: true,
            uses_transition_constraints: true,
        }
    }
}

impl MulAir {
    pub fn random_valid_trace<F: Field>(&self, rows: usize, valid: bool) -> RowMajorMatrix<F>
    where
        StandardUniform: Distribution<F>,
    {
        let mut rng = SmallRng::seed_from_u64(1);
        let mut trace_values = F::zero_vec(rows * TRACE_WIDTH);
        for (i, (a, b, c)) in trace_values.iter_mut().tuples().enumerate() {
            let row = i / REPETITIONS;
            *a = if self.uses_transition_constraints {
                F::from_usize(i)
            } else {
                rng.random()
            };
            *b = if self.uses_boundary_constraints && row == 0 {
                a.square() + F::ONE
            } else {
                rng.random()
            };
            *c = a.exp_u64(self.degree - 1) * *b;

            if !valid {
                // make it invalid
                *c *= F::TWO;
            }
        }
        RowMajorMatrix::new(trace_values, TRACE_WIDTH)
    }
}

impl<F> BaseAir<F> for MulAir {
    fn width(&self) -> usize {
        TRACE_WIDTH
    }
}

impl<AB: AirBuilder> Air<AB> for MulAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let main_local = main.current_slice();
        let main_next = main.next_slice();

        for i in 0..REPETITIONS {
            let start = i * 3;
            let a = main_local[start];
            let b = main_local[start + 1];
            let c = main_local[start + 2];
            builder.assert_zero(a.into().exp_u64(self.degree - 1) * b - c);
            if self.uses_boundary_constraints {
                builder.when_first_row().assert_eq(a * a + AB::Expr::ONE, b);
            }
            if self.uses_transition_constraints {
                let next_a = main_next[start];
                builder
                    .when_transition()
                    .assert_eq(a + AB::Expr::from_u8(REPETITIONS as u8), next_a);
            }
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn do_test<SC: StarkGenericConfig>(
    config: SC,
    air: MulAir,
    log_height: usize,
) -> Result<(), impl Debug>
where
    SC::Challenger: Clone,
    StandardUniform: Distribution<Val<SC>>,
{
    let trace = air.random_valid_trace(log_height, true);

    let proof = prove(&config, &air, trace, &[]);

    let serialized_proof = postcard::to_allocvec(&proof).expect("unable to serialize proof");
    tracing::debug!("serialized_proof len: {} bytes", serialized_proof.len());

    let deserialized_proof =
        postcard::from_bytes(&serialized_proof).expect("unable to deserialize proof");

    verify(&config, &air, &deserialized_proof, &[])
}

fn do_test_bb_trivial(degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    type Pcs = TrivialPcs<Val, Radix2DitParallel<Val>>;
    let pcs = TrivialPcs {
        dft,
        log_n,
        _phantom: PhantomData,
    };
    let challenger = Challenger::new(perm);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir {
        degree,
        ..Default::default()
    };

    do_test(config, air, 1 << log_n)
}

#[test]
fn prove_bb_trivial_deg2() -> Result<(), impl Debug> {
    do_test_bb_trivial(2, 8)
}

#[test]
fn prove_bb_trivial_deg3() -> Result<(), impl Debug> {
    do_test_bb_trivial(3, 8)
}

#[test]
fn prove_bb_trivial_deg4() -> Result<(), impl Debug> {
    do_test_bb_trivial(4, 8)
}

fn do_test_bb_twoadic(log_blowup: usize, degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs =
        MerkleTreeMmcs<<Val as Field>::Packing, <Val as Field>::Packing, MyHash, MyCompress, 2, 8>;
    let val_mmcs = ValMmcs::new(hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 3,
        max_log_arity: 2,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    type Pcs = TwoAdicFriPcs<Val, Dft, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params);
    let challenger = Challenger::new(perm);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir {
        degree,
        ..Default::default()
    };

    do_test(config, air, 1 << log_n)
}

#[test]
fn prove_bb_twoadic_deg2() -> Result<(), impl Debug> {
    do_test_bb_twoadic(1, 2, 5)
}

#[test]
fn prove_bb_twoadic_deg2_zk() -> Result<(), impl Debug> {
    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type Perm = Poseidon2BabyBear<16>;
    let mut rng = SmallRng::seed_from_u64(1);
    let perm = Perm::new_from_rng_128(&mut rng);

    type MyHash = PaddingFreeSponge<Perm, 16, 8, 8>;
    let hash = MyHash::new(perm.clone());

    type MyCompress = TruncatedPermutation<Perm, 2, 8, 16>;
    let compress = MyCompress::new(perm.clone());

    type ValMmcs = MerkleTreeHidingMmcs<
        <Val as Field>::Packing,
        <Val as Field>::Packing,
        MyHash,
        MyCompress,
        SmallRng,
        2,
        8,
        4,
    >;

    let val_mmcs = ValMmcs::new(hash, compress, 0, rng);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Dft = Radix2DitParallel<Val>;
    let dft = Dft::default();

    type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

    let fri_params = FriParameters::new_testing_zk(challenge_mmcs);
    type HidingPcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, SmallRng>;
    let pcs = HidingPcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(1));
    type MyConfig = StarkConfig<HidingPcs, Challenge, Challenger>;
    let challenger = Challenger::new(perm);
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir {
        degree: 3,
        ..Default::default()
    };
    do_test(config, air, 1 << 8)
}

#[test]
fn prove_bb_twoadic_deg3() -> Result<(), impl Debug> {
    do_test_bb_twoadic(1, 3, 5)
}

#[test]
fn prove_bb_twoadic_deg4() -> Result<(), impl Debug> {
    do_test_bb_twoadic(2, 4, 4)
}

#[test]
fn prove_bb_twoadic_deg5() -> Result<(), impl Debug> {
    do_test_bb_twoadic(2, 5, 4)
}

fn do_test_m31_circle(log_blowup: usize, degree: u64, log_n: usize) -> Result<(), impl Debug> {
    type Val = Mersenne31;
    type Challenge = BinomialExtensionField<Val, 3>;

    type ByteHash = Keccak256Hash;
    type FieldHash = SerializingHasher<ByteHash>;
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);

    type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
    let compress = MyCompress::new(byte_hash);

    type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;

    let fri_params = FriParameters {
        log_blowup,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };

    type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let air = MulAir {
        degree,
        uses_boundary_constraints: true,
        uses_transition_constraints: true,
    };

    do_test(config, air, 1 << log_n)
}

#[test]
fn prove_m31_circle_deg2() -> Result<(), impl Debug> {
    do_test_m31_circle(1, 2, 6)
}

#[test]
fn prove_m31_circle_deg3() -> Result<(), impl Debug> {
    do_test_m31_circle(1, 3, 7)
}

#[test]
fn bench_fields() {
    // RUSTFLAGS='-C target-cpu=native' cargo test --release --package p3-uni-stark --test mul_air -- bench_fields --exact --nocapture --include-ignored
    let n = 10_000_000;
    assert_eq!(<Goldilocks as Field>::Packing::WIDTH, 2);
    assert_eq!(<KoalaBear as Field>::Packing::WIDTH, 4);

    let mut a = <Goldilocks as Field>::Packing::from_i32(3);

    let time = std::time::Instant::now();
    for _ in 0..n / <Goldilocks as Field>::Packing::WIDTH {
        a = a * a;
    }
    let _ = black_box(a);
    println!(
        "Goldilocks: {:.3}M muls/sec",
        (n as f64 / time.elapsed().as_secs_f64()) / 1e6
    );

    let time = std::time::Instant::now();
    for _ in 0..n / <Goldilocks as Field>::Packing::WIDTH {
        a = a + a;
    }
    let _ = black_box(a);
    println!(
        "Goldilocks: {:.3}M adds/sec",
        (n as f64 / time.elapsed().as_secs_f64()) / 1e6
    );

    let mut a = <KoalaBear as Field>::Packing::from_i32(3);

    let time = std::time::Instant::now();
    for _ in 0..n / <KoalaBear as Field>::Packing::WIDTH {
        a = a * a;
    }
    let _ = black_box(a);
    println!(
        "KoalaBear: {:.3}M muls/sec",
        (n as f64 / time.elapsed().as_secs_f64()) / 1e6
    );

    let time = std::time::Instant::now();
    for _ in 0..n / <KoalaBear as Field>::Packing::WIDTH {
        a = a + a;
    }
    let _ = black_box(a);
    println!(
        "KoalaBear: {:.3}M adds/sec",
        (n as f64 / time.elapsed().as_secs_f64()) / 1e6
    );
}

#[test]
fn bench_ext_fields() {
    // RUSTFLAGS='-C target-cpu=native' cargo test --release --package p3-uni-stark --test mul_air -- bench_ext_fields --exact --nocapture --include-ignored
    let n = 500_000_000;
    const SCALAR_SLICE_LEN: usize = 256;

    type G3 = CubicTrinomialExtensionField<Goldilocks>;
    type K5 = QuinticTrinomialExtensionField<KoalaBear>;
    type G3P = <G3 as ExtensionField<Goldilocks>>::ExtensionPacking;
    type K5P = <K5 as ExtensionField<KoalaBear>>::ExtensionPacking;
    type GP = <Goldilocks as Field>::Packing;
    type KP = <KoalaBear as Field>::Packing;

    let g_w = GP::WIDTH;
    let k_w = KP::WIDTH;

    println!("Goldilocks packing width: {}", g_w);
    println!("KoalaBear packing width: {}", k_w);

    fn bench_pairwise<TA, TB, TR>(
        n_total: usize,
        width: usize,
        slice_a: &[TA],
        slice_b: &[TB],
        slice_out: &mut [TR],
        mut step: impl FnMut(TA, TB) -> TR,
    ) -> f64
    where
        TA: Copy,
        TB: Copy,
        TR: Copy,
    {
        let slice_len = slice_a.len();
        assert_eq!(slice_b.len(), slice_len);
        assert_eq!(slice_out.len(), slice_len);
        let outer_iters = n_total / (slice_len * width);
        let t = std::time::Instant::now();
        for _ in 0..outer_iters {
            for i in 0..slice_len {
                slice_out[i] = step(slice_a[i], slice_b[i]);
            }
            let _ = black_box(&slice_out);
        }
        let elapsed = t.elapsed().as_secs_f64();
        let total_scalar = outer_iters * slice_len * width;
        (total_scalar as f64 / elapsed) / 1e6
    }

    let mut rng = SmallRng::seed_from_u64(0xC0FFEE);

    let g_packed_len = SCALAR_SLICE_LEN / g_w;
    let k_packed_len = SCALAR_SLICE_LEN / k_w;

    let g3_a: Vec<G3P> = (0..g_packed_len).map(|_| rng.random()).collect();
    let g3_b: Vec<G3P> = (0..g_packed_len).map(|_| rng.random()).collect();
    let gp_a: Vec<GP> = (0..g_packed_len).map(|_| rng.random()).collect();
    let gp_b: Vec<GP> = (0..g_packed_len).map(|_| rng.random()).collect();

    let k5_a: Vec<K5P> = (0..k_packed_len).map(|_| rng.random()).collect();
    let k5_b: Vec<K5P> = (0..k_packed_len).map(|_| rng.random()).collect();
    let kp_a: Vec<KP> = (0..k_packed_len).map(|_| rng.random()).collect();
    let kp_b: Vec<KP> = (0..k_packed_len).map(|_| rng.random()).collect();

    let mut g3_out: Vec<G3P> = vec![G3P::ZERO; g_packed_len];
    let mut gp_out: Vec<GP> = vec![GP::ZERO; g_packed_len];
    let mut k5_out: Vec<K5P> = vec![K5P::ZERO; k_packed_len];
    let mut kp_out: Vec<KP> = vec![KP::ZERO; k_packed_len];

    // Goldilocks measurements.
    let g3_ext_mul = bench_pairwise(n, g_w, &g3_a, &g3_b, &mut g3_out, |a, b| a * b);
    let g3_base_mul = bench_pairwise(n, g_w, &g3_a, &gp_b, &mut g3_out, |a, b| a * b);
    let g3_ext_add = bench_pairwise(n, g_w, &g3_a, &g3_b, &mut g3_out, |a, b| a + b);
    let g_base_mul = bench_pairwise(n, g_w, &gp_a, &gp_b, &mut gp_out, |a, b| a * b);
    let g_base_add = bench_pairwise(n, g_w, &gp_a, &gp_b, &mut gp_out, |a, b| a + b);

    // KoalaBear measurements.
    let k5_ext_mul = bench_pairwise(n, k_w, &k5_a, &k5_b, &mut k5_out, |a, b| a * b);
    let k5_base_mul = bench_pairwise(n, k_w, &k5_a, &kp_b, &mut k5_out, |a, b| a * b);
    let k5_ext_add = bench_pairwise(n, k_w, &k5_a, &k5_b, &mut k5_out, |a, b| a + b);
    let k_base_mul = bench_pairwise(n, k_w, &kp_a, &kp_b, &mut kp_out, |a, b| a * b);
    let k_base_add = bench_pairwise(n, k_w, &kp_a, &kp_b, &mut kp_out, |a, b| a + b);

    let rows: [(&str, f64, f64); 5] = [
        ("ext * ext  ", g3_ext_mul, k5_ext_mul),
        ("ext * base ", g3_base_mul, k5_base_mul),
        ("ext + ext  ", g3_ext_add, k5_ext_add),
        ("base * base", g_base_mul, k_base_mul),
        ("base + base", g_base_add, k_base_add),
    ];

    println!();
    println!(
        "{:<11} | {:>14} | {:>14}",
        "operation", "Goldilocks^3", "KoalaBear^5"
    );
    println!("{:-<12}+{:-<16}+{:-<15}", "", "", "");
    for (label, g, k) in rows {
        println!("{:<11} | {:>10.0} M/s | {:>10.0} M/s", label, g, k);
    }
}
