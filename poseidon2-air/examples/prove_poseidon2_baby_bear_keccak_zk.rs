use core::fmt::Debug;

use p3_baby_bear::{BabyBear, GenericPoseidon2LinearLayersBabyBear};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_fri::{HidingFriPcs, create_benchmark_fri_params_zk};
use p3_keccak::{Keccak256Hash, KeccakF};
use p3_merkle_tree::MerkleTreeHidingMmcs;
use p3_poseidon2_air::{RoundConstants, VectorizedPoseidon2Air};
use p3_symmetric::{CompressionFunctionFromHasher, PaddingFreeSponge, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};
use rand::SeedableRng;
use rand::rngs::SmallRng;
#[cfg(target_family = "unix")]
use tikv_jemallocator::Jemalloc;
use tracing_forest::ForestLayer;
use tracing_forest::util::LevelFilter;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Registry};

#[cfg(target_family = "unix")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

const WIDTH: usize = 16;
const SBOX_DEGREE: u64 = 7;
const SBOX_REGISTERS: usize = 1;
const HALF_FULL_ROUNDS: usize = 4;
const PARTIAL_ROUNDS: usize = 20;

const NUM_ROWS: usize = 1 << 16;
const VECTOR_LEN: usize = 1 << 3;
const NUM_PERMUTATIONS: usize = NUM_ROWS * VECTOR_LEN;

type Dft = p3_dft::Radix2DitParallel<BabyBear>;

fn main() -> Result<(), impl Debug> {
    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy();

    Registry::default()
        .with(env_filter)
        .with(ForestLayer::default())
        .init();

    type Val = BabyBear;
    type Challenge = BinomialExtensionField<Val, 4>;

    type ByteHash = Keccak256Hash;
    let byte_hash = ByteHash {};

    type U64Hash = PaddingFreeSponge<KeccakF, 25, 17, 4>;
    let u64_hash = U64Hash::new(KeccakF {});

    type FieldHash = SerializingHasher<U64Hash>;
    let field_hash = FieldHash::new(u64_hash);

    type MyCompress = CompressionFunctionFromHasher<U64Hash, 2, 4>;
    let compress = MyCompress::new(u64_hash);

    // WARNING: DO NOT USE SmallRng in proper applications! Use a real PRNG instead!
    type ValMmcs = MerkleTreeHidingMmcs<
        [Val; p3_keccak::VECTOR_LEN],
        [u64; p3_keccak::VECTOR_LEN],
        FieldHash,
        MyCompress,
        SmallRng,
        4,
        4,
    >;
    let mut rng = SmallRng::seed_from_u64(1);
    let constants = RoundConstants::from_rng(&mut rng);
    let val_mmcs = ValMmcs::new(field_hash, compress, rng);

    type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());

    type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
    let challenger = Challenger::from_hasher(vec![], byte_hash);

    let air: VectorizedPoseidon2Air<
        Val,
        GenericPoseidon2LinearLayersBabyBear,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
        VECTOR_LEN,
    > = VectorizedPoseidon2Air::new(constants);

    let fri_params = create_benchmark_fri_params_zk(challenge_mmcs);

    let trace = air.generate_vectorized_trace_rows(NUM_PERMUTATIONS, fri_params.log_blowup);

    let dft = Dft::default();

    type Pcs = HidingFriPcs<Val, Dft, ValMmcs, ChallengeMmcs, SmallRng>;
    let pcs = Pcs::new(dft, val_mmcs, fri_params, 4, SmallRng::seed_from_u64(1));

    type MyConfig = StarkConfig<Pcs, Challenge, Challenger>;
    let config = MyConfig::new(pcs, challenger);

    let proof = prove(&config, &air, trace, &vec![]);

    verify(&config, &air, &proof, &vec![])
}



#[cfg(test)]
fn exp_biguint<F: p3_field::Field>(base: F, exponent: &num_bigint::BigUint) -> F {
    use num_bigint::BigUint;

    if exponent == &BigUint::from(0u8) {
        return F::ONE;
    }

    let mut result = F::ONE;
    let mut current_base = base;

    // Convert the exponent to bytes
    let exponent_bytes = exponent.to_bytes_le();

    // Process 4 bytes (32 bits) at a time
    for chunk_idx in 0..((exponent_bytes.len() + 3) / 4) {
        // Extract the current 32-bit chunk
        let start = chunk_idx * 4;
        let end = std::cmp::min((chunk_idx + 1) * 4, exponent_bytes.len());

        let mut chunk_value: u32 = 0;
        for (i, &byte) in exponent_bytes[start..end].iter().enumerate() {
            chunk_value |= (byte as u32) << (i * 8);
        }

        // If this chunk has a non-zero value, apply it
        if chunk_value > 0 {
            result *= current_base.exp_u64(chunk_value as u64);
        }

        // Prepare the base for the next chunk (base^(2^32))
        current_base = current_base.exp_u64(1u64 << (4 * 8));
    }

    result
}

#[test]
fn test() {
    use num_bigint::BigUint;
    use p3_field::Field;
    use p3_field::PrimeCharacteristicRing;
    // 2^25 × 3^2 × 7 × 67 × 127 × 283 × 1254833 × 9679978477096567 × 1513303300498959019
    let decomposition = [
        (BigUint::new(vec![2]), 30),
        (BigUint::new(vec![3]), 1),
        (BigUint::new(vec![5]), 1),
        (BigUint::new(vec![17]), 1),
        (BigUint::new(vec![31]), 1),
        (BigUint::new(vec![97]), 1),
        (BigUint::new(vec![12241]), 1),
        (BigUint::new(vec![1666201]), 1),
        (BigUint::new(vec![32472031]), 1),
        (BigUint::new(vec![74565857]), 1),
        (BigUint::new(vec![1702001361, 397]), 1),
        (BigUint::new(vec![3175058489, 3577574990, 210]), 1),
    ];

    let mut prod = BigUint::from(1usize);
    for (base, exp) in decomposition.iter() {
        prod *= base.pow(*exp);
    }
    assert_eq!(
        prod,
        BigUint::from((1usize << 31) - (1 << 27) + 1).pow(8) - BigUint::from(1_usize)
    );
    type F = BinomialExtensionField<KoalaBear, 6>;
    assert!(exp_biguint(F::GENERATOR, &prod) == F::ONE);
    for (factor, _) in decomposition.iter() {
        let f = &prod / factor;
        assert!(exp_biguint(F::GENERATOR, &f) != F::ONE);
    }
}
