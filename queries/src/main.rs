use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use memmap2::Mmap;
use rayon::prelude::*;

const HEADER_BYTES: usize = 8;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Metric {
    L2,
    Cosine,
}

#[derive(Parser, Debug)]
struct Args
{
    /// Path to the input dataset in .vecf serialization format
    data_path: PathBuf,

    /// Output path for the generated queries (.vecf format)
    query_out_path: PathBuf,

    /// Output path for the ground truth CSV
    ground_truth_csv: PathBuf,

    /// Number of queries to generate
    #[arg(short = 'n', long = "n-queries", default_value_t = 1000)]
    n_queries: usize,

    /// Number of nearest neighbors to compute per query
    #[arg(short = 'k', long, default_value_t = 100)]
    k: usize,

    /// Standard deviation of the Gaussian noise added to sampled base points
    #[arg(long = "std-dev", default_value_t = 5.5)]
    std_dev: f64,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Distance metric used for ground truth computation
    #[arg(long, value_enum, default_value_t = Metric::L2)]
    metric: Metric,

    /// Batch size used when computing ground truths
    #[arg(long = "batch-size", default_value_t = 2000)]
    batch_size: usize,

    /// Number of threads to be used in generation concurrently
    #[arg(short, long = "threads", default_value_t = 4)] 
    threads: usize, 
}

/// Mmap'd .vecf file wrapper
struct Vecf 
{
    _mmap: Mmap,
    n_vectors: u32,
    dim: u32
}

impl Vecf
{
    fn open(path: &Path) -> std::io::Result<Self> {
        todo!();
    }
}

fn main() -> std::io::Result<()> {

    // Parsing CLI 
    let args = Args::parse();

    // Setting up Rayon threadpool
    let rayon_setup = rayon::ThreadPoolBuilder::new()
                        .num_threads(args.threads)
                        .build_global();
    match rayon_setup {
        Ok(_) => println!("Global threadpool initialized successfully."),
        Err(e) => panic!("Failed to initialize global threadpool. Error: {}", e)
    }

    println!("Hello, world!");

    Ok(())
}
