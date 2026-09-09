use std::{io::{BufWriter, Write}, path::{Path, PathBuf}};

use clap::{Parser, ValueEnum};
use memmap2::Mmap;
use rayon::prelude::*;

const HEADER_BYTES: usize = 8;
const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

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
    /// Loading vecfile via mmap
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Very crude file size error checking
        if mmap.len() < HEADER_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "File size smaller than header size",
            ));
        }
        // Decode the headers
        let n_vectors = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap());

        // Size checking again
        let expected_bytes = HEADER_BYTES + (n_vectors as usize) * (dim as usize) * 4;
        if mmap.len() < expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("File size mismatch: expected {expected_bytes} bytes, found {}", mmap.len()),
            ));
        }
        

        return Ok(Self {_mmap: mmap, n_vectors, dim});
    }
}

/// Writing .vecf serialization
fn write_vecf(path: &Path, vectors_data: &[f32], n_vectors: u32, dim: u32) -> std::io::Result<()>
{

    Ok(())
}

/// Computing queries ground truths under L2 metric
fn compute_ground_truths_l2(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    queries: &[f32],
    n_queries: usize,
    k: usize,
) -> (Vec<u32>, Vec<f32>) 
{
    todo!();
}


/// Saving ground truths results to CSV
fn save_ground_truths_csv(
    path: &Path, 
    gt_indices: &[u32], 
    gt_distances: &[f32], 
    n_queries: usize, 
    k: usize,
) -> std::io::Result<()> 
{
    let result_file = std::fs::File::create(path)?;
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, result_file);

    writeln!(writer, "query_id,neighbor_rank,neighbor_id,distance")?;
    for q in 0..n_queries {
        for r in 0..k {
            let q_idx = q*k + r;
            writeln!(writer, "{},{},{},{}", q, r+1, gt_indices[q_idx], gt_distances[q_idx])?;
        }
    }
    writer.flush()?;

    Ok(())
}

fn main() -> std::io::Result<()> {

    // Parsing CLI 
    let mut args = Args::parse();

    // Setting up Rayon threadpool
    if args.threads < 1 {
        eprintln!("Thread count can not be less than 1! Default thread count to 1.");
        args.threads = 1;
    }

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
