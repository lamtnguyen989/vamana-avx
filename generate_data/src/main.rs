use std::{io::{BufWriter, Write}, path::{Path, PathBuf}};

use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;
#[derive(Parser, Debug)]
struct Args 
{
    /// Output path: [Internally default relative paths to <PROJECT_ROOT>/<output_file>]
    output_file: PathBuf, 

    /// Total number of vectors to be generated
    #[arg(long = "n-vectors", default_value_t = 50_000)]
    num_vectors: usize, 
    
    /// Dimension of the vector space
    #[arg(long, default_value_t = 128)] 
    dim: usize,

    /// Number of vector clusters
    #[arg(long = "n-clusters", default_value_t = 200)]
    num_clusters: usize,

    /// Standard deviations for each vector blobs
    #[arg(long = "cluster-std-dev", default_value_t = 1.5)] 
    cluster_std_dev: f64, 

    /// Number of threads to be used in generation concurrently
    #[arg(long = "threads", default_value_t = 4)] 
    threads: usize, 
    
    /// RNG seed
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    /// Batch size for vector writes
    #[arg(long, default_value_t = 2000)] 
    batch_size: usize, 
}

fn resolve_output_path(user_path: &Path) -> std::io::Result<PathBuf> {
    // Obey the absolute path outputs
    if user_path.is_absolute() {
        if let Some(parent) = user_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        return Ok(user_path.to_path_buf());
    }

    // Relative path processing
    // Note the default is to write to <PROJECT_ROOT>/<output_file>
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);
    let full_path = project_root.join(user_path);

    // Create all path leading up to the `user_path`
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(full_path)
}

/// Generating clusters centers (deliberately double precision and shouldn't matter due to truncation when writing)
/// Furthermore, the generations is purely sequential and independent from blob vectors generations that follows
fn generate_centers(seed: u64, n_clusters: usize, dim: usize) -> Vec<Vec<f64>>
{
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    return (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.random_range(-10.0..10.0)).collect())
        .collect();
}

/// Generating a single blob vector (deterministically from seed)
fn generate_vector(centers: &[Vec<f64>], dim: usize, std_dev: f64, index: u64, base_seed: u64) -> Vec<f32>
{
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(index));
    let mut out: Vec<f32> = Vec::with_capacity(dim);

    let cluster = &centers[rng.random_range(0..centers.len())];
    for d in 0..dim {
        let normal = Normal::new(cluster[d], std_dev).expect("Invalid std_dev (must be > 0)");
        out.push(normal.sample(&mut rng) as f32);
    }

    return out;
}

/// Generate vector blobs and serialize to the vecfile `.vecf` format
fn generate_vecfile(args: &Args) -> std::io::Result<()> 
{
    // Open up file for serializing
    let vecfile = std::fs::File::create(&args.output_file)?;

    // Initialize the buffered writer (mmap possibility later but now safety first)
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, vecfile);

    // Writing header
    writer.write_all(&(args.num_vectors as u32).to_le_bytes())?;
    writer.write_all(&(args.dim as u32).to_le_bytes())?;

    // Generate blob centers
    let centers = generate_centers(args.seed, args.num_clusters, args.dim);

    // Generate blob vectors surrounding the centers
    let mut start = 0_usize;
    while start < args.num_vectors {
        // Marking the batch work boundary
        let end = (start + args.batch_size).min(args.num_vectors);

        // Generate batch concurrently via rayon
        let batch: Vec<Vec<f32>> = (start..end)
                                        .into_par_iter()
                                        .map(|k| generate_vector(&centers, args.dim, args.cluster_std_dev, k as u64, args.seed))
                                        .collect();

        // Prep byte buffer for writing contiguous FP32s
        let mut byte_buf = Vec::with_capacity((end - start) * args.dim * 4);
        for vec in &batch {
            for &float in vec {
                byte_buf.extend_from_slice(&float.to_le_bytes());
            }
        }
        // Write
        writer.write_all(&byte_buf)?;

        // Advance to the start of next batch
        start = end;
    }
    // Flush the byte buffer for remaining vectors
    writer.flush()?;

    Ok(())
}

fn main() -> std::io::Result<()> {
    // Parsing CLI
    let mut args = Args::parse();

    // Resolving output path
    args.output_file = resolve_output_path(&args.output_file)?;

    // Setting up Rayon threadpool
    let rayon_setup = rayon::ThreadPoolBuilder::new()
                        .num_threads(args.threads)
                        .build_global();
    match rayon_setup {
        Ok(_) => println!("Global threadpool initialized successfully."),
        Err(e) => panic!("Failed to Initialize global threadpool. Error: {}", e)
    }

    // Generate data and serialize to file
    generate_vecfile(&args)?;


    println!("Wrote data to {:?}", args.output_file);

    Ok(())
}
