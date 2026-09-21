mod query_item;
mod vecf;

use std::{collections::BinaryHeap, io::{BufWriter, Write}, path::{Path, PathBuf}};

use clap::{Parser, ValueEnum};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::{query_item::{QueryItem, SortQueryItemsExt}, vecf::{Vecf, WRITE_BUFFER_SIZE, write_vecf}};


#[derive(Copy, Clone, Debug, ValueEnum)]
enum Metric {
    L2,
    Cosine,
}

#[derive(Debug, Parser)]
struct Args
{
    /// Data file path: [Internally default relative paths to <PROJECT_ROOT>/<data_path>]
    data_path: PathBuf,

    /// Queries output path [Internally default relative paths to <PROJECT_ROOT>/<queries_out_path>]
    queries_out_path: PathBuf,

    /// Output path for the ground truth CSV [Internally default relative paths to <PROJECT_ROOT>/<ground_truths_csv>]
    ground_truths_csv: PathBuf,

    /// Number of queries to generate
    #[arg(short = 'n', long = "n-queries", default_value_t = 1000)]
    n_queries: usize,

    /// Number of nearest neighbors to compute per query
    #[arg(short = 'k', long, default_value_t = 100)]
    neighbors_count: usize,

    /// Standard deviation of the Gaussian noise added to sampled base points
    #[arg(short = 's', long = "std-dev", default_value_t = 0.5)]
    std_dev: f64,

    /// Number of threads to be used in generation concurrently
    #[arg(short, long = "threads", default_value_t = 4)] 
    threads: usize, 

    /// RNG seed
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    /// Batch size for vector writes
    #[arg(short, long, default_value_t = 2000)] 
    batch_size: usize, 

    /// Distance metric used for ground truth computation
    #[arg(long, value_enum, default_value_t = Metric::L2)]
    metric: Metric,
}

/// Resolving user output file paths
/// Absolute paths are honered as-is
/// Realtive paths are default to <PROJECT_ROOT>/<output_file>
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

/// Resolving user input file paths
/// Absolute paths are honered as-is
/// Realtive paths are default to <PROJECT_ROOT>/<input_file>
fn resolve_input_path(user_path: &Path) -> std::io::Result<PathBuf> {
    if user_path.is_absolute() {
        return Ok(user_path.to_path_buf());
    }
 
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);

    Ok(project_root.join(user_path))
}

/// Generating queries from the data set
fn generate_queries(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    std_dev: f64,
    base_seed: u64,
) -> Vec<f32> {
    // Logic between each queries generation are independent from one another
    // So parallelized between queries but generating logic stays sequential
    return (0..n_queries).into_par_iter()
                        .flat_map_iter(|q_id| {
                            // Initialize randomness and noise
                            let mut rng = Xoshiro256PlusPlus::seed_from_u64(base_seed.wrapping_add(q_id as u64));
                            let base_idx = rng.random_range(0..n_vectors);
                            let noise = Normal::new(0.0_f64, std_dev).expect("Invalid Normal sampling configuration.");

                            // Adding noise to data sampling and call it a query vector
                            let base_vector = &data[base_idx*dim..(base_idx+1)*dim];
                            return (0..dim).map(move |d| {
                                (base_vector[d] as f64 + noise.sample(&mut rng)) as f32
                            });
                        })
                        .collect();
}

/// Computing L2 nearest neighbors with respect to a single query
fn knn_l2(
    query: &[f32], 
    data: &[f32], 
    n_vectors: usize, 
    dim: usize, 
    k: usize,
    idx_out: &mut [u32],
    dists_out: &mut [f32],
){
    let mut heap: BinaryHeap<QueryItem> = BinaryHeap::with_capacity(k+1);

    for i in 0..n_vectors {
        // Computing L2 distance from query vector to the data vector
        let data_vector = &data[i*dim..(i+1)*dim];
        let mut diff_sq = 0.0_f32;
        for j in 0..dim {
            diff_sq += (data_vector[j] - query[j]) * (data_vector[j] - query[j]);
        }
        let dist = diff_sq.sqrt();

        // Push Item onto the max-binary heap
        if heap.len() < k {
            heap.push(QueryItem { dist: dist, id: i as u32 });
        }
        else if let Some(worst) = heap.peek() {
            if dist < worst.dist {
                heap.pop();
                heap.push(QueryItem { dist: dist, id: i as u32 });
            }
        }
    }
    heap.sort_query_into(idx_out, dists_out);
}

/// Computing ground truths with respect to L2-metric
fn ground_truths_l2(
    data: &[f32],
    n_vectors: usize,
    dim: usize,
    queries: &[f32],
    n_queries: usize,
    k: usize,
    batch_size: usize,
) -> (Vec<u32>, Vec<f32>)
{
    let mut gt_indices = vec![0_u32; n_queries * k];
    let mut gt_distances = vec![0.0_f32; n_queries * k];

    let mut start = 0_usize;
    while start < n_queries {
        // Setting batch work boundary
        let end = (start + batch_size).min(n_queries);

        // Computing ground truths from queries concurrently from the batch
        gt_indices[start*k..end*k].par_chunks_mut(k)
                                    .zip(gt_distances[start*k..end*k].par_chunks_mut(k))
                                    .zip((start..end).into_par_iter())
                                    .for_each(|((id_chunk, d_chunk), q)| {
                                        let query = &queries[q*dim..(q+1)*dim];
                                        knn_l2(query, data, n_vectors, dim, k, id_chunk, d_chunk);
                                    });


        // Advance to the next batch     
        start = end;
    }

    return (gt_indices, gt_distances);
}

fn write_ground_truths_csv(
    path: &Path, 
    gt_indices: &[u32], 
    gt_dists: &[f32], 
    n_queries: usize, 
    k: usize,
) -> std::io::Result<()> {
    // Initialize file writing
    let result_file = std::fs::File::create(path)?;
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, result_file);

    // Write results
    writer.write_all("query_id,neighbor_rank,neighbor_id,distance\n".as_bytes())?;
    for query in 0..n_queries {
        for rank in 0..k {
            let q_idx = query*k + rank;
            writeln!(writer, "{},{},{},{:.6}", query, rank+1, gt_indices[q_idx], gt_dists[q_idx])?;
        }
    }
    writer.flush()?;
    
    Ok(())
}

fn main() -> std::io::Result<()> 
{
    // Parsing CLI
    let mut args = Args::parse();

    // Resolving paths
    args.data_path = resolve_input_path(&args.data_path)?;
    args.queries_out_path = resolve_output_path(&args.queries_out_path)?;
    args.ground_truths_csv = resolve_output_path(&args.ground_truths_csv)?;


    // Setting up Rayon threadpool
    if args.threads < 1 {
        eprintln!("Thread count can not be less than 1! Default thread count to 1.");
        args.threads = 1;
    }
    rayon::ThreadPoolBuilder::new()
                        .num_threads(args.threads)
                        .build_global()
                        .unwrap_or_else(|e| panic!("Failed to initialize global threadpool. Error: {}", e));
    // Loading dataset
    let vecf = Vecf::open(&args.data_path)?;
    let n_vectors = vecf.n_vectors as usize;
    let dim = vecf.dim as usize;
    let data = vecf.data();
    println!("Loaded dataset has {n_vectors} points in {dim} dimensions.");

    // Generate queries
    let queries = generate_queries(data, n_vectors, dim, args.n_queries, args.std_dev, args.seed);
    write_vecf(&args.queries_out_path, &queries, args.n_queries as u32, vecf.dim)?;
    println!("Write {} queries (Gaussian std={}) to {:?}", args.n_queries, args.std_dev, args.queries_out_path);

    // Compute ground truths
    println!("Computing top-{} ground truths using {:?} metric...", args.neighbors_count, args.metric);
    let (gt_indices, gt_dists) = match args.metric {
        Metric::L2 => {ground_truths_l2(data, n_vectors, dim, &queries, args.n_queries, args.neighbors_count, args.batch_size)}
        Metric::Cosine => {todo!("Cosine metric has not been implemented yet!")}
    };

    // Serialize ground truths to CSV
    println!("Saving ground truth CSV to {:?}...", args.ground_truths_csv);
    write_ground_truths_csv(&args.ground_truths_csv, &gt_indices, &gt_dists, args.n_queries, args.neighbors_count)?;
    println!("Done!");

    Ok(())
}
