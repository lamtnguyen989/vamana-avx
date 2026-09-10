import argparse
import pandas as pd


def load_data(file_path: str, max_k: int) -> pd.DataFrame:
    """Reads CSV and returns a filtered DataFrame up to max_k rank."""
    df = pd.read_csv(file_path, usecols=["query_id", "neighbor_rank", "neighbor_id"])
    if max_k is not None:
        df = df[df["neighbor_rank"] <= max_k]
    return df


def calculate_recall_at_k(
    results_df: pd.DataFrame, gt_df: pd.DataFrame, k: int
) -> tuple[float, int]:
    """Computes Mean Recall@K across common queries using Pandas vectorized merges."""
    res_k = results_df[results_df["neighbor_rank"] <= k]
    gt_k = gt_df[gt_df["neighbor_rank"] <= k]

    common_queries = set(res_k["query_id"]).intersection(gt_k["query_id"])
    if not common_queries:
        return 0.0, 0

    # Filter to queries present in both results and ground truth
    res_k = res_k[res_k["query_id"].isin(common_queries)]
    gt_k = gt_k[gt_k["query_id"].isin(common_queries)]

    # Count matching neighbor_ids per query
    matches = res_k.merge(gt_k, on=["query_id", "neighbor_id"])
    match_counts = matches.groupby("query_id").size()

    # Determine ground truth items count per query (handling ground truth < k)
    gt_counts = gt_k.groupby("query_id").size().clip(upper=k)

    # Reindex match_counts to include queries with 0 matches
    match_counts = match_counts.reindex(gt_counts.index, fill_value=0)

    # Compute mean recall across all common queries
    recall = (match_counts / gt_counts).mean()
    return float(recall), len(common_queries)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Recall@K using Pandas."
    )
    parser.add_argument("results_csv", help="Path to ANN search output CSV")
    parser.add_argument("ground_truth_csv", help="Path to ground truth CSV")
    parser.add_argument(
        "k_values",
        type=int,
        nargs="+",
        metavar="K",
        help="One or more values of K (e.g., 1 10 100)",
    )

    args = parser.parse_args()
    max_k = max(args.k_values)

    results_df = load_data(args.results_csv, max_k)
    gt_df = load_data(args.ground_truth_csv, max_k)

    print("\n" + "=" * 50)
    print(f"{'K':<10} | {'Recall@K':<15} | {'Queries Evaluated':<15}")
    print("-" * 50)

    for k in sorted(args.k_values):
        recall, count = calculate_recall_at_k(results_df, gt_df, k)
        print(f"{k:<10} | {recall * 100:>13.2f}% | {count:>15}")

    print("=" * 50)


if __name__ == "__main__":
    main()