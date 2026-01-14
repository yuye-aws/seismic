#!/usr/bin/env python3
"""
Manually calculate Recall@10 by comparing search results against v2_ground_truth_int.txt

This script:
1. Loads the groundtruth from v2_ground_truth_int.txt (10 relevant docs per query)
2. Loads the search results file
3. Calculates Recall@10 for each query
4. Reports overall statistics
"""

import pandas as pd
import numpy as np
import argparse
from collections import defaultdict


def load_groundtruth_from_txt(groundtruth_file, query_ids_file):
    """
    Load groundtruth from v2_ground_truth_int.txt format.
    
    Format: Each line contains comma-separated document IDs (10 docs per query)
    Line number corresponds to query index (0-based)
    
    Returns:
        dict: {query_index: set of relevant doc_ids}
    """
    print(f"Loading groundtruth from: {groundtruth_file}")
    
    # Load query IDs mapping
    query_ids = np.load(query_ids_file)
    print(f"Loaded {len(query_ids)} query IDs")
    
    groundtruth = {}
    total_docs = 0
    
    with open(groundtruth_file, 'r') as f:
        for query_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Parse comma-separated document IDs
            doc_ids = [int(doc_id.strip()) for doc_id in line.split(',')]
            groundtruth[query_idx] = set(doc_ids)
            total_docs += len(doc_ids)
    
    print(f"Loaded groundtruth for {len(groundtruth)} queries")
    print(f"Total relevant documents: {total_docs}")
    print(f"Docs per query: {total_docs / len(groundtruth):.1f}")
    
    return groundtruth, query_ids


def load_results(results_file):
    """
    Load search results from TSV file.
    
    Format: query_id \t doc_id \t rank \t score
    
    Returns:
        dict: {query_index: list of (doc_id, rank, score) tuples}
    """
    print(f"\nLoading results from: {results_file}")
    
    results_df = pd.read_csv(results_file, sep='\t', names=['query_id', 'doc_id', 'rank', 'score'])
    
    print(f"Total entries: {len(results_df)}")
    print(f"Queries: {len(results_df['query_id'].unique())}")
    
    # Group by query
    results = defaultdict(list)
    for _, row in results_df.iterrows():
        query_idx = int(row['query_id'])
        doc_id = int(row['doc_id'])
        rank = int(row['rank'])
        score = float(row['score'])
        results[query_idx].append((doc_id, rank, score))
    
    # Sort by rank for each query
    for query_idx in results:
        results[query_idx].sort(key=lambda x: x[1])  # Sort by rank
    
    return dict(results)


def calculate_recall_at_k(groundtruth, results, k=10):
    """
    Calculate Recall@K for each query.
    
    Args:
        groundtruth: dict {query_idx: set of relevant doc_ids}
        results: dict {query_idx: list of (doc_id, rank, score)}
        k: number of top results to consider
    
    Returns:
        dict with recall statistics
    """
    print(f"\nCalculating Recall@{k}...")
    
    recall_per_query = []
    total_relevant = 0
    total_retrieved_relevant = 0
    
    queries_with_results = set(results.keys())
    queries_with_groundtruth = set(groundtruth.keys())
    
    # Check for missing queries
    missing_results = queries_with_groundtruth - queries_with_results
    if missing_results:
        print(f"⚠️  WARNING: {len(missing_results)} queries have groundtruth but no results!")
    
    extra_results = queries_with_results - queries_with_groundtruth
    if extra_results:
        print(f"⚠️  WARNING: {len(extra_results)} queries have results but no groundtruth!")
    
    # Calculate recall for each query
    for query_idx in sorted(queries_with_groundtruth):
        relevant_docs = groundtruth[query_idx]
        
        if query_idx not in results:
            # No results for this query
            recall_per_query.append(0.0)
            total_relevant += len(relevant_docs)
            continue
        
        # Get top-k results
        top_k_results = results[query_idx][:k]
        retrieved_docs = set([doc_id for doc_id, _, _ in top_k_results])
        
        # Calculate recall
        relevant_retrieved = relevant_docs.intersection(retrieved_docs)
        recall = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
        
        recall_per_query.append(recall)
        total_relevant += len(relevant_docs)
        total_retrieved_relevant += len(relevant_retrieved)
    
    # Calculate statistics
    recall_array = np.array(recall_per_query)
    
    stats = {
        'mean_recall': np.mean(recall_array),
        'median_recall': np.median(recall_array),
        'min_recall': np.min(recall_array),
        'max_recall': np.max(recall_array),
        'std_recall': np.std(recall_array),
        'total_relevant': total_relevant,
        'total_retrieved_relevant': total_retrieved_relevant,
        'overall_recall': total_retrieved_relevant / total_relevant if total_relevant > 0 else 0.0,
        'num_queries': len(recall_per_query),
        'recall_per_query': recall_array,
    }
    
    return stats


def print_statistics(stats, k=10):
    """Print detailed statistics about Recall@K"""
    
    print("\n" + "="*70)
    print(f"Recall@{k} Results")
    print("="*70)
    
    print(f"\nOverall Recall@{k}: {stats['overall_recall']:.6f} ({stats['overall_recall']*100:.2f}%)")
    print(f"Mean Recall@{k}: {stats['mean_recall']:.6f} ({stats['mean_recall']*100:.2f}%)")
    
    print(f"\nTotal relevant docs: {stats['total_relevant']:,}")
    print(f"Total retrieved: {stats['total_retrieved_relevant']:,}")
    print(f"Missed: {stats['total_relevant'] - stats['total_retrieved_relevant']:,}")
    
    print(f"\nRecall distribution:")
    print(f"  Mean: {stats['mean_recall']:.4f}")
    print(f"  Median: {stats['median_recall']:.4f}")
    print(f"  Min: {stats['min_recall']:.4f}")
    print(f"  Max: {stats['max_recall']:.4f}")
    print(f"  Std: {stats['std_recall']:.4f}")
    
    # Count queries by recall value
    recall_counts = {}
    for recall in stats['recall_per_query']:
        recall_rounded = round(recall, 1)
        recall_counts[recall_rounded] = recall_counts.get(recall_rounded, 0) + 1
    
    print(f"\nRecall histogram:")
    for recall_val in sorted(recall_counts.keys()):
        count = recall_counts[recall_val]
        pct = count / stats['num_queries'] * 100
        bar = '█' * int(pct / 2)
        print(f"  {recall_val:.1f}: {count:4d} queries ({pct:5.1f}%) {bar}")
    
    # Special cases
    perfect_recall = np.sum(stats['recall_per_query'] == 1.0)
    zero_recall = np.sum(stats['recall_per_query'] == 0.0)
    
    print(f"\nSpecial cases:")
    print(f"  Queries with perfect recall (1.0): {perfect_recall} ({perfect_recall/stats['num_queries']*100:.1f}%)")
    print(f"  Queries with zero recall (0.0): {zero_recall} ({zero_recall/stats['num_queries']*100:.1f}%)")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(stats['recall_per_query'], p)
        print(f"  {p}th percentile: {val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Recall@10 from groundtruth and results files"
    )
    parser.add_argument(
        "--groundtruth",
        required=True,
        help="Path to v2_ground_truth_int.txt"
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results TSV file (e.g., results_recall_90)"
    )
    parser.add_argument(
        "--query-ids",
        required=True,
        help="Path to queries_ids.npy"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="K value for Recall@K (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Load data
    groundtruth, query_ids = load_groundtruth_from_txt(args.groundtruth, args.query_ids)
    results = load_results(args.results)
    
    # Calculate recall
    stats = calculate_recall_at_k(groundtruth, results, k=args.k)
    
    # Print statistics
    print_statistics(stats, k=args.k)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
