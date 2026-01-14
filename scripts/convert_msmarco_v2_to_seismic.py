#!/usr/bin/env python3

import scipy.sparse as sp
import numpy as np
import struct
import os
from tqdm import tqdm
import argparse
import pandas as pd

def read_csr_file(fname):
    """Read custom CSR format file"""
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype="int32", count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype="float32", count=nnz)
        return data, indices, indptr, nrow, ncol

def write_sparse_vectors_to_binary_file(filename, vectors):
    """
    Write sparse vectors to binary file in Seismic format.
    A binary sequence is a sequence of integers prefixed by its length,
    where both the sequence integers and the length are written as 32-bit little-endian unsigned integers.
    Followed by a sequence of f32, with the same length
    """
    def write_binary_sequence(indices, values, file):
        file.write((len(indices)).to_bytes(4, byteorder="little", signed=False))
        for idx in indices:
            file.write((int(idx)).to_bytes(4, byteorder="little", signed=False))
        for val in values:
            ba = bytearray(struct.pack("f", float(val)))
            file.write(ba)

    with open(filename, "wb") as fout:
        fout.write((len(vectors)).to_bytes(4, byteorder="little", signed=False))
        for indices, values in tqdm(vectors, desc=f"Writing {os.path.basename(filename)}"):
            write_binary_sequence(indices, values, fout)

def extract_doc_ids_from_qrels(qrels_file):
    """Extract unique document IDs from MSMarco v2 qrels file"""
    print(f"Extracting document IDs from {qrels_file}")
    
    # Read qrels file
    df = pd.read_csv(qrels_file, sep='\t')
    
    # Extract unique document IDs
    unique_doc_ids = df['doc_id'].unique()
    
    # Create mapping from doc_id to integer index
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}
    
    print(f"Found {len(unique_doc_ids)} unique document IDs")
    return unique_doc_ids, doc_id_to_idx


def get_num_docs_from_csr(csr_file):
    """Get the number of documents from a CSR file"""
    with open(csr_file, "rb") as f:
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        return nrow

def convert_qrels_format_v2(input_file, output_file, doc_id_to_idx):
    """Convert MS MARCO v2 qrels format to seismic format with integer doc IDs"""
    print(f"Converting qrels format from {input_file} to {output_file}")
    
    df = pd.read_csv(input_file, sep='\t')
    
    with open(output_file, 'w') as f_out:
        for _, row in df.iterrows():
            query_id = row['query_id']
            doc_id_str = row['doc_id']
            relevance = row['relevance']
            
            # Extract numeric ID from msmarco_passage_XX_YYYYYYY format
            if isinstance(doc_id_str, str) and doc_id_str.startswith('msmarco_passage_'):
                parts = doc_id_str.split('_')
                if len(parts) >= 4:
                    try:
                        numeric_doc_id = int(parts[-1])
                        if numeric_doc_id in doc_id_to_idx:
                            doc_id_idx = doc_id_to_idx[numeric_doc_id]
                            # Write in expected format: query_id \t 0 \t doc_id_idx \t relevance
                            f_out.write(f"{query_id}\t0\t{doc_id_idx}\t{relevance}\n")
                    except ValueError:
                        continue

def create_groundtruth_from_v2_file(groundtruth_file, output_file, doc_id_to_idx, query_ids_array):
    """Create groundtruth.tsv from v2_ground_truth_int.txt file for MSMarco v2"""
    print(f"Creating groundtruth file from {groundtruth_file}")
    
    groundtruth_entries = []
    missing_docs = set()
    
    with open(groundtruth_file, 'r') as f:
        for query_idx, line in enumerate(tqdm(f, desc="Processing ground truth")):
            line = line.strip()
            if not line:
                continue
            
            # Parse comma-separated document IDs (each line has 10 docs for recall@10)
            doc_ids_str = line.split(',')
            
            for rank, doc_id_str in enumerate(doc_ids_str, 1):
                try:
                    doc_id = int(doc_id_str.strip())
                    
                    if doc_id in doc_id_to_idx:
                        doc_index = doc_id_to_idx[doc_id]
                        groundtruth_entries.append([query_idx, doc_index, rank, 1.0])
                    else:
                        missing_docs.add(doc_id)
                            
                except ValueError:
                    print(f"Warning: Invalid doc ID '{doc_id_str}' in query {query_idx}")
                    continue
    
    print(f"Created {len(groundtruth_entries)} groundtruth entries")
    print(f"Missing documents: {len(missing_docs)}")
    
    if missing_docs and len(missing_docs) < 20:
        print(f"Missing doc IDs: {list(missing_docs)}")
    
    # Save groundtruth
    if groundtruth_entries:
        df = pd.DataFrame(groundtruth_entries, columns=['query_id', 'doc_id', 'rank', 'score'])
        df.to_csv(output_file, sep='\t', header=False, index=False)
        
        print(f"Query index range: {df['query_id'].min()} to {df['query_id'].max()}")
        print(f"Doc index range: {df['doc_id'].min()} to {df['doc_id'].max()}")
        print(f"Saved groundtruth to: {output_file}")
        
        return True
    else:
        print("ERROR: No valid groundtruth entries created!")
        return False


def create_groundtruth_from_v2_file_direct(groundtruth_file, output_file, num_docs):
    """
    Create groundtruth.tsv from v2_ground_truth_int.txt file for MSMarco v2.
    The doc IDs in groundtruth are already the CSR matrix indices, so no mapping needed.
    """
    print(f"Creating groundtruth file from {groundtruth_file}")
    
    groundtruth_entries = []
    invalid_docs = []
    
    with open(groundtruth_file, 'r') as f:
        for query_idx, line in enumerate(tqdm(f, desc="Processing ground truth")):
            line = line.strip()
            if not line:
                continue
            
            # Parse comma-separated document IDs (each line has 10 docs for recall@10)
            doc_ids_str = line.split(',')
            
            for rank, doc_id_str in enumerate(doc_ids_str, 1):
                try:
                    doc_id = int(doc_id_str.strip())
                    
                    # Validate doc_id is within corpus range
                    if 0 <= doc_id < num_docs:
                        # Doc ID is already the index, use directly
                        groundtruth_entries.append([query_idx, doc_id, rank, 1.0])
                    else:
                        invalid_docs.append(doc_id)
                            
                except ValueError:
                    print(f"Warning: Invalid doc ID '{doc_id_str}' in query {query_idx}")
                    continue
    
    print(f"Created {len(groundtruth_entries)} groundtruth entries")
    if invalid_docs:
        print(f"Invalid documents (out of range): {len(invalid_docs)}")
        if len(invalid_docs) < 20:
            print(f"Invalid doc IDs: {invalid_docs}")
    
    # Save groundtruth
    if groundtruth_entries:
        df = pd.DataFrame(groundtruth_entries, columns=['query_id', 'doc_id', 'rank', 'score'])
        df.to_csv(output_file, sep='\t', header=False, index=False)
        
        print(f"Query index range: {df['query_id'].min()} to {df['query_id'].max()}")
        print(f"Doc index range: {df['doc_id'].min()} to {df['doc_id'].max()}")
        print(f"Saved groundtruth to: {output_file}")
        
        return True
    else:
        print("ERROR: No valid groundtruth entries created!")
        return False


def convert_qrels_format_v2_direct(input_file, output_file):
    """
    Convert MS MARCO v2 qrels format to seismic format.
    Extract the numeric doc ID from msmarco_passage_XX_YYYYYYY format.
    """
    print(f"Converting qrels format from {input_file} to {output_file}")
    
    df = pd.read_csv(input_file, sep='\t')
    
    with open(output_file, 'w') as f_out:
        for _, row in df.iterrows():
            query_id = row['query_id']
            doc_id_str = row['doc_id']
            relevance = row['relevance']
            
            # Extract numeric ID from msmarco_passage_XX_YYYYYYY format
            if isinstance(doc_id_str, str) and doc_id_str.startswith('msmarco_passage_'):
                parts = doc_id_str.split('_')
                if len(parts) >= 4:
                    try:
                        numeric_doc_id = int(parts[-1])
                        # Write in expected format: query_id \t 0 \t doc_id \t relevance
                        f_out.write(f"{query_id}\t0\t{numeric_doc_id}\t{relevance}\n")
                    except ValueError:
                        continue

def extract_doc_ids_from_groundtruth_and_qrels(qrels_file, groundtruth_file):
    """Extract document IDs from both qrels and ground truth files for validation"""
    print(f"Extracting document IDs from {qrels_file} and {groundtruth_file}")
    
    # Get doc IDs from qrels
    df_qrels = pd.read_csv(qrels_file, sep='\t')
    qrels_doc_ids = set(df_qrels['doc_id'].unique())
    print(f"Found {len(qrels_doc_ids)} unique document IDs in qrels")
    
    # Get doc IDs from ground truth
    groundtruth_doc_ids = set()
    with open(groundtruth_file, 'r') as f:
        for line in tqdm(f, desc="Processing ground truth"):
            line = line.strip()
            if not line:
                continue
            doc_ids_str = line.split(',')
            for doc_id_str in doc_ids_str:
                try:
                    doc_id = int(doc_id_str.strip())
                    groundtruth_doc_ids.add(doc_id)
                except ValueError:
                    continue
    
    print(f"Found {len(groundtruth_doc_ids)} unique document IDs in ground truth")
    
    # Extract numeric IDs from qrels doc IDs (msmarco_passage_XX_YYYYYYY format)
    qrels_numeric_ids = set()
    for doc_id_str in qrels_doc_ids:
        if isinstance(doc_id_str, str) and doc_id_str.startswith('msmarco_passage_'):
            parts = doc_id_str.split('_')
            if len(parts) >= 4:
                try:
                    numeric_id = int(parts[-1])  # Last part is the numeric ID
                    qrels_numeric_ids.add(numeric_id)
                except ValueError:
                    continue
    
    print(f"Extracted {len(qrels_numeric_ids)} numeric IDs from qrels")
    
    return groundtruth_doc_ids, qrels_numeric_ids


def create_doc_ids_from_corpus(num_docs):
    """
    Create doc_ids array for all documents in the corpus.
    For MS MARCO v2, documents are indexed sequentially from 0 to num_docs-1.
    The doc_ids array maps each index to itself (identity mapping).
    """
    print(f"Creating doc_ids array for {num_docs} documents")
    # For MS MARCO v2, the document index in the CSR matrix IS the document ID
    # The groundtruth file already uses these integer indices
    doc_ids_array = np.arange(num_docs, dtype=np.int64)
    return doc_ids_array

def convert_csr_to_seismic_format(csr_file, output_dir, file_type="documents"):
    """Convert CSR matrix to Seismic binary format"""
    
    print(f"Loading {csr_file}...")
    
    try:
        # Load using custom CSR format
        data, indices, indptr, nrow, ncol = read_csr_file(csr_file)
        csr_matrix = sp.csr_matrix((data, indices, indptr), shape=(nrow, ncol))
        
        print(f"Matrix shape: {csr_matrix.shape}")
        print(f"Matrix nnz: {csr_matrix.nnz}")
        print(f"Matrix dtype: {csr_matrix.dtype}")
        
    except Exception as e:
        print(f"Error: Could not load {csr_file}: {e}")
        return None
    
    # Convert to list of (indices, values) pairs
    vectors = []
    for i in tqdm(range(csr_matrix.shape[0]), desc="Converting vectors"):
        row = csr_matrix.getrow(i)
        indices = row.indices
        values = row.data
        vectors.append((indices, values))
    
    # Write to binary file
    output_file = os.path.join(output_dir, f"{file_type}.bin")
    write_sparse_vectors_to_binary_file(output_file, vectors)
    
    print(f"Converted {len(vectors)} vectors to {output_file}")
    return vectors

def main():
    parser = argparse.ArgumentParser(description="Convert MS MARCO v2 CSR format to Seismic binary format")
    parser.add_argument("--input-dir", required=True, help="Input directory containing MS MARCO v2 CSR files")
    parser.add_argument("--output-dir", required=True, help="Output directory for binary files")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Check for required files
    qrels_file = os.path.join(input_dir, "qrels.tsv")
    groundtruth_file = os.path.join(input_dir, "v2_ground_truth_int.txt")
    docs_file = os.path.join(input_dir, "merged_passages.csr")
    
    if not os.path.exists(qrels_file):
        print(f"Error: {qrels_file} not found")
        return
    
    if not os.path.exists(groundtruth_file):
        print(f"Error: {groundtruth_file} not found")
        return
    
    if not os.path.exists(docs_file):
        print(f"Error: {docs_file} not found")
        return
    
    # Get the number of documents from the CSR file
    num_docs = get_num_docs_from_csr(docs_file)
    print(f"Total documents in corpus: {num_docs}")
    
    # Create doc_ids array for ALL documents in the corpus
    # For MS MARCO v2, document indices in CSR are the document IDs used in groundtruth
    doc_ids_array = create_doc_ids_from_corpus(num_docs)
    np.save(os.path.join(data_dir, "doc_ids.npy"), doc_ids_array)
    print(f"Saved {len(doc_ids_array)} document IDs to doc_ids.npy")
    
    # Validate against groundtruth and qrels
    groundtruth_doc_ids, qrels_numeric_ids = extract_doc_ids_from_groundtruth_and_qrels(qrels_file, groundtruth_file)
    max_gt_doc_id = max(groundtruth_doc_ids) if groundtruth_doc_ids else 0
    print(f"Max document ID in groundtruth: {max_gt_doc_id}")
    if max_gt_doc_id >= num_docs:
        print(f"WARNING: Groundtruth contains doc IDs ({max_gt_doc_id}) >= num_docs ({num_docs})")
    else:
        print(f"Validation passed: all groundtruth doc IDs are within corpus range")
    
    # Convert documents
    print("Converting merged passages file...")
    convert_csr_to_seismic_format(docs_file, data_dir, "documents")
    
    # Convert queries  
    queries_file = os.path.join(input_dir, "queries.csr")
    if os.path.exists(queries_file):
        convert_csr_to_seismic_format(queries_file, data_dir, "queries")
    else:
        print(f"Warning: {queries_file} not found")
    
    # Convert query IDs
    query_ids_file = os.path.join(input_dir, "query_ids.txt")
    query_ids_array = None
    if os.path.exists(query_ids_file):
        print(f"Converting {query_ids_file} to queries_ids.npy")
        with open(query_ids_file, 'r') as f:
            query_ids = [line.strip() for line in f]
        # Convert to numpy array of integers
        query_ids_array = np.array([int(query_id) for query_id in query_ids], dtype=np.int64)
        np.save(os.path.join(data_dir, "queries_ids.npy"), query_ids_array)
        print(f"Saved {len(query_ids_array)} query IDs")
    else:
        print(f"Warning: {query_ids_file} not found")
    
    # Create groundtruth - for v2, doc IDs in groundtruth are already the CSR indices
    if query_ids_array is not None:
        success = create_groundtruth_from_v2_file_direct(
            groundtruth_file, 
            os.path.join(data_dir, "groundtruth.tsv"),
            num_docs
        )
        if not success:
            print("Failed to create groundtruth file!")
            return
    else:
        print("Cannot create groundtruth without query IDs")
        return
    
    # Convert qrels file - need to map string doc IDs to integer indices
    dst = os.path.join(output_dir, "qrels.msmarco_v2.tsv")
    convert_qrels_format_v2_direct(qrels_file, dst)
    print(f"Converted {qrels_file} to {dst}")
    
    print(f"Conversion complete! Files saved to {data_dir}")

if __name__ == "__main__":
    main()