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
    
    # Create mapping from doc_id string to integer index
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(unique_doc_ids)}
    
    print(f"Found {len(unique_doc_ids)} unique document IDs")
    return unique_doc_ids, doc_id_to_idx

def convert_qrels_format_v2(input_file, output_file, doc_id_to_idx):
    """Convert MS MARCO v2 qrels format to seismic format with integer doc IDs"""
    print(f"Converting qrels format from {input_file} to {output_file}")
    
    df = pd.read_csv(input_file, sep='\t')
    
    with open(output_file, 'w') as f_out:
        for _, row in df.iterrows():
            query_id = row['query_id']
            doc_id_str = row['doc_id']
            relevance = row['relevance']
            
            # Convert doc_id string to integer index
            doc_id_idx = doc_id_to_idx[doc_id_str]
            
            # Write in expected format: query_id \t 0 \t doc_id_idx \t relevance
            f_out.write(f"{query_id}\t0\t{doc_id_idx}\t{relevance}\n")

def create_groundtruth_from_qrels(qrels_file, output_file, doc_id_to_idx):
    """Create groundtruth.tsv from qrels file for MSMarco v2"""
    print(f"Creating groundtruth file from {qrels_file}")
    
    df = pd.read_csv(qrels_file, sep='\t')
    
    # Group by query_id to get all relevant documents per query
    grouped = df.groupby('query_id')
    
    with open(output_file, 'w') as f_out:
        for query_id, group in grouped:
            # Get all relevant doc IDs for this query
            doc_ids = [doc_id_to_idx[doc_id] for doc_id in group['doc_id'].values]
            
            # Write in TSV format: query_id \t doc_id \t rank \t score
            for rank, doc_id in enumerate(doc_ids):
                f_out.write(f"{query_id}\t{doc_id}\t{rank + 1}\t1.0\n")

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
    
    # First, extract document IDs from qrels
    qrels_file = os.path.join(input_dir, "qrels.tsv")
    if not os.path.exists(qrels_file):
        print(f"Error: {qrels_file} not found")
        return
    
    unique_doc_ids, doc_id_to_idx = extract_doc_ids_from_qrels(qrels_file)
    
    # Save doc_ids.npy
    doc_ids_array = np.array(unique_doc_ids, dtype='<U50')  # Unicode string array
    np.save(os.path.join(data_dir, "doc_ids.npy"), doc_ids_array)
    print(f"Saved {len(doc_ids_array)} document IDs to doc_ids.npy")
    
    # Convert documents using merged passages file
    docs_file = os.path.join(input_dir, "merged_passages.csr")
    if os.path.exists(docs_file):
        print("Converting merged passages file...")
        convert_csr_to_seismic_format(docs_file, data_dir, "documents")
    else:
        print(f"Error: {docs_file} not found")
        return
    
    # Convert queries  
    queries_file = os.path.join(input_dir, "queries.csr")
    if os.path.exists(queries_file):
        convert_csr_to_seismic_format(queries_file, data_dir, "queries")
    else:
        print(f"Warning: {queries_file} not found")
    
    # Convert query IDs
    query_ids_file = os.path.join(input_dir, "query_ids.txt")
    if os.path.exists(query_ids_file):
        print(f"Converting {query_ids_file} to queries_ids.npy")
        with open(query_ids_file, 'r') as f:
            query_ids = [line.strip() for line in f]
        # Convert to numpy array of integers
        query_ids_array = np.array([int(query_id) for query_id in query_ids], dtype=np.int64)
        np.save(os.path.join(data_dir, "queries_ids.npy"), query_ids_array)
        print(f"Saved {len(query_ids_array)} query IDs")
    
    # Create groundtruth from qrels
    create_groundtruth_from_qrels(qrels_file, os.path.join(data_dir, "groundtruth.tsv"), doc_id_to_idx)
    
    # Convert qrels file
    dst = os.path.join(output_dir, "qrels.msmarco_v2.tsv")
    convert_qrels_format_v2(qrels_file, dst, doc_id_to_idx)
    print(f"Converted {qrels_file} to {dst}")
    
    print(f"Conversion complete! Files saved to {data_dir}")

if __name__ == "__main__":
    main()