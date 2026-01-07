#!/usr/bin/env python3

import scipy.sparse as sp
import numpy as np
import struct
import os
from tqdm import tqdm
import argparse

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

def convert_groundtruth_format(input_file, output_file):
    """Convert NQ groundtruth format (comma-separated) to TSV format expected by Seismic"""
    print(f"Converting groundtruth format from {input_file} to {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for query_id, line in enumerate(f_in):
            # Parse comma-separated document IDs
            doc_ids = line.strip().split(',')
            
            # Convert to TSV format: query_id \t doc_id \t rank \t score
            for rank, doc_id_str in enumerate(doc_ids):
                try:
                    # Convert scientific notation to integer
                    doc_id = int(float(doc_id_str))
                    # Write in TSV format: query_id, doc_id, rank (1-based), score (dummy)
                    f_out.write(f"{query_id}\t{doc_id}\t{rank + 1}\t1.0\n")
                except ValueError:
                    # Skip invalid entries
                    continue

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
    parser = argparse.ArgumentParser(description="Convert CSR format to Seismic binary format")
    parser.add_argument("--input-dir", required=True, help="Input directory containing CSR files")
    parser.add_argument("--output-dir", required=True, help="Output directory for binary files")
    
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Create output directory
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Convert documents
    docs_file = os.path.join(input_dir, "nq_docs.csr")
    if os.path.exists(docs_file):
        convert_csr_to_seismic_format(docs_file, data_dir, "documents")
    
    # Convert queries  
    queries_file = os.path.join(input_dir, "nq_queries.csr")
    if os.path.exists(queries_file):
        convert_csr_to_seismic_format(queries_file, data_dir, "queries")
    
    # Copy other files
    for filename in ["nq_doc_ids.txt", "nq_query_ids.txt", "nq_ground_truth.txt", "nq_qrels.tsv"]:
        src = os.path.join(input_dir, filename)
        if os.path.exists(src):
            if filename.endswith('.txt'):
                # Convert text files to numpy arrays
                if 'doc_ids' in filename:
                    ids = np.loadtxt(src, dtype=str)
                    np.save(os.path.join(data_dir, "doc_ids.npy"), ids)
                    print(f"Converted {src} to doc_ids.npy")
                elif 'query_ids' in filename:
                    # Convert query IDs to integers
                    ids = np.loadtxt(src, dtype=str)
                    # Convert string IDs to integers (0, 1, 2, ...)
                    int_ids = np.array([int(id_str) for id_str in ids], dtype=np.int64)
                    np.save(os.path.join(data_dir, "queries_ids.npy"), int_ids)
                    print(f"Converted {src} to queries_ids.npy (as integers)")
                elif 'ground_truth' in filename:
                    # Convert ground truth from comma-separated to TSV format
                    convert_groundtruth_format(src, os.path.join(data_dir, "groundtruth.tsv"))
                    print(f"Converted {src} to groundtruth.tsv")
            elif filename.endswith('.tsv'):
                # Copy qrels file
                import shutil
                dst = os.path.join(output_dir, "qrels.nq.tsv")
                shutil.copy(src, dst)
                print(f"Copied {src} to {dst}")
    
    print(f"Conversion complete! Files saved to {data_dir}")

if __name__ == "__main__":
    main()