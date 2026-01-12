#!/usr/bin/env python3
"""
MSMarco CoCondenser CSR to Seismic Format Converter

This script converts MSMarco v1 passage CoCondenser dataset from CSR format
to the binary format expected by the Seismic search system.

Usage:
    python convert_msmarco_csr_to_seismic.py --input-dir /path/to/msmarco/data --output-dir /path/to/output

The script expects the following files in the input directory:
- base_46M.csr: The main CSR sparse matrix file
- doc_ids.npy: Document IDs mapping
- queries_ids.npy: Query IDs mapping  
- queries.bin: Query vectors (already in binary format)
- groundtruth.tsv: Ground truth relevance judgments
- token_to_id_mapping.json: Token to ID mapping
- knn_graph_cikm.npy: KNN graph (optional)

The script will create:
- documents.bin: Document vectors in Seismic binary format
- queries.bin: Query vectors (copied from input)
- doc_ids.npy: Document IDs (copied from input)
- queries_ids.npy: Query IDs (copied from input)
- groundtruth.tsv: Ground truth (copied from input)
- token_to_id_mapping.json: Token mapping (copied from input)
- knn_graph_cikm.npy: KNN graph (copied if exists)
"""

import argparse
import json
import numpy as np
import os
import shutil
import struct
import sys
from tqdm import tqdm


def read_sparse_matrix_fields(fname):
    """Read CSR matrix fields without instantiating the full matrix"""
    print(f"Reading CSR matrix from: {fname}")
    
    with open(fname, "rb") as f:
        # Read matrix dimensions and number of non-zero elements
        sizes = np.fromfile(f, dtype="int64", count=3)
        nrow, ncol, nnz = sizes
        
        print(f"Matrix dimensions: {nrow} x {ncol}")
        print(f"Non-zero elements: {nnz}")
        
        # Read row pointers (indptr)
        indptr = np.fromfile(f, dtype="int64", count=nrow + 1)
        assert nnz == indptr[-1], f"NNZ mismatch: expected {nnz}, got {indptr[-1]}"
        
        # Read column indices
        indices = np.fromfile(f, dtype="int32", count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol), "Invalid column indices"
        
        # Read values
        data = np.fromfile(f, dtype="float32", count=nnz)
        
        return data, indices, indptr, ncol, nrow


def convert_csr_to_seismic_format(input_file, output_file):
    """Convert CSR file to Seismic binary format"""
    print(f"Converting CSR to Seismic format...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    
    # Read CSR matrix fields
    data, indices, indptr, ncol, nrow = read_sparse_matrix_fields(input_file)
    
    # Check if component indices can be represented as u16
    max_index = np.max(indices)
    if max_index > 65535:
        print(f"Warning: Component indices exceed u16 range (max: {max_index})")
        print("Writing as u32 instead of u16")
    
    # Write in Seismic binary format
    with open(output_file, "wb") as f:
        # Write number of vectors (documents)
        f.write(struct.pack("<I", nrow))
        
        # Write each document vector
        for i in tqdm(range(nrow), desc="Converting documents"):
            start, end = indptr[i], indptr[i+1]
            n_elements = end - start
            
            # Get current document's indices and values
            doc_indices = indices[start:end]
            doc_values = data[start:end]
            
            # Write number of non-zero elements
            f.write(struct.pack("<I", n_elements))
            
            # Write component indices (as u32 for compatibility)
            for idx in doc_indices:
                f.write(struct.pack("<I", idx))
            
            # Write values (as f32)
            for val in doc_values:
                f.write(struct.pack("<f", val))
    
    print(f"Conversion completed!")
    print(f"Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def copy_file_if_exists(src, dst, description):
    """Copy a file if it exists, with error handling"""
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {description}: {os.path.basename(src)}")
        return True
    else:
        print(f"Warning: {description} not found: {src}")
        return False


def validate_input_directory(input_dir):
    """Validate that required files exist in input directory"""
    required_files = [
        "base_46M.csr",
        "doc_ids.npy", 
        "queries_ids.npy",
        "queries.bin",
        "groundtruth.tsv"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(input_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Error: Missing required files in {input_dir}:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert MSMarco CoCondenser CSR format to Seismic binary format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input-dir", 
        required=True,
        help="Input directory containing MSMarco CSR files"
    )
    
    parser.add_argument(
        "--output-dir",
        required=True, 
        help="Output directory for Seismic format files"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output directory if it exists"
    )
    
    args = parser.parse_args()
    
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print("MSMarco CoCondenser CSR to Seismic Format Converter")
    print("=" * 55)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Validate input directory
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    if not validate_input_directory(input_dir):
        sys.exit(1)
    
    # Create output directory
    if os.path.exists(output_dir):
        if not args.force:
            print(f"Error: Output directory already exists: {output_dir}")
            print("Use --force to overwrite")
            sys.exit(1)
        else:
            print(f"Warning: Overwriting existing output directory")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert main CSR file to documents.bin
    csr_file = os.path.join(input_dir, "base_46M.csr")
    documents_file = os.path.join(output_dir, "documents.bin")
    convert_csr_to_seismic_format(csr_file, documents_file)
    
    print("\nCopying additional files...")
    
    # Copy required files
    files_to_copy = [
        ("doc_ids.npy", "document IDs"),
        ("queries_ids.npy", "query IDs"), 
        ("queries.bin", "query vectors"),
        ("groundtruth.tsv", "ground truth"),
        ("token_to_id_mapping.json", "token mapping"),
        ("knn_graph_cikm.npy", "KNN graph")
    ]
    
    for filename, description in files_to_copy:
        src = os.path.join(input_dir, filename)
        dst = os.path.join(output_dir, filename)
        copy_file_if_exists(src, dst, description)
    
    print(f"\nConversion completed successfully!")
    print(f"Seismic format files saved to: {output_dir}")
    
    # Print summary
    print(f"\nOutput files:")
    for file in os.listdir(output_dir):
        filepath = os.path.join(output_dir, file)
        size_mb = os.path.getsize(filepath) / (1024*1024)
        print(f"  - {file} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()