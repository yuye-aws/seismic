#!/usr/bin/env python3

import numpy as np
import scipy.sparse as sp
import sys

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

def test_csr_loader(filename):
    """Test the CSR loader function"""
    print(f"Testing CSR loader on: {filename}")
    
    try:
        data, indices, indptr, nrow, ncol = read_csr_file(filename)
        
        print(f"Successfully loaded!")
        print(f"Matrix shape: ({nrow}, {ncol})")
        print(f"Non-zeros: {len(data)}")
        print(f"Data dtype: {data.dtype}")
        print(f"Indices dtype: {indices.dtype}")
        print(f"Indptr dtype: {indptr.dtype}")
        
        # Create scipy sparse matrix
        csr_matrix = sp.csr_matrix((data, indices, indptr), shape=(nrow, ncol))
        print(f"Created scipy CSR matrix: {csr_matrix.shape}")
        print(f"Matrix nnz: {csr_matrix.nnz}")
        
        # Show first row info
        first_row = csr_matrix.getrow(0)
        print(f"First row nnz: {first_row.nnz}")
        if first_row.nnz > 0:
            print(f"First row indices: {first_row.indices[:10]}")
            print(f"First row data: {first_row.data[:10]}")
        
        return csr_matrix
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_custom_csr.py <csr_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    matrix = test_csr_loader(filename)
    
    if matrix is not None:
        print("Success! CSR matrix loaded correctly.")
    else:
        print("Failed to load CSR matrix.")