//! Vector Operations Rust Library
//! 
//! High-performance vector operations for the knowledge ingestion system.
//! Provides optimized implementations of mathematical computations including
//! vector normalization, similarity calculations, and arithmetic operations.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int, c_size_t};
use std::slice;
use rayon::prelude::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

/// Error codes for Rust functions
const SUCCESS: c_int = 0;
const ERROR_INVALID_INPUT: c_int = -1;
const ERROR_MEMORY_ALLOCATION: c_int = -2;
const ERROR_COMPUTATION: c_int = -3;
const ERROR_UNKNOWN_OPERATION: c_int = -4;

/// Initialize the library (called from Python)
#[no_mangle]
pub extern "C" fn init_library() -> c_int {
    // Initialize thread pool if needed
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap_or(());
    
    SUCCESS
}

/// Normalize vectors using specified norm type
/// 
/// # Arguments
/// * `input_vectors` - Pointer to input vectors (flattened)
/// * `num_vectors` - Number of vectors
/// * `vector_dim` - Dimension of each vector
/// * `norm_type` - Type of normalization ("l1", "l2", "max")
/// * `output_vectors` - Pointer to output buffer
/// 
/// # Returns
/// * Success code (0 for success, negative for error)
#[no_mangle]
pub extern "C" fn normalize_vectors_f32(
    input_vectors: *const c_float,
    num_vectors: c_size_t,
    vector_dim: c_size_t,
    norm_type: *const c_char,
    output_vectors: *mut c_float,
) -> c_int {
    if input_vectors.is_null() || output_vectors.is_null() || norm_type.is_null() {
        return ERROR_INVALID_INPUT;
    }

    let norm_type_str = unsafe {
        match CStr::from_ptr(norm_type).to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_INPUT,
        }
    };

    let input_slice = unsafe {
        slice::from_raw_parts(input_vectors, num_vectors * vector_dim)
    };
    
    let output_slice = unsafe {
        slice::from_raw_parts_mut(output_vectors, num_vectors * vector_dim)
    };

    // Convert to ndarray for easier manipulation
    let input_array = match Array2::from_shape_vec((num_vectors, vector_dim), input_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };

    let result = match norm_type_str {
        "l1" => normalize_l1(&input_array),
        "l2" => normalize_l2(&input_array),
        "max" => normalize_max(&input_array),
        _ => return ERROR_UNKNOWN_OPERATION,
    };

    let result_array = match result {
        Ok(arr) => arr,
        Err(_) => return ERROR_COMPUTATION,
    };

    // Copy result to output buffer
    output_slice.copy_from_slice(result_array.as_slice().unwrap());

    SUCCESS
}

/// Compute similarity matrix between two sets of vectors
/// 
/// # Arguments
/// * `vectors_a` - Pointer to first set of vectors
/// * `num_vectors_a` - Number of vectors in first set
/// * `vectors_b` - Pointer to second set of vectors
/// * `num_vectors_b` - Number of vectors in second set
/// * `vector_dim` - Dimension of vectors
/// * `metric` - Similarity metric ("cosine", "euclidean", "dot_product")
/// * `output_matrix` - Pointer to output similarity matrix
/// 
/// # Returns
/// * Success code (0 for success, negative for error)
#[no_mangle]
pub extern "C" fn similarity_matrix_f32(
    vectors_a: *const c_float,
    num_vectors_a: c_size_t,
    vectors_b: *const c_float,
    num_vectors_b: c_size_t,
    vector_dim: c_size_t,
    metric: *const c_char,
    output_matrix: *mut c_float,
) -> c_int {
    if vectors_a.is_null() || vectors_b.is_null() || output_matrix.is_null() || metric.is_null() {
        return ERROR_INVALID_INPUT;
    }

    let metric_str = unsafe {
        match CStr::from_ptr(metric).to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_INPUT,
        }
    };

    let vectors_a_slice = unsafe {
        slice::from_raw_parts(vectors_a, num_vectors_a * vector_dim)
    };
    
    let vectors_b_slice = unsafe {
        slice::from_raw_parts(vectors_b, num_vectors_b * vector_dim)
    };
    
    let output_slice = unsafe {
        slice::from_raw_parts_mut(output_matrix, num_vectors_a * num_vectors_b)
    };

    // Convert to ndarray
    let array_a = match Array2::from_shape_vec((num_vectors_a, vector_dim), vectors_a_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };
    
    let array_b = match Array2::from_shape_vec((num_vectors_b, vector_dim), vectors_b_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };

    let result = match metric_str {
        "cosine" => compute_cosine_similarity_matrix(&array_a, &array_b),
        "euclidean" => compute_euclidean_similarity_matrix(&array_a, &array_b),
        "dot_product" => compute_dot_product_matrix(&array_a, &array_b),
        _ => return ERROR_UNKNOWN_OPERATION,
    };

    let result_matrix = match result {
        Ok(matrix) => matrix,
        Err(_) => return ERROR_COMPUTATION,
    };

    // Copy result to output buffer
    output_slice.copy_from_slice(result_matrix.as_slice().unwrap());

    SUCCESS
}

/// Perform vector arithmetic operations
/// 
/// # Arguments
/// * `operation` - Operation type ("add", "subtract", "multiply", "divide", "scale")
/// * `vector_a` - Pointer to first vector set
/// * `vector_b` - Pointer to second vector set (can be null for scalar operations)
/// * `num_vectors` - Number of vectors
/// * `vector_dim` - Dimension of vectors
/// * `scalar` - Scalar value for scalar operations
/// * `output_vectors` - Pointer to output buffer
/// 
/// # Returns
/// * Success code (0 for success, negative for error)
#[no_mangle]
pub extern "C" fn vector_arithmetic_f32(
    operation: *const c_char,
    vector_a: *const c_float,
    vector_b: *const c_float,
    num_vectors: c_size_t,
    vector_dim: c_size_t,
    scalar: c_float,
    output_vectors: *mut c_float,
) -> c_int {
    if operation.is_null() || vector_a.is_null() || output_vectors.is_null() {
        return ERROR_INVALID_INPUT;
    }

    let operation_str = unsafe {
        match CStr::from_ptr(operation).to_str() {
            Ok(s) => s,
            Err(_) => return ERROR_INVALID_INPUT,
        }
    };

    let vector_a_slice = unsafe {
        slice::from_raw_parts(vector_a, num_vectors * vector_dim)
    };
    
    let output_slice = unsafe {
        slice::from_raw_parts_mut(output_vectors, num_vectors * vector_dim)
    };

    let array_a = match Array2::from_shape_vec((num_vectors, vector_dim), vector_a_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };

    let result = if vector_b.is_null() {
        // Scalar operations
        match operation_str {
            "add" => Ok(&array_a + scalar),
            "subtract" => Ok(&array_a - scalar),
            "multiply" | "scale" => Ok(&array_a * scalar),
            "divide" => {
                if scalar.abs() < f32::EPSILON {
                    return ERROR_INVALID_INPUT;
                }
                Ok(&array_a / scalar)
            },
            _ => return ERROR_UNKNOWN_OPERATION,
        }
    } else {
        // Vector operations
        let vector_b_slice = unsafe {
            slice::from_raw_parts(vector_b, num_vectors * vector_dim)
        };
        
        let array_b = match Array2::from_shape_vec((num_vectors, vector_dim), vector_b_slice.to_vec()) {
            Ok(arr) => arr,
            Err(_) => return ERROR_MEMORY_ALLOCATION,
        };

        match operation_str {
            "add" => Ok(&array_a + &array_b),
            "subtract" => Ok(&array_a - &array_b),
            "multiply" => Ok(&array_a * &array_b),
            "divide" => {
                // Element-wise division with zero protection
                let mut result = Array2::zeros((num_vectors, vector_dim));
                for ((i, j), val) in array_b.indexed_iter() {
                    if val.abs() < f32::EPSILON {
                        result[[i, j]] = array_a[[i, j]] / f32::EPSILON;
                    } else {
                        result[[i, j]] = array_a[[i, j]] / val;
                    }
                }
                Ok(result)
            },
            _ => return ERROR_UNKNOWN_OPERATION,
        }
    };

    let result_array = match result {
        Ok(arr) => arr,
        Err(_) => return ERROR_COMPUTATION,
    };

    // Copy result to output buffer
    output_slice.copy_from_slice(result_array.as_slice().unwrap());

    SUCCESS
}

/// Compute batch cosine similarity with top-k results
/// 
/// # Arguments
/// * `query_vectors` - Pointer to query vectors
/// * `num_queries` - Number of query vectors
/// * `database_vectors` - Pointer to database vectors
/// * `num_database` - Number of database vectors
/// * `vector_dim` - Dimension of vectors
/// * `top_k` - Number of top results to return
/// * `similarities` - Pointer to output similarities matrix
/// * `indices` - Pointer to output indices matrix
/// 
/// # Returns
/// * Success code (0 for success, negative for error)
#[no_mangle]
pub extern "C" fn batch_cosine_similarity_f32(
    query_vectors: *const c_float,
    num_queries: c_size_t,
    database_vectors: *const c_float,
    num_database: c_size_t,
    vector_dim: c_size_t,
    top_k: c_int,
    similarities: *mut c_float,
    indices: *mut c_int,
) -> c_int {
    if query_vectors.is_null() || database_vectors.is_null() || 
       similarities.is_null() || indices.is_null() || top_k <= 0 {
        return ERROR_INVALID_INPUT;
    }

    let top_k = top_k as usize;
    if top_k > num_database {
        return ERROR_INVALID_INPUT;
    }

    let query_slice = unsafe {
        slice::from_raw_parts(query_vectors, num_queries * vector_dim)
    };
    
    let database_slice = unsafe {
        slice::from_raw_parts(database_vectors, num_database * vector_dim)
    };
    
    let similarities_slice = unsafe {
        slice::from_raw_parts_mut(similarities, num_queries * top_k)
    };
    
    let indices_slice = unsafe {
        slice::from_raw_parts_mut(indices, num_queries * top_k)
    };

    let query_array = match Array2::from_shape_vec((num_queries, vector_dim), query_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };
    
    let database_array = match Array2::from_shape_vec((num_database, vector_dim), database_slice.to_vec()) {
        Ok(arr) => arr,
        Err(_) => return ERROR_MEMORY_ALLOCATION,
    };

    // Compute full similarity matrix
    let similarity_matrix = match compute_cosine_similarity_matrix(&query_array, &database_array) {
        Ok(matrix) => matrix,
        Err(_) => return ERROR_COMPUTATION,
    };

    // Extract top-k for each query
    for (query_idx, query_similarities) in similarity_matrix.axis_iter(Axis(0)).enumerate() {
        // Get indices sorted by similarity (descending)
        let mut indexed_similarities: Vec<(usize, f32)> = query_similarities
            .iter()
            .enumerate()
            .map(|(idx, &sim)| (idx, sim))
            .collect();
        
        indexed_similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Take top-k
        for (k, &(db_idx, sim)) in indexed_similarities.iter().take(top_k).enumerate() {
            let output_idx = query_idx * top_k + k;
            similarities_slice[output_idx] = sim;
            indices_slice[output_idx] = db_idx as c_int;
        }
    }

    SUCCESS
}

// Helper functions

fn normalize_l1(input: &Array2<f32>) -> Result<Array2<f32>, &'static str> {
    let mut result = input.clone();
    
    result.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let norm = row.iter().map(|x| x.abs()).sum::<f32>();
        if norm > f32::EPSILON {
            row /= norm;
        }
    });
    
    Ok(result)
}

fn normalize_l2(input: &Array2<f32>) -> Result<Array2<f32>, &'static str> {
    let mut result = input.clone();
    
    result.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > f32::EPSILON {
            row /= norm;
        }
    });
    
    Ok(result)
}

fn normalize_max(input: &Array2<f32>) -> Result<Array2<f32>, &'static str> {
    let mut result = input.clone();
    
    result.axis_iter_mut(Axis(0)).into_par_iter().for_each(|mut row| {
        let max_val = row.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        if max_val > f32::EPSILON {
            row /= max_val;
        }
    });
    
    Ok(result)
}

fn compute_cosine_similarity_matrix(
    vectors_a: &Array2<f32>, 
    vectors_b: &Array2<f32>
) -> Result<Array2<f32>, &'static str> {
    // Normalize vectors
    let norm_a = normalize_l2(vectors_a)?;
    let norm_b = normalize_l2(vectors_b)?;
    
    // Compute dot product matrix
    let similarity_matrix = norm_a.dot(&norm_b.t());
    
    Ok(similarity_matrix)
}

fn compute_euclidean_similarity_matrix(
    vectors_a: &Array2<f32>, 
    vectors_b: &Array2<f32>
) -> Result<Array2<f32>, &'static str> {
    let (num_a, dim) = vectors_a.dim();
    let (num_b, _) = vectors_b.dim();
    
    let mut similarity_matrix = Array2::zeros((num_a, num_b));
    
    // Parallel computation of pairwise distances
    similarity_matrix.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            let vec_a = vectors_a.row(i);
            for (j, mut sim) in row.iter_mut().enumerate() {
                let vec_b = vectors_b.row(j);
                let distance = (&vec_a - &vec_b).mapv(|x| x * x).sum().sqrt();
                *sim = 1.0 / (1.0 + distance); // Convert distance to similarity
            }
        });
    
    Ok(similarity_matrix)
}

fn compute_dot_product_matrix(
    vectors_a: &Array2<f32>, 
    vectors_b: &Array2<f32>
) -> Result<Array2<f32>, &'static str> {
    let dot_product_matrix = vectors_a.dot(&vectors_b.t());
    Ok(dot_product_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_normalize_l2() {
        let input = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let result = normalize_l2(&input).unwrap();
        
        // Check that each row has unit norm
        for row in result.axis_iter(Axis(0)) {
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cosine_similarity_matrix() {
        let vectors_a = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let vectors_b = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).unwrap();
        
        let result = compute_cosine_similarity_matrix(&vectors_a, &vectors_b).unwrap();
        
        // First vector should be similar to first vector in b, not similar to second
        assert!((result[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(result[[0, 1]].abs() < 1e-6);
    }
}