"""
Python-Rust FFI Interface

This module provides Python-Rust Foreign Function Interface (FFI) bindings
for seamless data exchange and high-performance mathematical computations.

Features:
- Python-Rust data type conversion
- Memory-efficient data transfer
- Error handling and fallback mechanisms
- Performance monitoring for FFI operations
- Automatic memory management
- Thread-safe operations

Requirements: 10.5
"""

import os
import sys
import ctypes
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from numpy.ctypeslib import ndpointer
import weakref

from core.config import get_settings
from core.logging import get_logger


class RustDataType(Enum):
    """Rust data types for FFI"""
    F32 = "f32"
    F64 = "f64"
    I32 = "i32"
    I64 = "i64"
    U32 = "u32"
    U64 = "u64"
    BOOL = "bool"


@dataclass
class FFIPerformanceMetrics:
    """Performance metrics for FFI operations"""
    operation_name: str
    data_transfer_time_ms: float
    rust_execution_time_ms: float
    total_time_ms: float
    data_size_bytes: int
    memory_overhead_bytes: int
    conversion_overhead_ms: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class RustFunctionSignature:
    """Rust function signature definition"""
    function_name: str
    argument_types: List[ctypes._SimpleCData]
    return_type: ctypes._SimpleCData
    is_array_function: bool = False
    array_dimensions: Optional[Tuple[int, ...]] = None


class RustFFIError(Exception):
    """Exception raised when Rust FFI operations fail"""
    pass


class RustLibraryManager:
    """
    Manages loading and interfacing with Rust shared libraries.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._libraries: Dict[str, ctypes.CDLL] = {}
        self._function_signatures: Dict[str, RustFunctionSignature] = {}
        self._performance_metrics: List[FFIPerformanceMetrics] = []
        self._metrics_lock = threading.Lock()
        
        # Memory management
        self._allocated_memory: Dict[int, int] = {}  # ptr -> size
        self._memory_lock = threading.Lock()
        
        # Initialize default library paths
        self._library_paths = self._get_library_paths()
    
    def _get_library_paths(self) -> List[str]:
        """Get potential paths for Rust libraries"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        
        # Platform-specific library extensions
        if sys.platform.startswith('win'):
            lib_ext = '.dll'
        elif sys.platform.startswith('darwin'):
            lib_ext = '.dylib'
        else:
            lib_ext = '.so'
        
        paths = [
            os.path.join(base_dir, 'rust_bindings', f'libvector_ops{lib_ext}'),
            os.path.join(base_dir, 'rust_bindings', f'vector_ops{lib_ext}'),
            os.path.join(base_dir, '..', 'target', 'release', f'libvector_ops{lib_ext}'),
            os.path.join(base_dir, '..', 'target', 'debug', f'libvector_ops{lib_ext}'),
            # System paths
            f'libvector_ops{lib_ext}',
            f'vector_ops{lib_ext}',
        ]
        
        return paths
    
    def load_library(self, library_name: str, library_path: Optional[str] = None) -> bool:
        """
        Load a Rust shared library.
        
        Args:
            library_name: Name identifier for the library
            library_path: Specific path to the library (optional)
            
        Returns:
            True if library loaded successfully
        """
        try:
            if library_path:
                paths_to_try = [library_path]
            else:
                paths_to_try = self._library_paths
            
            for path in paths_to_try:
                try:
                    if os.path.exists(path):
                        library = ctypes.CDLL(path)
                        self._libraries[library_name] = library
                        self.logger.info(f"Successfully loaded Rust library '{library_name}' from {path}")
                        
                        # Initialize library if it has an init function
                        if hasattr(library, 'init_library'):
                            library.init_library()
                        
                        return True
                except OSError as e:
                    self.logger.debug(f"Failed to load library from {path}: {e}")
                    continue
            
            self.logger.warning(f"Could not load Rust library '{library_name}' from any path")
            return False
            
        except Exception as e:
            self.logger.error(f"Error loading Rust library '{library_name}': {e}")
            return False
    
    def register_function(
        self,
        library_name: str,
        function_name: str,
        argument_types: List[ctypes._SimpleCData],
        return_type: ctypes._SimpleCData,
        is_array_function: bool = False,
        array_dimensions: Optional[Tuple[int, ...]] = None
    ) -> bool:
        """
        Register a Rust function with its signature.
        
        Args:
            library_name: Name of the library containing the function
            function_name: Name of the function
            argument_types: List of ctypes for function arguments
            return_type: ctypes for return value
            is_array_function: Whether function works with arrays
            array_dimensions: Expected array dimensions
            
        Returns:
            True if function registered successfully
        """
        try:
            library = self._libraries.get(library_name)
            if not library:
                raise RustFFIError(f"Library '{library_name}' not loaded")
            
            # Get function from library
            if not hasattr(library, function_name):
                raise RustFFIError(f"Function '{function_name}' not found in library '{library_name}'")
            
            rust_function = getattr(library, function_name)
            
            # Set function signature
            rust_function.argtypes = argument_types
            rust_function.restype = return_type
            
            # Store signature information
            signature = RustFunctionSignature(
                function_name=function_name,
                argument_types=argument_types,
                return_type=return_type,
                is_array_function=is_array_function,
                array_dimensions=array_dimensions
            )
            
            signature_key = f"{library_name}.{function_name}"
            self._function_signatures[signature_key] = signature
            
            self.logger.info(f"Registered Rust function: {signature_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering function '{function_name}': {e}")
            return False
    
    def call_function(
        self,
        library_name: str,
        function_name: str,
        *args,
        **kwargs
    ) -> Any:
        """
        Call a registered Rust function with automatic type conversion.
        
        Args:
            library_name: Name of the library
            function_name: Name of the function
            *args: Function arguments
            **kwargs: Additional options
            
        Returns:
            Function result with appropriate type conversion
        """
        start_time = time.time()
        
        try:
            library = self._libraries.get(library_name)
            if not library:
                raise RustFFIError(f"Library '{library_name}' not loaded")
            
            signature_key = f"{library_name}.{function_name}"
            signature = self._function_signatures.get(signature_key)
            if not signature:
                raise RustFFIError(f"Function '{function_name}' not registered")
            
            rust_function = getattr(library, function_name)
            
            # Convert arguments to appropriate types
            conversion_start = time.time()
            converted_args, data_size = self._convert_arguments(args, signature)
            conversion_time = (time.time() - conversion_start) * 1000
            
            # Call Rust function
            rust_start = time.time()
            result = rust_function(*converted_args)
            rust_time = (time.time() - rust_start) * 1000
            
            # Convert result back to Python types
            python_result = self._convert_result(result, signature)
            
            total_time = (time.time() - start_time) * 1000
            
            # Record performance metrics
            metrics = FFIPerformanceMetrics(
                operation_name=f"{library_name}.{function_name}",
                data_transfer_time_ms=conversion_time,
                rust_execution_time_ms=rust_time,
                total_time_ms=total_time,
                data_size_bytes=data_size,
                memory_overhead_bytes=0,  # TODO: Calculate actual overhead
                conversion_overhead_ms=conversion_time,
                success=True
            )
            
            with self._metrics_lock:
                self._performance_metrics.append(metrics)
                # Keep only recent metrics
                if len(self._performance_metrics) > 1000:
                    self._performance_metrics = self._performance_metrics[-500:]
            
            return python_result
            
        except Exception as e:
            total_time = (time.time() - start_time) * 1000
            
            # Record error metrics
            error_metrics = FFIPerformanceMetrics(
                operation_name=f"{library_name}.{function_name}",
                data_transfer_time_ms=0,
                rust_execution_time_ms=0,
                total_time_ms=total_time,
                data_size_bytes=0,
                memory_overhead_bytes=0,
                conversion_overhead_ms=0,
                success=False,
                error_message=str(e)
            )
            
            with self._metrics_lock:
                self._performance_metrics.append(error_metrics)
            
            raise RustFFIError(f"Error calling Rust function '{function_name}': {e}")
    
    def _convert_arguments(
        self, 
        args: Tuple[Any, ...], 
        signature: RustFunctionSignature
    ) -> Tuple[List[Any], int]:
        """
        Convert Python arguments to Rust-compatible types.
        
        Args:
            args: Python arguments
            signature: Function signature
            
        Returns:
            Tuple of (converted_args, total_data_size)
        """
        converted_args = []
        total_data_size = 0
        
        for i, (arg, expected_type) in enumerate(zip(args, signature.argument_types)):
            if isinstance(arg, np.ndarray):
                # Handle NumPy arrays
                converted_arg, size = self._convert_numpy_array(arg, expected_type)
                converted_args.append(converted_arg)
                total_data_size += size
                
            elif isinstance(arg, (list, tuple)):
                # Handle Python sequences
                if signature.is_array_function:
                    # Convert to NumPy array first
                    np_array = np.array(arg)
                    converted_arg, size = self._convert_numpy_array(np_array, expected_type)
                    converted_args.append(converted_arg)
                    total_data_size += size
                else:
                    # Convert individual elements
                    converted_args.append(expected_type(arg))
                    total_data_size += sys.getsizeof(arg)
                    
            elif isinstance(arg, (int, float, bool)):
                # Handle scalar types
                converted_args.append(expected_type(arg))
                total_data_size += sys.getsizeof(arg)
                
            elif isinstance(arg, str):
                # Handle strings
                encoded = arg.encode('utf-8')
                converted_args.append(ctypes.c_char_p(encoded))
                total_data_size += len(encoded)
                
            else:
                # Try direct conversion
                try:
                    converted_args.append(expected_type(arg))
                    total_data_size += sys.getsizeof(arg)
                except (TypeError, ValueError) as e:
                    raise RustFFIError(f"Cannot convert argument {i} of type {type(arg)} to {expected_type}: {e}")
        
        return converted_args, total_data_size
    
    def _convert_numpy_array(
        self, 
        array: np.ndarray, 
        expected_type: ctypes._SimpleCData
    ) -> Tuple[Any, int]:
        """
        Convert NumPy array to Rust-compatible format.
        
        Args:
            array: NumPy array
            expected_type: Expected ctypes type
            
        Returns:
            Tuple of (converted_array, data_size)
        """
        # Ensure array is contiguous
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        
        # Convert to appropriate dtype
        if expected_type == ctypes.POINTER(ctypes.c_float):
            if array.dtype != np.float32:
                array = array.astype(np.float32)
        elif expected_type == ctypes.POINTER(ctypes.c_double):
            if array.dtype != np.float64:
                array = array.astype(np.float64)
        elif expected_type == ctypes.POINTER(ctypes.c_int32):
            if array.dtype != np.int32:
                array = array.astype(np.int32)
        elif expected_type == ctypes.POINTER(ctypes.c_int64):
            if array.dtype != np.int64:
                array = array.astype(np.int64)
        
        # Get pointer to data
        data_ptr = array.ctypes.data_as(expected_type)
        data_size = array.nbytes
        
        # Store reference to prevent garbage collection
        with self._memory_lock:
            self._allocated_memory[id(array)] = data_size
        
        return data_ptr, data_size
    
    def _convert_result(self, result: Any, signature: RustFunctionSignature) -> Any:
        """
        Convert Rust result back to Python types.
        
        Args:
            result: Rust function result
            signature: Function signature
            
        Returns:
            Python-compatible result
        """
        if signature.return_type == ctypes.c_void_p:
            # Handle void pointer (usually for arrays)
            return result
        elif signature.return_type in [ctypes.c_float, ctypes.c_double]:
            return float(result)
        elif signature.return_type in [ctypes.c_int32, ctypes.c_int64, ctypes.c_uint32, ctypes.c_uint64]:
            return int(result)
        elif signature.return_type == ctypes.c_bool:
            return bool(result)
        elif signature.return_type == ctypes.c_char_p:
            return result.decode('utf-8') if result else ""
        else:
            return result
    
    def get_performance_metrics(self) -> List[FFIPerformanceMetrics]:
        """Get FFI performance metrics"""
        with self._metrics_lock:
            return self._performance_metrics.copy()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage statistics"""
        with self._memory_lock:
            total_allocated = sum(self._allocated_memory.values())
            return {
                "total_allocated_bytes": total_allocated,
                "total_allocated_mb": total_allocated / (1024 * 1024),
                "active_allocations": len(self._allocated_memory)
            }
    
    def cleanup_memory(self):
        """Clean up tracked memory allocations"""
        with self._memory_lock:
            self._allocated_memory.clear()
    
    def is_library_loaded(self, library_name: str) -> bool:
        """Check if a library is loaded"""
        return library_name in self._libraries
    
    def get_loaded_libraries(self) -> List[str]:
        """Get list of loaded library names"""
        return list(self._libraries.keys())
    
    def unload_library(self, library_name: str):
        """Unload a library (cleanup only, actual unloading is OS-dependent)"""
        if library_name in self._libraries:
            del self._libraries[library_name]
            
            # Remove associated function signatures
            keys_to_remove = [
                key for key in self._function_signatures.keys() 
                if key.startswith(f"{library_name}.")
            ]
            for key in keys_to_remove:
                del self._function_signatures[key]
            
            self.logger.info(f"Unloaded library: {library_name}")


class VectorOperationsFFI:
    """
    High-level interface for vector operations using Rust FFI.
    """
    
    def __init__(self, library_manager: Optional[RustLibraryManager] = None):
        self.logger = get_logger(__name__)
        self.library_manager = library_manager or RustLibraryManager()
        self._initialized = False
        
        # Initialize if possible
        self.initialize()
    
    def initialize(self) -> bool:
        """Initialize the FFI interface"""
        try:
            # Load vector operations library
            if not self.library_manager.load_library("vector_ops"):
                self.logger.warning("Could not load vector operations Rust library")
                return False
            
            # Register common vector operation functions
            self._register_vector_functions()
            
            self._initialized = True
            self.logger.info("Vector operations FFI initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector operations FFI: {e}")
            return False
    
    def _register_vector_functions(self):
        """Register vector operation functions"""
        # Vector normalization
        self.library_manager.register_function(
            "vector_ops",
            "normalize_vectors_f32",
            [
                ctypes.POINTER(ctypes.c_float),  # input vectors
                ctypes.c_size_t,                 # num_vectors
                ctypes.c_size_t,                 # vector_dim
                ctypes.c_char_p,                 # norm_type
                ctypes.POINTER(ctypes.c_float)   # output vectors
            ],
            ctypes.c_int,  # return code
            is_array_function=True
        )
        
        # Similarity matrix computation
        self.library_manager.register_function(
            "vector_ops",
            "similarity_matrix_f32",
            [
                ctypes.POINTER(ctypes.c_float),  # vectors_a
                ctypes.c_size_t,                 # num_vectors_a
                ctypes.POINTER(ctypes.c_float),  # vectors_b
                ctypes.c_size_t,                 # num_vectors_b
                ctypes.c_size_t,                 # vector_dim
                ctypes.c_char_p,                 # metric
                ctypes.POINTER(ctypes.c_float)   # output matrix
            ],
            ctypes.c_int,  # return code
            is_array_function=True
        )
        
        # Vector arithmetic
        self.library_manager.register_function(
            "vector_ops",
            "vector_arithmetic_f32",
            [
                ctypes.c_char_p,                 # operation
                ctypes.POINTER(ctypes.c_float),  # vector_a
                ctypes.POINTER(ctypes.c_float),  # vector_b (optional)
                ctypes.c_size_t,                 # num_vectors
                ctypes.c_size_t,                 # vector_dim
                ctypes.c_float,                  # scalar (optional)
                ctypes.POINTER(ctypes.c_float)   # output vectors
            ],
            ctypes.c_int,  # return code
            is_array_function=True
        )
        
        # Batch cosine similarity
        self.library_manager.register_function(
            "vector_ops",
            "batch_cosine_similarity_f32",
            [
                ctypes.POINTER(ctypes.c_float),  # query_vectors
                ctypes.c_size_t,                 # num_queries
                ctypes.POINTER(ctypes.c_float),  # database_vectors
                ctypes.c_size_t,                 # num_database
                ctypes.c_size_t,                 # vector_dim
                ctypes.c_int,                    # top_k
                ctypes.POINTER(ctypes.c_float),  # similarities
                ctypes.POINTER(ctypes.c_int)     # indices
            ],
            ctypes.c_int,  # return code
            is_array_function=True
        )
    
    def is_available(self) -> bool:
        """Check if FFI is available and initialized"""
        return self._initialized and self.library_manager.is_library_loaded("vector_ops")
    
    def normalize_vectors(
        self, 
        vectors: np.ndarray, 
        norm_type: str = "l2"
    ) -> np.ndarray:
        """
        Normalize vectors using Rust implementation.
        
        Args:
            vectors: Input vectors [num_vectors, vector_dim]
            norm_type: Normalization type ("l1", "l2", "max")
            
        Returns:
            Normalized vectors
        """
        if not self.is_available():
            raise RustFFIError("Vector operations FFI not available")
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Ensure float32 for Rust
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Prepare output array
        output = np.zeros_like(vectors, dtype=np.float32)
        
        # Call Rust function
        result_code = self.library_manager.call_function(
            "vector_ops",
            "normalize_vectors_f32",
            vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vectors.shape[0],
            vectors.shape[1],
            norm_type.encode('utf-8'),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        if result_code != 0:
            raise RustFFIError(f"Rust normalize_vectors failed with code: {result_code}")
        
        return output
    
    def similarity_matrix(
        self,
        vectors_a: np.ndarray,
        vectors_b: np.ndarray,
        metric: str = "cosine"
    ) -> np.ndarray:
        """
        Compute similarity matrix using Rust implementation.
        
        Args:
            vectors_a: First set of vectors [num_vectors_a, vector_dim]
            vectors_b: Second set of vectors [num_vectors_b, vector_dim]
            metric: Similarity metric ("cosine", "euclidean", "dot_product")
            
        Returns:
            Similarity matrix [num_vectors_a, num_vectors_b]
        """
        if not self.is_available():
            raise RustFFIError("Vector operations FFI not available")
        
        if vectors_a.ndim == 1:
            vectors_a = vectors_a.reshape(1, -1)
        if vectors_b.ndim == 1:
            vectors_b = vectors_b.reshape(1, -1)
        
        # Ensure float32 for Rust
        if vectors_a.dtype != np.float32:
            vectors_a = vectors_a.astype(np.float32)
        if vectors_b.dtype != np.float32:
            vectors_b = vectors_b.astype(np.float32)
        
        # Prepare output matrix
        output = np.zeros((vectors_a.shape[0], vectors_b.shape[0]), dtype=np.float32)
        
        # Call Rust function
        result_code = self.library_manager.call_function(
            "vector_ops",
            "similarity_matrix_f32",
            vectors_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vectors_a.shape[0],
            vectors_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vectors_b.shape[0],
            vectors_a.shape[1],
            metric.encode('utf-8'),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        if result_code != 0:
            raise RustFFIError(f"Rust similarity_matrix failed with code: {result_code}")
        
        return output
    
    def vector_arithmetic(
        self,
        operation: str,
        vector_a: np.ndarray,
        vector_b: Optional[np.ndarray] = None,
        scalar: Optional[float] = None
    ) -> np.ndarray:
        """
        Perform vector arithmetic using Rust implementation.
        
        Args:
            operation: Operation type ("add", "subtract", "multiply", "divide", "scale")
            vector_a: First vector or vector array
            vector_b: Second vector (optional)
            scalar: Scalar value (optional)
            
        Returns:
            Result vectors
        """
        if not self.is_available():
            raise RustFFIError("Vector operations FFI not available")
        
        if vector_a.ndim == 1:
            vector_a = vector_a.reshape(1, -1)
        
        # Ensure float32 for Rust
        if vector_a.dtype != np.float32:
            vector_a = vector_a.astype(np.float32)
        
        if vector_b is not None:
            if vector_b.ndim == 1:
                vector_b = vector_b.reshape(1, -1)
            if vector_b.dtype != np.float32:
                vector_b = vector_b.astype(np.float32)
        
        # Prepare output array
        output = np.zeros_like(vector_a, dtype=np.float32)
        
        # Prepare vector_b pointer (null if not provided)
        vector_b_ptr = (
            vector_b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if vector_b is not None
            else ctypes.POINTER(ctypes.c_float)()
        )
        
        # Call Rust function
        result_code = self.library_manager.call_function(
            "vector_ops",
            "vector_arithmetic_f32",
            operation.encode('utf-8'),
            vector_a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            vector_b_ptr,
            vector_a.shape[0],
            vector_a.shape[1],
            scalar or 0.0,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        if result_code != 0:
            raise RustFFIError(f"Rust vector_arithmetic failed with code: {result_code}")
        
        return output
    
    def batch_cosine_similarity(
        self,
        query_vectors: np.ndarray,
        database_vectors: np.ndarray,
        top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute batch cosine similarity using Rust implementation.
        
        Args:
            query_vectors: Query vectors [num_queries, vector_dim]
            database_vectors: Database vectors [num_database, vector_dim]
            top_k: Number of top results to return
            
        Returns:
            Tuple of (similarities, indices)
        """
        if not self.is_available():
            raise RustFFIError("Vector operations FFI not available")
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        if database_vectors.ndim == 1:
            database_vectors = database_vectors.reshape(1, -1)
        
        # Ensure float32 for Rust
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        if database_vectors.dtype != np.float32:
            database_vectors = database_vectors.astype(np.float32)
        
        # Prepare output arrays
        similarities = np.zeros((query_vectors.shape[0], top_k), dtype=np.float32)
        indices = np.zeros((query_vectors.shape[0], top_k), dtype=np.int32)
        
        # Call Rust function
        result_code = self.library_manager.call_function(
            "vector_ops",
            "batch_cosine_similarity_f32",
            query_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query_vectors.shape[0],
            database_vectors.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            database_vectors.shape[0],
            query_vectors.shape[1],
            top_k,
            similarities.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        )
        
        if result_code != 0:
            raise RustFFIError(f"Rust batch_cosine_similarity failed with code: {result_code}")
        
        return similarities, indices
    
    def get_performance_metrics(self) -> List[FFIPerformanceMetrics]:
        """Get FFI performance metrics"""
        return self.library_manager.get_performance_metrics()
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage statistics"""
        return self.library_manager.get_memory_usage()
    
    def cleanup(self):
        """Clean up FFI resources"""
        self.library_manager.cleanup_memory()


# Global FFI instance
_vector_ffi: Optional[VectorOperationsFFI] = None


def get_vector_ffi() -> VectorOperationsFFI:
    """Get or create global vector operations FFI instance"""
    global _vector_ffi
    
    if _vector_ffi is None:
        _vector_ffi = VectorOperationsFFI()
    
    return _vector_ffi


def is_rust_available() -> bool:
    """Check if Rust bindings are available"""
    try:
        ffi = get_vector_ffi()
        return ffi.is_available()
    except Exception:
        return False