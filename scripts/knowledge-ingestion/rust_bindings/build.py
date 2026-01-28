#!/usr/bin/env python3
"""
Build script for Rust vector operations library

This script compiles the Rust library and makes it available for Python FFI.
"""

import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path


def get_library_extension():
    """Get the appropriate library extension for the current platform"""
    system = platform.system().lower()
    if system == "windows":
        return ".dll"
    elif system == "darwin":
        return ".dylib"
    else:
        return ".so"


def check_rust_installation():
    """Check if Rust is installed and available"""
    try:
        result = subprocess.run(
            ["cargo", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        print(f"Found Rust: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Rust/Cargo not found. Please install Rust from https://rustup.rs/")
        return False


def build_rust_library():
    """Build the Rust library"""
    print("Building Rust vector operations library...")
    
    # Change to the rust_bindings directory
    rust_dir = Path(__file__).parent
    original_dir = os.getcwd()
    
    try:
        os.chdir(rust_dir)
        
        # Build in release mode for performance
        result = subprocess.run(
            ["cargo", "build", "--release"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Error building Rust library:")
            print(result.stderr)
            return False
        
        print("Rust library built successfully!")
        return True
        
    except Exception as e:
        print(f"Error during build: {e}")
        return False
    finally:
        os.chdir(original_dir)


def copy_library():
    """Copy the built library to the appropriate location"""
    rust_dir = Path(__file__).parent
    target_dir = rust_dir / "target" / "release"
    
    lib_ext = get_library_extension()
    
    # Find the built library
    if platform.system().lower() == "windows":
        lib_name = "vector_ops.dll"
    else:
        lib_name = f"libvector_ops{lib_ext}"
    
    source_path = target_dir / lib_name
    
    if not source_path.exists():
        print(f"Error: Built library not found at {source_path}")
        return False
    
    # Copy to the rust_bindings directory for easy access
    dest_path = rust_dir / lib_name
    
    try:
        shutil.copy2(source_path, dest_path)
        print(f"Library copied to {dest_path}")
        
        # Also copy to a generic name for easier loading
        generic_name = f"vector_ops{lib_ext}"
        generic_path = rust_dir / generic_name
        shutil.copy2(source_path, generic_path)
        print(f"Library also copied to {generic_path}")
        
        return True
        
    except Exception as e:
        print(f"Error copying library: {e}")
        return False


def run_tests():
    """Run Rust tests"""
    print("Running Rust tests...")
    
    rust_dir = Path(__file__).parent
    original_dir = os.getcwd()
    
    try:
        os.chdir(rust_dir)
        
        result = subprocess.run(
            ["cargo", "test"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Some tests failed:")
            print(result.stderr)
            print(result.stdout)
            return False
        
        print("All Rust tests passed!")
        print(result.stdout)
        return True
        
    except Exception as e:
        print(f"Error running tests: {e}")
        return False
    finally:
        os.chdir(original_dir)


def main():
    """Main build process"""
    print("Building Rust vector operations library for Python FFI")
    print("=" * 60)
    
    # Check if Rust is installed
    if not check_rust_installation():
        sys.exit(1)
    
    # Build the library
    if not build_rust_library():
        print("Build failed!")
        sys.exit(1)
    
    # Copy the library to accessible location
    if not copy_library():
        print("Failed to copy library!")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("Tests failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("Build completed successfully!")
    print("The Rust library is now available for Python FFI.")
    
    # Show library location
    rust_dir = Path(__file__).parent
    lib_ext = get_library_extension()
    lib_path = rust_dir / f"vector_ops{lib_ext}"
    print(f"Library location: {lib_path}")


if __name__ == "__main__":
    main()