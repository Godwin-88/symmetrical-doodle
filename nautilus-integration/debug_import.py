#!/usr/bin/env python3
"""
Debug import issues
"""

import sys
import os

try:
    # Try to execute the file directly
    with open('src/nautilus_integration/core/error_handling.py', 'r') as f:
        content = f.read()
    
    print(f"File size: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}")
    
    # Try to compile the code
    compiled = compile(content, 'error_handling.py', 'exec')
    print("✓ Code compiled successfully")
    
    # Try to execute it
    namespace = {}
    exec(compiled, namespace)
    print("✓ Code executed successfully")
    
    # Check what's in the namespace
    classes = [name for name in namespace if isinstance(namespace[name], type)]
    print(f"✓ Classes found: {classes}")
    
    if 'NautilusErrorType' in namespace:
        print("✓ NautilusErrorType found in namespace")
    else:
        print("✗ NautilusErrorType not found in namespace")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()