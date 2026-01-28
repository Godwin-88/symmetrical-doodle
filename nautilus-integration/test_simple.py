#!/usr/bin/env python3
"""
Simple test for error handling framework
"""

try:
    from nautilus_integration.core.error_handling import (
        NautilusErrorType,
        ErrorSeverity,
        classify_error,
        determine_error_severity
    )
    
    print("✓ Successfully imported error handling components")
    
    # Test error classification
    connection_error = ConnectionError("Connection failed")
    error_type = classify_error(connection_error)
    print(f"✓ Connection error classified as: {error_type}")
    
    # Test error severity determination
    severity = determine_error_severity(error_type, "network")
    print(f"✓ Error severity determined as: {severity}")
    
    print("✓ Basic error handling framework is working!")
    
except Exception as e:
    print(f"✗ Error testing framework: {e}")
    import traceback
    traceback.print_exc()