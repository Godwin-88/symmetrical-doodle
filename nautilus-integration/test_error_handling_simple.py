"""Simple test of error handling classes."""

import sys
sys.path.insert(0, 'src')

try:
    from nautilus_integration.core.error_handling import CircuitBreaker, ErrorRecoveryManager
    print("Import successful!")
    print("CircuitBreaker:", CircuitBreaker)
    print("ErrorRecoveryManager:", ErrorRecoveryManager)
except ImportError as e:
    print(f"Import error: {e}")
    
    # Try importing the module directly
    try:
        import nautilus_integration.core.error_handling as eh
        print("Module imported successfully")
        print("Available attributes:", [attr for attr in dir(eh) if not attr.startswith('_')])
    except Exception as e2:
        print(f"Module import error: {e2}")