#!/usr/bin/env python3

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    print("Testing imports...")
    
    # Test basic imports
    from nautilus_integration.core.logging import get_logger
    print("✓ Logging module imported")
    
    # Test error handling imports
    import nautilus_integration.core.error_handling as eh
    print("✓ Error handling module imported")
    
    # Check what's in the module
    print(f"Module contents: {[name for name in dir(eh) if not name.startswith('_')]}")
    
    # Try to import specific classes
    from nautilus_integration.core.error_handling import NautilusErrorType
    print("✓ NautilusErrorType imported")
    
    from nautilus_integration.core.error_handling import ErrorRecoveryManager
    print("✓ ErrorRecoveryManager imported")
    
    print("All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()