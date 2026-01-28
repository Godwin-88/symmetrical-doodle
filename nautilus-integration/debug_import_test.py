#!/usr/bin/env python3

import sys
import traceback

print("Testing imports step by step...")

try:
    print("1. Importing basic modules...")
    import asyncio
    import time
    from pathlib import Path
    from typing import Dict, Any, Optional, List
    from dataclasses import dataclass
    from enum import Enum
    from datetime import datetime
    print("   Basic modules imported successfully")
    
    print("2. Importing logging...")
    from nautilus_integration.core.logging import get_logger
    print("   Logging imported successfully")
    
    print("3. Importing error_handling module...")
    import nautilus_integration.core.error_handling as eh
    print("   Module imported successfully")
    
    print("4. Checking module contents...")
    print(f"   Module attributes: {[x for x in dir(eh) if not x.startswith('_')]}")
    
    print("5. Checking for specific classes...")
    classes_to_check = ['ErrorRecoveryManager', 'GracefulDegradationManager', 'CircuitBreaker', 'RateLimiter']
    for cls_name in classes_to_check:
        if hasattr(eh, cls_name):
            print(f"   ✓ {cls_name} found")
        else:
            print(f"   ✗ {cls_name} NOT found")
    
    print("6. Trying direct import...")
    from nautilus_integration.core.error_handling import ErrorRecoveryManager
    print("   ✓ ErrorRecoveryManager imported successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    print("Traceback:")
    traceback.print_exc()